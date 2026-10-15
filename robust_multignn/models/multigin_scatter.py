"""Scatter-based GINe message passing for edge classification.

Shares multigin.MultiGIN's message passing but replaces every
``torch.sparse.mm`` with index_add over (node, edge, weight) messages, so the
backward pass stays O(nnz * d) instead of materializing a dense [V, E] gradient
when the incidence carries autograd (PRBCD/PGD attacks on AML graphs). The
message-passing layers are numerically equivalent to multigin.MultiGIN up to
floating-point reduction order; the readout differs (per-edge logits here vs.
the node-level head in :mod:`multigin`).

Unlike :mod:`line_graph_gnn_scatter`, MultiGIN keeps the source and destination
incidences separate (it never forms ``B_dst - B_src``), so each sparse matmul
maps to a single scatter and no grad-path splitting is needed.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import BatchNorm
from torch_geometric.typing import OptTensor

from robust_multignn.data import (
    GraphDataBatch,
    compute_batch_ports,
)
from robust_multignn.models.model_utils import (
    _aggregate_to_edges,
    _aggregate_to_nodes,
    _sparse_to_messages,
    seed_edge_mask,
)


def _mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
    )


class MultiGINLayer(nn.Module):
    """Scatter-based single GINe layer; H and Z must already be in hidden_dim space."""

    def __init__(
        self,
        hidden_dim: int,
        eps: float,
        train_eps: bool,
        reverse_mp: bool,
        edge_updates: bool,
    ):
        super().__init__()
        self.reverse_mp = reverse_mp
        self.edge_updates = edge_updates

        if train_eps:
            self.eps_fwd = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer("eps_fwd", torch.tensor(eps))

        self.mlp_fwd = _mlp(hidden_dim)
        self.bn = BatchNorm(hidden_dim)

        if reverse_mp:
            if train_eps:
                self.eps_rev = nn.Parameter(torch.tensor(eps))
            else:
                self.register_buffer("eps_rev", torch.tensor(eps))
            self.mlp_rev = _mlp(hidden_dim)

        if edge_updates:
            self.edge_mlp = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(
        self,
        H: Tensor,
        Z: Tensor,
        src_node: Tensor,
        src_edge: Tensor,
        src_w: Tensor,
        dst_node: Tensor,
        dst_edge: Tensor,
        dst_w: Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> Tuple[Tensor, Tensor]:
        """Compute one GINe message-passing step.

        Args:
            H: Node features of shape (V, hidden_dim).
            Z: Edge features of shape (E, hidden_dim).
            src_node, src_edge, src_w: Source incidence as (node, edge, weight)
                messages (B_src in COO form, V x E).
            dst_node, dst_edge, dst_w: Destination incidence as (node, edge,
                weight) messages (B_dst in COO form, V x E).
            num_nodes: V.
            num_edges: E.

        Returns:
            Tuple (H_new, Z_new).
        """
        # B_src.T @ H  -> [E, d]
        BsT_H = _aggregate_to_edges(src_w, src_node, src_edge, H, num_edges)
        m_fwd = F.relu(BsT_H + Z)
        # B_dst @ m_fwd  -> [V, d]
        Bd_m_fwd = _aggregate_to_nodes(dst_w, dst_node, dst_edge, m_fwd, num_nodes)
        H_fwd = self.mlp_fwd((1 + self.eps_fwd) * H + Bd_m_fwd)

        if self.reverse_mp:
            # B_dst.T @ H  -> [E, d]
            BdT_H = _aggregate_to_edges(dst_w, dst_node, dst_edge, H, num_edges)
            m_rev = F.relu(BdT_H + Z)
            # B_src @ m_rev  -> [V, d]
            Bs_m_rev = _aggregate_to_nodes(src_w, src_node, src_edge, m_rev, num_nodes)
            H_rev = self.mlp_rev((1 + self.eps_rev) * H + Bs_m_rev)
            H_combined = (H_fwd + H_rev) / 2
        else:
            H_combined = H_fwd

        H_new = (H + F.relu(self.bn(H_combined))) / 2

        if self.edge_updates:
            edge_input = torch.cat(
                [
                    _aggregate_to_edges(src_w, src_node, src_edge, H_new, num_edges),
                    _aggregate_to_edges(dst_w, dst_node, dst_edge, H_new, num_edges),
                    Z,
                ],
                dim=-1,
            )
            Z_new = Z + self.edge_mlp(edge_input) / 2
        else:
            Z_new = Z

        return H_new, Z_new


class EdgePairPredictionHead(nn.Module):
    """Seed-edge classifier over ``[h_src, h_dst].relu() ++ z_edge`` features.

    Mirrors :class:`~robust_multignn.models.line_graph_gnn.EdgePredictionHead`
    (same ``e_id``/``input_id`` seed selection) but consumes the MultiGIN edge
    representation of width ``3 * hidden_dim`` instead of the edge features alone.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        train_inds: Tensor,
        val_inds: Tensor,
        test_inds: Tensor,
    ):
        super().__init__()
        self.register_buffer("train_inds", train_inds)
        self.register_buffer("val_inds", val_inds)
        self.register_buffer("test_inds", test_inds)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 50), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(50, 25), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(25, out_dim),
        )

    def forward(
        self, edge_repr: Tensor, batch: GraphDataBatch, split: str
    ) -> Tuple[Tensor, Tensor]:
        split_inds: Tensor = getattr(self, f"{split}_inds")
        mask = seed_edge_mask(split_inds, batch)
        return self.classifier(edge_repr[mask]), batch.y[mask]


class MultiGIN(nn.Module):
    """Scatter-based GINe for edge classification (AML), batch-driven.

    Same interface as
    :class:`~robust_multignn.models.line_graph_gnn_scatter.LineGraphGNNModel`
    (``forward(batch, split, tgt_edge_inc)`` returning seed-edge ``(pred, y)``),
    so it plugs into the same edge-level train/attack routines. Internally it runs
    MultiGIN message passing with ``torch.sparse.mm`` replaced by scatter ops, so
    the backward stays O(nnz * d) when ``tgt_edge_inc`` carries autograd
    (PRBCD/PGD attacks), then reads out per-edge logits from
    ``[h_src, h_dst].relu() ++ z_edge`` using the clean ``batch.edge_index`` for
    endpoints (so a soft/perturbed ``tgt_edge_inc`` cannot corrupt the readout).
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        eps: float,
        train_eps: bool,
        out_dim: int,
        ego_dim: int,
        port_vocab_size: int,
        port_emb_dim: int,
        dropout: float,
        reverse_mp: bool,
        edge_updates: bool,
        train_inds: Tensor,
        val_inds: Tensor,
        test_inds: Tensor,
    ):
        super().__init__()
        self.ego_dim = ego_dim
        self.port_vocab_size = port_vocab_size
        use_ports = bool(port_vocab_size) and bool(port_emb_dim)
        self.use_ports = use_ports
        self.dropout = dropout

        self.port_emb = None
        if use_ports:
            self.port_emb = nn.Embedding(port_vocab_size, port_emb_dim)

        self.node_emb = nn.Linear(in_dim + ego_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim + (2 * port_emb_dim if use_ports else 0), hidden_dim)
        self.layers = nn.ModuleList(
            [
                MultiGINLayer(hidden_dim, eps, train_eps, reverse_mp, edge_updates)
                for _ in range(num_layers)
            ]
        )

        self.classifier = EdgePairPredictionHead(
            hidden_dim, out_dim, dropout, train_inds, val_inds, test_inds
        )

    def forward(
        self,
        batch: GraphDataBatch,
        split: str,
        tgt_edge_inc: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run a forward pass and return seed-edge logits and labels.

        Args:
            batch: Mini-batch from a ``LinkNeighborLoader`` carrying ``x``,
                ``edge_index``, ``edge_attr``, ``src_edge_inc``,
                ``tgt_edge_inc``, ``e_id``, ``input_id`` and ``y``.
            split: One of ``'train'``, ``'val'``, ``'test'`` (selects the seed
                index buffer in the head).
            tgt_edge_inc: Optional override for ``batch.tgt_edge_inc`` (e.g. a
                soft perturbed incidence injected by a PGD/PRBCD attack).

        Returns:
            ``(pred, y)`` where ``pred`` are seed-edge logits of shape
            ``[num_seeds_in_batch, out_dim]`` and ``y`` the aligned labels.
        """
        N, M = batch.num_nodes, batch.num_edges
        edge_index = batch.edge_index

        src_inc = batch.src_edge_inc
        tgt_inc = tgt_edge_inc if tgt_edge_inc is not None else batch.tgt_edge_inc

        # Unpack incidences to (node, edge, weight) messages once and reuse
        # across layers. MultiGIN does not normalize the incidences.
        src_node, src_edge, src_w = _sparse_to_messages(src_inc)
        dst_node, dst_edge, dst_w = _sparse_to_messages(tgt_inc)

        # Edge features: encode raw attrs, optionally with port embeddings.
        # Ports are recomputed from the (possibly perturbed) connectivity here,
        # mirroring the node-level loop, since the edge loop does not provide them.
        edge_attr = batch.edge_attr
        if self.use_ports:
            in_ports, out_ports = compute_batch_ports(
                edge_index, getattr(batch, "timestamps", None), N
            )
            # Ports are the local neighborhood rank of each edge, so a node with
            # >= port_vocab_size edges (e.g. under a PRBCD/PGD attack that piles
            # edges onto a node) produces port IDs past the embedding table and
            # trips a CUDA device-side assert. Clamp into the last row as a backstop.
            in_ports = in_ports.clamp(max=self.port_vocab_size - 1)
            out_ports = out_ports.clamp(max=self.port_vocab_size - 1)
            in_emb = self.port_emb(in_ports)
            out_emb = self.port_emb(out_ports)
            parts = ([edge_attr] if edge_attr is not None else []) + [in_emb, out_emb]
            edge_attr = torch.cat(parts, dim=-1)
        elif edge_attr is None:
            edge_attr = batch.x.new_zeros(M, 0)

        # Node features arrive already ego-augmented by the loader transform
        # (AddEgoIdsForLinkNeighbor); the model just consumes batch.x.
        H = self.node_emb(batch.x)
        Z = self.edge_emb(edge_attr)

        for layer in self.layers:
            H, Z = layer(
                H, Z,
                src_node, src_edge, src_w,
                dst_node, dst_edge, dst_w,
                N, M,
            )
            H = F.dropout(H, p=self.dropout, training=self.training)

        # Edge representation: relu over the (clean) endpoint pair, then append
        # the edge features -> [h_src, h_dst].relu() ++ z_edge -> [E, 3*hidden].
        node_pair = torch.cat([H[edge_index[0]], H[edge_index[1]]], dim=-1).relu()
        edge_repr = torch.cat([node_pair, Z], dim=-1)
        return self.classifier(edge_repr, batch, split)


