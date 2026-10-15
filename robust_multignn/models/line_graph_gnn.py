from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import BatchNorm
from torch_geometric.typing import OptTensor

from robust_multignn.data import GraphDataBatch
from robust_multignn.models.model_utils import seed_edge_mask


def normalize_incidence(B, num_nodes):
    B = B.coalesce()
    row, _ = B.indices()
    vals = B.values()

    deg = torch.zeros(num_nodes, device=B.device)
    deg = deg.scatter_add(0, row, torch.ones_like(row, dtype=torch.float))

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

    norm_vals = vals * deg_inv_sqrt[row]

    B_norm = torch.sparse_coo_tensor(
        B.indices(),
        norm_vals,
        B.size(),
        device=B.device
    )
    return B_norm


class EdgePredictionHead(nn.Module):
    """Selects seed edges from a sampled subgraph and classifies them.

    Seed edges are identified via ``batch.e_id`` (original edge IDs as set by
    :class:`~torch_geometric.loader.LinkNeighborLoader`) compared against the
    per-split local index tensors stored at construction time.  This avoids
    prepending artificial ID columns to ``edge_attr``.

    Parameters
    ----------
    in_dim : int
        Input feature dimension (= model hidden dim).
    out_dim : int
        Output logit dimension (1 for binary classification).
    train_inds : Tensor
        Local edge indices of training-period edges in the training loader's
        underlying graph (typically ``torch.arange(n_train)``).
    val_inds : Tensor
        Local edge indices of validation-period edges in the val loader's
        underlying graph (``torch.arange(n_train, n_train + n_val)``).
    test_inds : Tensor
        Local edge indices of test-period edges in the test loader's underlying
        graph (``torch.arange(n_train + n_val, n_train + n_val + n_test)``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        train_inds: Tensor,
        val_inds: Tensor,
        test_inds: Tensor,
    ):
        super().__init__()
        self.register_buffer("train_inds", train_inds)
        self.register_buffer("val_inds", val_inds)
        self.register_buffer("test_inds", test_inds)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def _seed_mask(self, batch: GraphDataBatch, split: str) -> Tensor:
        """Boolean mask over ``batch.edge_attr`` selecting the seed edges."""
        split_inds: Tensor = getattr(self, f"{split}_inds")
        return seed_edge_mask(split_inds, batch)

    def forward(self, batch: GraphDataBatch, split: str) -> Tuple[Tensor, Tensor]:
        mask = self._seed_mask(batch, split)
        return self.classifier(batch.edge_attr[mask]), batch.y[mask]



class Edge2EdgePropReverseMPConv(nn.Module):
    """Edge-to-edge convolution layer that incorporates both incoming and outgoing messages.

    Parameters
    ----------
    in_dim : int
        The dimensionality of input edge features.
    out_dim : int
        The dimensionality of output edge features.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fwd_lin_edge = nn.Linear(in_dim, out_dim, bias=True)
        self.rev_lin_edge = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_edge = nn.Linear(out_dim, out_dim, bias=True)
        self.bn_fwd = BatchNorm(in_dim)
        self.bn_rev = BatchNorm(in_dim)
        self.act_fn = nn.ReLU()

    def forward(
        self,
        x: Tensor,
        src_edge_inc: Tensor,
        tgt_edge_inc: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """Forward pass for Edge2EdgePropReverseMPConv.

        Parameters
        ----------
        x : Tensor
            Node feature matrix ``[num_nodes, in_dim]``.
            Not updated here; kept for interface compatibility.
        src_edge_inc : Tensor
            Source-edge incidence matrix ``[num_nodes, num_edges]`` (sparse).
        tgt_edge_inc : Tensor
            Target-edge incidence matrix ``[num_nodes, num_edges]`` (sparse).
        edge_attr : Tensor
            Edge feature matrix ``[num_edges, in_dim]``.

        Returns
        -------
        Tensor
            Updated edge features ``[num_edges, out_dim]``.
        """
        # Aggregate edge features at each node via incoming / outgoing edges
        BE_fwd = torch.sparse.mm(tgt_edge_inc, edge_attr)          # [N, d]
        x_in = self.act_fn(self.bn_fwd(self.fwd_lin_edge(BE_fwd)))

        BE_rev = torch.sparse.mm(src_edge_inc, edge_attr)           # [N, d]
        x_out = self.act_fn(self.bn_rev(self.rev_lin_edge(BE_rev)))

        BE_combined = x_in + x_out                                  # [N, d]

        # Signed incidence B = tgt - src; propagate node messages back to edges
        B = tgt_edge_inc - src_edge_inc
        BtBE = torch.sparse.mm(B.t(), BE_combined)                  # [M, d]

        return self.lin_edge(BtBE)                                   # [M, d]


class LineGraphGNNModel(nn.Module):
    """Edge-level GNN using line-graph message passing.

    Parameters
    ----------
    in_dim : int
        Node feature dimension (accepted but unused; kept for interface parity).
    hidden_dim : int
        Hidden edge-feature dimension used throughout the network.
    edge_dim : int
        Raw edge-feature dimension (input to the encoder).
    out_dim : int
        Output logit dimension (1 for binary classification).
    num_layers : int
        Number of :class:`Edge2EdgePropReverseMPConv` layers.
    dropout : float
        Dropout probability applied after each conv layer.
    train_inds : Tensor
        Local edge indices for training-period seeds (see :class:`EdgePredictionHead`).
    val_inds : Tensor
        Local edge indices for validation-period seeds.
    test_inds : Tensor
        Local edge indices for test-period seeds.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        edge_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        train_inds: Tensor,
        val_inds: Tensor,
        test_inds: Tensor,
    ):
        super().__init__()
        self.dropout = dropout

        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.bn_in = BatchNorm(hidden_dim)

        self.conv_layers = nn.ModuleList(
            [Edge2EdgePropReverseMPConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [BatchNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.classifier = EdgePredictionHead(hidden_dim, out_dim, train_inds, val_inds, test_inds)

    def forward(
        self,
        batch: GraphDataBatch,
        split: str,
        tgt_edge_inc: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for LineGraphGNNModel.

        Parameters
        ----------
        batch : GraphDataBatch
            Mini-batch from :class:`~torch_geometric.loader.LinkNeighborLoader`.
            Must carry ``e_id``, ``input_id``, ``edge_label``, ``edge_attr``,
            ``src_edge_inc`` and ``tgt_edge_inc``.
        split : str
            One of ``'train'``, ``'val'``, ``'test'``.  Used by
            :class:`EdgePredictionHead` to look up the correct seed-index buffer.
        tgt_edge_inc : Tensor, optional
            Override for ``batch.tgt_edge_inc``; used by PGD-style attacks to
            inject a soft-perturbed incidence matrix.

        Returns
        -------
        Tuple[Tensor, Tensor]
            ``(pred, y)`` where ``pred`` are seed-edge logits of shape
            ``[num_seeds_in_batch, out_dim]`` and ``y`` are aligned seed-edge
            labels.
        """
        x = batch.x
        src_inc = batch.src_edge_inc
        tgt_inc = tgt_edge_inc if tgt_edge_inc is not None else batch.tgt_edge_inc

        src_inc = normalize_incidence(src_inc, batch.num_nodes)
        tgt_inc = normalize_incidence(tgt_inc, batch.num_nodes)
        
        edge_attr = batch.edge_attr

        # Encode raw edge features
        edge_attr = F.relu(self.bn_in(self.edge_encoder(edge_attr)))

        # Stacked edge-to-edge conv blocks
        for conv, bn in zip(self.conv_layers, self.bns):
            edge_attr = conv(x, src_inc, tgt_inc, edge_attr)
            edge_attr = bn(edge_attr)
            edge_attr = F.relu(edge_attr)
            edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        # Write updated features back so EdgePredictionHead can slice them
        batch.edge_attr = edge_attr

        return self.classifier(batch, split)
