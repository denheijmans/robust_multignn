"""Scatter-based equivalent of line_graph_gnn.LineGraphGNNModel.

Replaces torch.sparse.mm with index_add over (node, edge, weight) messages so
the backward pass stays O(nnz * d) instead of materializing a dense [N, M]
gradient when the incidence carries autograd (PRBCD/PGD attacks on AML graphs).
Forward output matches LineGraphGNNModel up to floating-point reduction order.
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import BatchNorm
from torch_geometric.typing import OptTensor

from robust_multignn.data import GraphDataBatch
from robust_multignn.models.line_graph_gnn import EdgePredictionHead
from robust_multignn.models.model_utils import (
    _aggregate_to_edges,
    _aggregate_to_nodes,
    _normalize_weights,
    _sparse_to_messages,
)


class Edge2EdgePropReverseMPConv(nn.Module):
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
        tgt_node_idx: Tensor,
        tgt_edge_idx: Tensor,
        tgt_weights: Tensor,
        src_node_idx: Tensor,
        src_edge_idx: Tensor,
        src_weights: Tensor,
        edge_attr: Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> Tensor:
        BE_fwd = _aggregate_to_nodes(
            tgt_weights, tgt_node_idx, tgt_edge_idx, edge_attr, num_nodes
        )
        x_in = self.act_fn(self.bn_fwd(self.fwd_lin_edge(BE_fwd)))

        BE_rev = _aggregate_to_nodes(
            src_weights, src_node_idx, src_edge_idx, edge_attr, num_nodes
        )
        x_out = self.act_fn(self.bn_rev(self.rev_lin_edge(BE_rev)))

        BE_combined = x_in + x_out

        # (B_tgt - B_src).T @ BE_combined split into two scatters so the
        # grad-tracked tgt path doesn't get unioned with the no-grad src path.
        BtBE_tgt = _aggregate_to_edges(
            tgt_weights, tgt_node_idx, tgt_edge_idx, BE_combined, num_edges
        )
        BtBE_src = _aggregate_to_edges(
            src_weights, src_node_idx, src_edge_idx, BE_combined, num_edges
        )
        return self.lin_edge(BtBE_tgt - BtBE_src)


class LineGraphGNNModel(nn.Module):
    """Drop-in scatter-based replacement for line_graph_gnn.LineGraphGNNModel."""

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
        self.classifier = EdgePredictionHead(
            hidden_dim, out_dim, train_inds, val_inds, test_inds
        )

    def forward(
        self,
        batch: GraphDataBatch,
        split: str,
        tgt_edge_inc: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        N, M = batch.num_nodes, batch.num_edges

        src_inc = batch.src_edge_inc
        tgt_inc = tgt_edge_inc if tgt_edge_inc is not None else batch.tgt_edge_inc

        src_node, src_edge, src_w = _sparse_to_messages(src_inc)
        tgt_node, tgt_edge, tgt_w = _sparse_to_messages(tgt_inc)

        src_w = _normalize_weights(src_w, src_node, N)
        tgt_w = _normalize_weights(tgt_w, tgt_node, N)

        edge_attr = F.relu(self.bn_in(self.edge_encoder(batch.edge_attr)))

        for conv, bn in zip(self.conv_layers, self.bns):
            edge_attr = conv(
                tgt_node, tgt_edge, tgt_w,
                src_node, src_edge, src_w,
                edge_attr, N, M,
            )
            edge_attr = bn(edge_attr)
            edge_attr = F.relu(edge_attr)
            edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        batch.edge_attr = edge_attr
        return self.classifier(batch, split)
