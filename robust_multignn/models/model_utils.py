"""Shared scatter helpers for incidence-based message passing.

These replace ``torch.sparse.mm`` with index_add over (node, edge, weight)
messages so the backward pass stays O(nnz * d) instead of materializing a dense
[N, M] gradient when the incidence carries autograd (PRBCD/PGD attacks on AML
graphs). Used by the scatter variants of the line-graph and MultiGIN models.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _sparse_to_messages(B: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Unpack a sparse [N, M] incidence into (node_idx, edge_idx, weights)."""
    B = B.coalesce()
    indices = B.indices()
    return indices[0], indices[1], B.values()


def _normalize_weights(
    weights: Tensor, node_idx: Tensor, num_nodes: int
) -> Tensor:
    """Mirror of line_graph_gnn.normalize_incidence: w_k /= sqrt(deg(row_k))."""
    deg = torch.zeros(num_nodes, device=weights.device, dtype=torch.float32)
    deg.scatter_add_(0, node_idx, torch.ones_like(node_idx, dtype=torch.float32))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    return weights * deg_inv_sqrt[node_idx]


def _aggregate_to_nodes(
    weights: Tensor,
    node_idx: Tensor,
    edge_idx: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
) -> Tensor:
    """``B @ edge_attr`` as scatter-add over (node, edge, weight) messages."""
    msgs = weights.unsqueeze(1) * edge_attr[edge_idx]
    out = torch.zeros(
        num_nodes, edge_attr.size(1),
        device=edge_attr.device, dtype=edge_attr.dtype,
    )
    return out.index_add_(0, node_idx, msgs)


def _aggregate_to_edges(
    weights: Tensor,
    node_idx: Tensor,
    edge_idx: Tensor,
    node_attr: Tensor,
    num_edges: int,
) -> Tensor:
    """``B.T @ node_attr`` as scatter-add over (node, edge, weight) messages."""
    msgs = weights.unsqueeze(1) * node_attr[node_idx]
    out = torch.zeros(
        num_edges, node_attr.size(1),
        device=node_attr.device, dtype=node_attr.dtype,
    )
    return out.index_add_(0, edge_idx, msgs)


def seed_edge_mask(split_inds: Tensor, batch) -> Tensor:
    """Boolean mask over the batch's edges selecting the split's seed edges.

    Uses ``batch.e_id`` (local edge IDs relative to the loader's graph) and
    ``batch.input_id`` (positions of seeds in ``edge_label_index``) to identify
    the seed edges within the sampled subgraph. Shared by the edge-prediction
    heads of the scatter models.
    """
    seed_global_ids = split_inds[batch.input_id.cpu()].to(batch.e_id.device)
    return torch.isin(batch.e_id, seed_global_ids)
