from collections.abc import Mapping
from typing import Callable, TypeAlias

import torch
from torch import Tensor

LossCallback: TypeAlias = Callable[[Tensor, Tensor, Tensor], Tensor]


def cycle_loss(pert_adj: Tensor, orig_adj: Tensor, normalize: str = "tanh") -> Tensor:
    def _normalize_sparse(A: Tensor) -> Tensor:
        A = A.coalesce()
        if normalize == "tanh":
            vals = A.values().tanh()
        elif normalize == "clamp":
            vals = A.values().clamp(0.0, 1.0)
        else:
            return A
        return torch.sparse_coo_tensor(A.indices(), vals, A.shape, device=A.device)

    def _traces(A_sparse: Tensor) -> tuple[Tensor, Tensor]:
        A_dense = A_sparse.to_dense()
        # trace(A²) = Σ_ij A_ij·A_ji — O(n²) einsum, no matrix multiply
        t2 = torch.einsum("ij,ji->", A_dense, A_dense)
        # trace(A³) = trace(A²·A): A² via sparse×dense — O(nnz·n) vs O(n³)
        A_sq = torch.sparse.mm(A_sparse, A_dense)
        t3 = torch.einsum("ij,ji->", A_sq, A_dense)
        return t2, t3

    pert_adj = _normalize_sparse(pert_adj)
    orig_adj = _normalize_sparse(orig_adj)

    num_nodes = pert_adj.size(0)
    max_k = 3

    trace_pert_2, trace_pert_3 = _traces(pert_adj)
    with torch.no_grad():
        trace_orig_2, trace_orig_3 = _traces(orig_adj)

    loss = (
        ((trace_pert_2 - trace_orig_2) / 2) ** 2
        + ((trace_pert_3 - trace_orig_3) / 3) ** 2
    )
    return loss / (num_nodes * (max_k - 1))


def _normalize_sparse(adj: Tensor, normalize: str) -> Tensor:
    """Apply value normalization to a sparse adjacency without densifying.

    ``tanh`` and ``clamp(min=0)`` both map 0 -> 0, so the implicit zeros stay
    zero and sparsity is preserved. This is equivalent to applying the
    normalization to the dense matrix in :func:`cycle_loss`.
    """
    adj = adj.coalesce()
    values = adj.values()
    if normalize == "tanh":
        values = values.tanh()
    elif normalize == "clamp":
        values = values.clamp(min=0.0, max=1.0)
    return torch.sparse_coo_tensor(adj.indices(), values, adj.size()).coalesce()


def _flat_indices(indices: Tensor, num_cols: int) -> Tensor:
    """Row-major flattened positions of COO indices (sorted for coalesced input)."""
    return indices[0] * num_cols + indices[1]


def _gather_sparse_values(flat_sorted: Tensor, values: Tensor, query: Tensor) -> Tensor:
    """Values of a coalesced sparse matrix at the queried flat positions (0 if absent)."""
    pos = torch.searchsorted(flat_sorted, query).clamp(max=flat_sorted.numel() - 1)
    hit = flat_sorted[pos] == query
    return torch.where(
        hit, values[pos], torch.zeros((), dtype=values.dtype, device=values.device)
    )


class _SparseTracePowers(torch.autograd.Function):
    """``(trace(A^2), trace(A^3))`` of a sparse ``A`` with exact sparse gradients.

    torch's autograd for ``torch.sparse.mm(sparse, sparse)`` materializes dense
    ``[num_nodes, num_nodes]`` intermediates in the backward pass (~47GB on an
    AML batch), so the traces are computed with a hand-derived backward instead:
    ``d trace(A^k)/dA = k (A^{k-1})^T``, which is only needed at the nonzero
    positions of ``A`` and is therefore a cheap gather.

    The forward needs a single sparse ``A @ A`` product (whose *forward* is
    cheap): ``trace(A^3) = sum_{(i,k) in nnz(A^2)} A^2[i,k] * A[k,i]``, so
    ``A^3`` is never formed either.
    """

    @staticmethod
    def forward(ctx, values: Tensor, indices: Tensor, num_nodes: int):
        adj = torch.sparse_coo_tensor(
            indices, values.detach(), (num_nodes, num_nodes)
        ).coalesce()
        adj2 = torch.sparse.mm(adj, adj).coalesce()

        adj_flat = _flat_indices(adj.indices(), num_nodes)
        adj2_idx = adj2.indices()

        trace2 = adj2.values()[adj2_idx[0] == adj2_idx[1]].sum()
        trace3 = (
            adj2.values()
            * _gather_sparse_values(
                adj_flat, adj.values(), adj2_idx[1] * num_nodes + adj2_idx[0]
            )
        ).sum()

        ctx.save_for_backward(
            adj_flat,
            adj.values(),
            _flat_indices(adj2_idx, num_nodes),
            adj2.values(),
            indices,
        )
        ctx.num_nodes = num_nodes
        return trace2, trace3

    @staticmethod
    def backward(ctx, grad_trace2: Tensor, grad_trace3: Tensor):
        adj_flat, adj_vals, adj2_flat, adj2_vals, indices = ctx.saved_tensors
        num_nodes = ctx.num_nodes
        # d trace(A^k)/dA = k (A^{k-1})^T, gathered at A's stored (i, j)
        # positions, i.e. read A and A^2 at the transposed positions (j, i).
        transposed = indices[1] * num_nodes + indices[0]
        grad_values = grad_trace2 * 2.0 * _gather_sparse_values(
            adj_flat, adj_vals, transposed
        ) + grad_trace3 * 3.0 * _gather_sparse_values(adj2_flat, adj2_vals, transposed)
        return grad_values, None, None


def cycle_loss_sparse(
    pert_adj: Tensor, orig_adj: Tensor, normalize: str = "tanh"
) -> Tensor:
    """Sparse equivalent of :func:`cycle_loss` that never densifies the adjacency.

    Mathematically identical to :func:`cycle_loss`: it penalizes the squared
    difference in the number of closed walks of length ``k`` (``trace(A^k)``)
    between the perturbed and original graph for ``k in {2, 3}``. Traces are
    computed by :class:`_SparseTracePowers`, whose hand-derived backward keeps
    the gradient sparse — torch's own backward through sparse-sparse matmul
    materializes dense ``[num_nodes, num_nodes]`` intermediates and runs out of
    memory on AML-scale batches.

    Usable both as an attack regularizer (gradient w.r.t. the perturbed
    adjacency values) and as a post-hoc metric.
    """
    pert_adj = _normalize_sparse(pert_adj, normalize)
    orig_adj = _normalize_sparse(orig_adj, normalize)

    num_nodes = pert_adj.size(0)
    max_k = 3

    def traces(adj: Tensor) -> tuple[Tensor, Tensor]:
        adj = adj.coalesce()
        return _SparseTracePowers.apply(adj.values(), adj.indices(), adj.size(0))

    pert2, pert3 = traces(pert_adj)
    orig2, orig3 = traces(orig_adj)
    loss = ((pert2 - orig2) / 2.0) ** 2 + ((pert3 - orig3) / 3.0) ** 2
    return loss / (num_nodes * (max_k - 1))


@torch.no_grad()
def cycle_stats(
    pert_adj: Tensor, orig_adj: Tensor, normalize: str = "tanh"
) -> dict[str, float]:
    """Diagnostic-only counts behind :func:`cycle_loss_sparse` (no gradients).

    ``cycle_loss_sparse`` penalizes the *difference* in closed-walk counts
    between perturbed and original graphs; if the batch subgraph has few
    closed walks to begin with (e.g. a mostly tree-like/DAG-like sampled
    neighborhood), both traces stay near zero and the regularizer has little
    signal to work with regardless of its weight. This reports the raw
    ingredients so that can be checked directly instead of inferred from the
    loss value.
    """
    pert_adj = _normalize_sparse(pert_adj.coalesce(), normalize)
    orig_adj = _normalize_sparse(orig_adj.coalesce(), normalize)
    num_nodes = pert_adj.size(0)

    def walk_stats(adj: Tensor) -> tuple[float, float, int]:
        adj2 = torch.sparse.mm(adj, adj).coalesce()
        idx = adj2.indices()
        diag_mask = idx[0] == idx[1]
        trace2 = float(adj2.values()[diag_mask].sum())
        adj_flat = _flat_indices(adj.indices(), num_nodes)
        trace3 = float(
            (
                adj2.values()
                * _gather_sparse_values(
                    adj_flat, adj.values(), idx[1] * num_nodes + idx[0]
                )
            ).sum()
        )
        num_closed_2walk_nodes = int(diag_mask.sum())
        return trace2, trace3, num_closed_2walk_nodes

    pert_trace2, pert_trace3, pert_closed_nodes = walk_stats(pert_adj)
    orig_trace2, orig_trace3, orig_closed_nodes = walk_stats(orig_adj)

    return {
        "num_nodes": num_nodes,
        "pert_trace2": pert_trace2,
        "pert_trace3": pert_trace3,
        "orig_trace2": orig_trace2,
        "orig_trace3": orig_trace3,
        "pert_closed_2walk_nodes": pert_closed_nodes,
        "orig_closed_2walk_nodes": orig_closed_nodes,
    }


def degree_loss(pert_adj: Tensor, orig_adj: Tensor) -> Tensor:
    pert_adj = pert_adj.to_dense()
    orig_adj = orig_adj.to_dense()
    pert_in_deg = pert_adj.sum(dim=0)
    pert_out_deg = pert_adj.sum(dim=1)
    orig_in_deg = orig_adj.sum(dim=0)
    orig_out_deg = orig_adj.sum(dim=1)
    in_loss = torch.mean((pert_in_deg - orig_in_deg) ** 2)
    out_loss = torch.mean((pert_out_deg - orig_out_deg) ** 2)
    return in_loss + out_loss


LOSS_CALLBACKS: dict[str, LossCallback] = {
    "cycle_loss": cycle_loss,
    "cycle_loss_sparse": cycle_loss_sparse,
    "degree_loss": degree_loss,
}


def build_incidence_loss_callbacks(
    regularizers: Mapping[str, float] | None,
) -> list[tuple[LossCallback, float]]:
    if regularizers is None:
        return []

    callbacks: list[tuple[LossCallback, float]] = []
    for name, weight in regularizers.items():
        callback = LOSS_CALLBACKS.get(str(name))
        if callback is None:
            raise ValueError(
                f"Unsupported regularizer term: {name}. Supported terms are: "
                f"{', '.join(sorted(LOSS_CALLBACKS))}"
            )
        if weight != 0.0:
            callbacks.append((callback, weight))

    return callbacks
