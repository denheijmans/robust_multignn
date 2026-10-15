"""Tests for attacks.regularizers.cycle_loss_sparse against the dense cycle_loss.

cycle_loss_sparse exists because torch's autograd for sparse-sparse matmul
densifies [num_nodes, num_nodes] intermediates in the backward pass (~47GB on
an AML batch, see the docstring on _SparseTracePowers). It replaces that with
a hand-derived sparse backward, so what needs checking is that the sparse path
computes the same forward value and the same gradient (at the sparse tensor's
nonzero positions) as the dense reference -- including on multigraphs, where
duplicate (i, j) index pairs must be summed by .coalesce() before either path
sees them.
"""

import torch

from robust_multignn.attacks.regularizers import (
    _SparseTracePowers,
    cycle_loss,
    cycle_loss_sparse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_sparse_adj(n, nnz, generator):
    idx = torch.randint(0, n, (2, nnz), generator=generator)
    vals = torch.rand(nnz, generator=generator)
    return torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()


def _assert_dense_sparse_agree(pert_adj, orig_adj, normalize="tanh", atol=1e-5):
    """cycle_loss (dense) and cycle_loss_sparse must agree on value and gradient.

    Gradients are compared only at the sparse tensor's nonzero positions: those
    are the only entries cycle_loss_sparse can produce a gradient for.
    """
    pert_dense_leaf = pert_adj.to_dense().clone().requires_grad_(True)
    orig_dense = orig_adj.to_dense()

    pert_sparse = pert_adj.coalesce()
    sparse_vals = pert_sparse.values().clone().requires_grad_(True)
    pert_sparse_leaf = torch.sparse_coo_tensor(
        pert_sparse.indices(), sparse_vals, pert_sparse.size()
    )

    loss_dense = cycle_loss(
        pert_dense_leaf.to_sparse(), orig_dense.to_sparse(), normalize=normalize
    )
    loss_sparse = cycle_loss_sparse(pert_sparse_leaf, orig_adj, normalize=normalize)

    assert torch.allclose(loss_dense, loss_sparse, atol=atol), (
        f"loss mismatch: dense={loss_dense.item()!r} sparse={loss_sparse.item()!r}"
    )

    loss_dense.backward()
    loss_sparse.backward()

    dense_grad_at_nnz = pert_dense_leaf.grad[
        pert_sparse.indices()[0], pert_sparse.indices()[1]
    ]
    assert torch.allclose(dense_grad_at_nnz, sparse_vals.grad, atol=atol), (
        f"gradient mismatch: max abs diff = "
        f"{(dense_grad_at_nnz - sparse_vals.grad).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_matches_dense_on_random_graph():
    g = torch.Generator().manual_seed(0)
    n = 6
    pert = _rand_sparse_adj(n, 10, g)
    orig = _rand_sparse_adj(n, 10, g)
    _assert_dense_sparse_agree(pert, orig)


def test_matches_dense_on_multigraph_with_duplicate_edges():
    """Duplicate (i, j) index pairs (parallel edges) must be coalesced/summed
    identically by both the dense and sparse paths before the loss is computed.
    """
    n = 6
    idx = torch.tensor(
        [[0, 0, 1, 2, 2, 2, 3, 4], [1, 1, 2, 3, 3, 0, 4, 5]]  # (0,1) x2, (2,3) x2
    )
    vals = torch.rand(idx.shape[1], generator=torch.Generator().manual_seed(1))
    pert = torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()

    g = torch.Generator().manual_seed(2)
    orig = _rand_sparse_adj(n, 9, g)
    _assert_dense_sparse_agree(pert, orig)


def test_matches_dense_with_clamp_normalize():
    g = torch.Generator().manual_seed(3)
    n = 6
    pert = _rand_sparse_adj(n, 12, g)
    orig = _rand_sparse_adj(n, 12, g)
    _assert_dense_sparse_agree(pert, orig, normalize="clamp")


def test_matches_dense_on_larger_random_graph():
    g = torch.Generator().manual_seed(4)
    n = 20
    pert = _rand_sparse_adj(n, 60, g)
    orig = _rand_sparse_adj(n, 60, g)
    _assert_dense_sparse_agree(pert, orig)


def test_sparse_trace_powers_gradcheck():
    """Analytic check of the hand-derived backward (d trace(A^k)/dA = k(A^{k-1})^T)
    against torch's numerical Jacobian, in float64 for the precision gradcheck needs.
    """
    n = 5
    idx = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 0, 4]])
    vals = torch.rand(
        idx.shape[1], dtype=torch.float64, generator=torch.Generator().manual_seed(5)
    ).requires_grad_(True)

    def f(v):
        trace2, trace3 = _SparseTracePowers.apply(v, idx, n)
        return trace2 + trace3

    assert torch.autograd.gradcheck(f, (vals,), eps=1e-6, atol=1e-4)


def test_backward_avoids_dense_intermediate():
    """Regression guard for the original OOM: backward must not scale like a
    dense [num_nodes, num_nodes] materialization. Runs a graph too large for a
    dense intermediate to be built accidentally without the test itself OOMing
    (5000x5000 float32 ~= 100MB per buffer, several of which the old dense
    sparse-mm backward needed simultaneously).
    """
    n = 5000
    nnz = 30000
    g = torch.Generator().manual_seed(6)
    idx = torch.randint(0, n, (2, nnz), generator=g)
    vals = torch.rand(nnz, generator=g).requires_grad_(True)
    pert = torch.sparse_coo_tensor(idx, vals, (n, n))

    orig = _rand_sparse_adj(n, nnz, torch.Generator().manual_seed(7))

    loss = cycle_loss_sparse(pert, orig)
    loss.backward()

    assert torch.isfinite(loss)
    assert vals.grad is not None
    assert vals.grad.shape == (nnz,)
    assert torch.isfinite(vals.grad).all()
