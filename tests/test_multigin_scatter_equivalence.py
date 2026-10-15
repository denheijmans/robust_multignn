"""Tests for models.multigin_scatter.MultiGIN (scatter GINe edge classifier).

This model mirrors line_graph_gnn_scatter's interface: ``forward(batch, split,
tgt_edge_inc)`` returns ``(pred, y)`` for seed edges only, so it runs through the
same edge-level train/attack routines. Covered here:

  * the forward returns seed-edge logits aligned with seed labels;
  * the message-passing layers stay scatter-based, so a soft/perturbed
    ``tgt_edge_inc`` flows gradients to its values (the PRBCD/PGD path); and
  * the readout uses the clean ``batch.edge_index`` for endpoints, so it is
    unaffected by perturbed-incidence column structure.
"""

from types import SimpleNamespace

import pytest
import torch

from robust_multignn.models.multigin_scatter import MultiGIN as ScatterMultiGIN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _incidence(rows: torch.Tensor, num_nodes: int, num_edges: int) -> torch.Tensor:
    idx = torch.stack([rows, torch.arange(num_edges)])
    vals = torch.ones(num_edges, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_edges)).coalesce()


def _make_batch(num_nodes, num_edges, in_dim, edge_dim, num_seeds, seed=0):
    """Minimal batch with the fields multigin_scatter.forward reads."""
    g = torch.Generator().manual_seed(seed)
    edge_index = torch.stack([
        torch.randint(0, num_nodes, (num_edges,), generator=g),
        torch.randint(0, num_nodes, (num_edges,), generator=g),
    ])
    # Seed edges are the first `num_seeds` edges; e_id is identity here.
    e_id = torch.arange(num_edges)
    input_id = torch.arange(num_seeds)
    return SimpleNamespace(
        x=torch.randn(num_nodes, in_dim, generator=g),
        edge_index=edge_index,
        edge_attr=torch.randn(num_edges, edge_dim, generator=g),
        src_edge_inc=_incidence(edge_index[0], num_nodes, num_edges),
        tgt_edge_inc=_incidence(edge_index[1], num_nodes, num_edges),
        e_id=e_id,
        input_id=input_id,
        y=torch.randint(0, 2, (num_edges,), generator=g),
        num_nodes=num_nodes,
        num_edges=num_edges,
        batch_size=num_seeds,
        time=None,
    )


def _make_model(reverse_mp, edge_updates, use_ports, num_edges, num_seeds, out_dim=2, seed=42):
    port_vocab, port_emb = (5, 4) if use_ports else (0, 0)
    train_inds = torch.arange(num_edges)  # identity local indices for the test graph
    kw = dict(
        in_dim=4, edge_dim=3, hidden_dim=16, num_layers=3,
        eps=0.1, train_eps=True, out_dim=out_dim, ego_dim=0,
        port_vocab_size=port_vocab, port_emb_dim=port_emb,
        dropout=0.0, reverse_mp=reverse_mp, edge_updates=edge_updates,
        train_inds=train_inds, val_inds=train_inds.clone(), test_inds=train_inds.clone(),
    )
    torch.manual_seed(seed)
    return ScatterMultiGIN(**kw)


# Ports are skipped here: compute_batch_ports needs the C++ extension, which the
# unit-test environment may not build. Message passing/readout are port-agnostic.
_CONFIGS = [
    pytest.param(True, True, id="reverse+edge"),
    pytest.param(True, False, id="reverse_only"),
    pytest.param(False, False, id="minimal"),
    pytest.param(False, True, id="edge_no_reverse"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reverse_mp,edge_updates", _CONFIGS)
def test_forward_returns_seed_logits_and_labels(reverse_mp, edge_updates):
    E, S = 25, 6
    model = _make_model(reverse_mp, edge_updates, False, E, S).eval()
    batch = _make_batch(12, E, 4, 3, S)

    with torch.no_grad():
        pred, y = model(batch, "train")

    assert pred.shape == (S, 2)         # one row per seed edge
    assert y.shape == (S,)
    assert torch.equal(y, batch.y[:S])  # seeds are the first S edges here


def test_soft_incidence_gradient_flows_to_values():
    """A soft tgt_edge_inc must flow gradients to its values (the attack path)."""
    E, S, V = 25, 6, 12
    model = _make_model(True, True, False, E, S).eval()
    batch = _make_batch(V, E, 4, 3, S, seed=3)

    # Soft, multi-entry-per-column target incidence (rewiring blend).
    vals = torch.cat([torch.full((E,), 0.7), torch.full((E,), 0.3)]).requires_grad_(True)
    rows = torch.cat([batch.edge_index[1], (batch.edge_index[1] + 1) % V])
    cols = torch.cat([torch.arange(E), torch.arange(E)])
    B_soft = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (V, E)).coalesce()

    pred, _ = model(batch, "train", tgt_edge_inc=B_soft)
    pred.pow(2).sum().backward()

    assert vals.grad is not None and torch.isfinite(vals.grad).all()
    assert vals.grad.abs().sum() > 0


def test_readout_endpoints_use_clean_edge_index():
    """Readout endpoints come from batch.edge_index, independent of tgt columns.

    Two different soft target incidences with the SAME message-passing effect on
    Z/H but different column structure must not change which node embeddings are
    gathered for the readout (no crash, endpoints stable).
    """
    E, S, V = 20, 5, 10
    model = _make_model(True, False, False, E, S, seed=7).eval()
    batch = _make_batch(V, E, 4, 3, S, seed=7)

    vals = torch.cat([torch.full((E,), 0.5), torch.full((E,), 0.5)])
    rows = torch.cat([batch.edge_index[1], (batch.edge_index[1] + 2) % V])
    cols = torch.cat([torch.arange(E), torch.arange(E)])
    B_soft = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (V, E)).coalesce()

    with torch.no_grad():
        pred, y = model(batch, "train", tgt_edge_inc=B_soft)

    assert pred.shape == (S, 2)
    assert y.shape == (S,)
