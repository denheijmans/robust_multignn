"""Validates that BlockAML.perturb_tgt_edge_inc produces the same result as
Block.perturb_tgt_edge_inc (the dense reference) given identical block state."""

from types import SimpleNamespace

import torch
import pytest

from robust_multignn.attacks.prbcd import Block
from robust_multignn.attacks.aml import BlockAML


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(num_nodes: int, num_edges: int, seed: int = 0):
    """Minimal graph-like object with the fields Block reads."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    edge_index = torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,), generator=rng),
            torch.randint(0, num_nodes, (num_edges,), generator=rng),
        ]
    )
    return SimpleNamespace(
        num_nodes=num_nodes,
        num_edges=num_edges,
        edge_index=edge_index,
    )


def _make_sparse_tgt_inc(n_id: torch.Tensor, e_id: torch.Tensor, edge_index: torch.Tensor):
    """Build a sparse tgt_edge_inc: one 1.0 per column at the target node's local row.

    Any edge whose target is not in n_id gets no entry (simulates partial batch coverage).
    """
    node_to_local = {int(n): i for i, n in enumerate(n_id)}
    rows, cols = [], []
    for j, eid in enumerate(e_id.tolist()):
        tgt = int(edge_index[1, eid])
        if tgt in node_to_local:
            rows.append(node_to_local[tgt])
            cols.append(j)
    if rows:
        idx = torch.tensor([rows, cols], dtype=torch.long)
        vals = torch.ones(len(rows), dtype=torch.float32)
    else:
        idx = torch.zeros(2, 0, dtype=torch.long)
        vals = torch.zeros(0, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, size=(len(n_id), len(e_id))).coalesce()


def _make_batch(n_id, e_id, edge_index):
    return SimpleNamespace(
        n_id=n_id,
        e_id=e_id,
        tgt_edge_inc=_make_sparse_tgt_inc(n_id, e_id, edge_index),
    )


def _blocks_with_shared_state(graph, block_num_nodes, block_num_edges, num_pert, device, seed=42):
    """Return a (Block, BlockAML) pair with identical internal state."""
    torch.manual_seed(seed)
    block = Block(
        graph=graph,
        size=(block_num_nodes, block_num_edges),
        num_pert=num_pert,
        device=device,
    )
    state = block.save_state()

    eligible = torch.arange(graph.num_edges, device=device)
    block_aml = BlockAML(
        graph=graph,
        size=(block_num_nodes, block_num_edges),
        num_pert=num_pert,
        device=device,
        eligible_edges=eligible,
    )
    block_aml.load_state(state)
    return block, block_aml


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_active_edges_in_batch(seed):
    """Batch contains several block edges: both methods must agree on the blended output."""
    device = torch.device("cpu")
    graph = _make_graph(num_nodes=20, num_edges=50, seed=seed)
    block, block_aml = _blocks_with_shared_state(
        graph, block_num_nodes=6, block_num_edges=12, num_pert=4, device=device, seed=seed
    )

    # Include all block edges in e_id so active is non-empty.
    e_id = block.block_edges.clone()
    n_id = torch.randperm(graph.num_nodes)[:10]
    batch = _make_batch(n_id, e_id, graph.edge_index)

    out_ref = block.perturb_tgt_edge_inc(batch).to_dense()
    out_aml = block_aml.perturb_tgt_edge_inc(batch).to_dense()

    torch.testing.assert_close(out_ref, out_aml, atol=1e-8, rtol=1e-7)


def test_no_active_edges_returns_original():
    """When no batch edge is in the block, both methods return tgt_edge_inc unchanged."""
    device = torch.device("cpu")
    graph = _make_graph(num_nodes=20, num_edges=50, seed=3)
    block, block_aml = _blocks_with_shared_state(
        graph, block_num_nodes=6, block_num_edges=12, num_pert=4, device=device
    )

    # e_id contains no block edges.
    non_block = torch.tensor(
        [i for i in range(graph.num_edges) if i not in block.block_edges.tolist()][:8]
    )
    n_id = torch.randperm(graph.num_nodes)[:8]
    batch = _make_batch(n_id, non_block, graph.edge_index)

    out_ref = block.perturb_tgt_edge_inc(batch).to_dense()
    out_aml = block_aml.perturb_tgt_edge_inc(batch).to_dense()

    torch.testing.assert_close(out_ref, out_aml)


def test_no_candidates_in_batch():
    """Block nodes are chosen so none appear in n_id: only scaling occurs, no new entries."""
    device = torch.device("cpu")
    graph = _make_graph(num_nodes=30, num_edges=60, seed=5)
    block, block_aml = _blocks_with_shared_state(
        graph, block_num_nodes=4, block_num_edges=8, num_pert=3, device=device
    )

    # n_id deliberately excludes all block_nodes.
    all_block_nodes = block.block_nodes.reshape(-1).unique()
    non_candidate_nodes = torch.tensor(
        [i for i in range(graph.num_nodes) if i not in all_block_nodes.tolist()]
    )
    n_id = non_candidate_nodes[:min(6, len(non_candidate_nodes))]
    e_id = block.block_edges[:4]
    batch = _make_batch(n_id, e_id, graph.edge_index)

    out_ref = block.perturb_tgt_edge_inc(batch).to_dense()
    out_aml = block_aml.perturb_tgt_edge_inc(batch).to_dense()

    torch.testing.assert_close(out_ref, out_aml, atol=1e-6, rtol=1e-5)


def test_gradient_flows_through_edge_probs():
    """Gradients must reach edge_probs and target_logits through the sparse path."""
    device = torch.device("cpu")
    graph = _make_graph(num_nodes=20, num_edges=40, seed=9)
    _, block_aml = _blocks_with_shared_state(
        graph, block_num_nodes=5, block_num_edges=10, num_pert=3, device=device
    )

    e_id = block_aml.block_edges[:5]
    n_id = torch.randperm(graph.num_nodes)[:10]
    batch = _make_batch(n_id, e_id, graph.edge_index)

    out = block_aml.perturb_tgt_edge_inc(batch)
    # Use the sparse values as the loss proxy.
    loss = out.coalesce().values().sum()
    loss.backward()

    assert block_aml.edge_probs.grad is not None, "No gradient on edge_probs"
    assert block_aml.target_logits.grad is not None, "No gradient on target_logits"
    assert not torch.all(block_aml.edge_probs.grad == 0), "edge_probs grad is all-zero"
