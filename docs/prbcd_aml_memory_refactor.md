# PR-BCD Attack on AML Graphs: Memory Bottlenecks and Sparse Refactor

## 1. Problem Statement

| # | Location | Symptom |
|---|---|---|
| P1 | `Block._sample_block_nodes` (prbcd.py:58-68) | Allocates `[num_edges × graph.num_nodes]` float32 weight matrix for `torch.multinomial`. On AML, `graph.num_nodes` is max account-ID (sparse) → ~33–680 GiB at typical block sizes. |
| P2 | `Block.perturb_tgt_edge_inc` (prbcd.py:144-166) | Calls `.to_dense()` on batch incidence `[num_batch_nodes × num_batch_edges]`. At `batch_size=8192` → `[56984 × 582459] ≈ 123 GiB`. At `batch_size=2048` → still ~7.7 GiB. |
| P3 | `LineGraphGNNModel` forward via `torch.sparse.mm(B, x)` | Backward path through grad-tracked `B.values()` materializes a dense `[N × M]` intermediate, reproducing the P2 blow-up during backprop. |
| P4 | `(B_tgt - B_src).T @ BE` in original conv | Sparse subtraction unions the two sparsity patterns; grad-tracked `tgt` entries get mixed with no-grad `src` entries at shared positions, enlarging the backward graph unnecessarily. |

## 2. Root Causes

| # | Cause |
|---|---|
| P1 | `multinomial` requires explicit per-row weight rows; uniform sampling implemented via dense weight matrix rather than direct integer sampling. |
| P2 | "Soft multi-target column blend" written as column-slice assignment on a dense matrix for code simplicity; not required by autograd. |
| P3 | PyTorch's sparse-mm backward (for sparse tensors with grad-tracked values) computes `d_out @ x.T` densely before masking to nonzero positions. |
| P4 | Sparse `-` builds the union of index sets; downstream backward walks both branches. |

## 3. Changes / Fixes

### 3.1 `BlockAML._sample_block_nodes` (override)

| Aspect | Before (parent) | After (override) |
|---|---|---|
| Allocation | `torch.ones([num_edges, n])` | `torch.randint(n, [num_edges, k])` |
| Mechanism | `multinomial(weights, k, replacement=False)` | Vectorized integer draw + sort-based duplicate detection + `randperm` fallback per duplicate row |
| Memory | O(`num_edges × n`) | O(`num_edges × k`) |
| Distinctness guarantee | Always (without replacement) | Initial draw is with-replacement; expected duplicate rate `k(k−1)/2n`; fixed deterministically |
| Sampling domain | `[0, graph.num_nodes)` | `[0, graph.num_nodes)` (unchanged — full ID space, not just active nodes) |

### 3.2 `BlockAML.perturb_tgt_edge_inc` (override)

Direct sparse COO construction; no `to_dense()` on the batch incidence.

| Stage | Parent (dense) | Override (sparse) |
|---|---|---|
| Read | `dense_tgt_inc = inc.to_dense()` (`[B × E]`) | `inc.coalesce().indices(), .values()` |
| Scale `(1 − p)` on active cols | `(1 − p_view) * dense[:, active]` on all B rows | `col_scale = ones(E).index_put(active_batch_pos, 1 − p)` then `scaled_vals = orig_vals * col_scale[orig_cols]` |
| Map candidates to batch rows | `block_nodes.unsqueeze(2) == n_id.view(1,1,-1)` → `[num_active, k, B]` then sum over slots | `searchsorted(sorted(n_id), block_nodes.reshape(-1))`; check `sorted_batch[ins_pos] == cand` |
| New entries | `p * batch_target_probs.T` written into dense slice | `(row, col, val)` triples concatenated to original; `coalesce()` sums duplicates |
| Output | `dense_tgt_inc.to_sparse().coalesce()` | `sparse_coo_tensor(...).coalesce()` |
| Differentiability | Standard dense arithmetic | `index_put`, gather, mul → all differentiable w.r.t. `edge_probs` / `target_logits` |

### 3.3 `models/line_graph_gnn_scatter.py` (new file)

Drop-in replacement for `LineGraphGNNModel`. Replaces every `torch.sparse.mm` with explicit scatter messaging.

| Operation | Sparse-mm form | Scatter form |
|---|---|---|
| `B @ edge_attr` | `torch.sparse.mm(B, edge_attr)` | `msgs = w.unsqueeze(1) * edge_attr[edge_idx]` → `out.index_add_(0, node_idx, msgs)` |
| `B.T @ node_attr` | `torch.sparse.mm(B.t(), node_attr)` | `msgs = w.unsqueeze(1) * node_attr[node_idx]` → `out.index_add_(0, edge_idx, msgs)` |
| `(B_tgt − B_src).T @ BE` | One sparse-mm over the union | Two independent scatters; result: `BtBE_tgt − BtBE_src` |
| Degree normalization | `normalize_incidence` on sparse `B` | `_normalize_weights`: `deg = scatter_add(ones, node_idx)`; multiply `w * deg.pow(-0.5)[node_idx]` |

Gradient path retained: `loss → out → msgs → weights → edge_probs / target_logits`. All ops (`gather`, `mul`, `index_add_`) are built-in autograd primitives.

Largest live tensor at any point: `msgs` of shape `[nnz, d]`.

### 3.4 Validation (`tests/test_perturb_tgt_edge_inc.py`)

| Test | Asserts |
|---|---|
| `test_active_edges_in_batch` (3 seeds) | Output of `BlockAML.perturb_tgt_edge_inc` matches `Block.perturb_tgt_edge_inc` to `atol=1e-8, rtol=1e-7` when state is shared via `load_state`. |
| `test_no_active_edges_returns_original` | Both paths return `batch.tgt_edge_inc` unchanged when no batch edge is in the block. |
| `test_no_candidates_in_batch` | Outputs match when all block candidates are absent from `n_id` (only scaling, no new entries). |
| `test_gradient_flows_through_edge_probs` | `edge_probs.grad` and `target_logits.grad` are non-None and non-zero after backward through the sparse output. |

## 4. Impact / Outcome

### Memory

| Site | Before | After | Conditions |
|---|---|---|---|
| `_sample_block_nodes` allocation | `num_edges × num_nodes × 4 B` | `num_edges × k × 4 B` (output only) | 17k edges × 10M nodes: ~680 GB → ~4 MB |
| `perturb_tgt_edge_inc` working set | `B × E × 4 B` dense | `nnz × 4 B` sparse triples | B=56984, E=582459: ~123 GiB → MB-scale |
| `LineGraphGNN` backward intermediate | `[N × M]` dense | `[nnz × d]` messages | Same order of reduction as above |

### Correctness

- Math equivalence between sparse and dense `perturb_tgt_edge_inc` proven by case-analysis (column-wise blend reduces to scaled original entry + per-slot candidate contributions summed via `coalesce()`).
- Numerical equivalence verified by parametric tests across random seeds.
- Gradient connectivity to `edge_probs` and `target_logits` verified.
- `LineGraphGNNModel` scatter forward matches sparse-mm forward up to floating-point reduction order.

### Scalability

- PR-BCD attack now tractable for AML graphs at any attack ratio bounded only by `O(num_pert)` rather than `O(num_pert × num_nodes)`.
- Forward and backward of the attacked model are O(`nnz × d`) end-to-end; no operation in the attack pipeline depends on `graph.num_nodes × graph.num_edges`.

### Assumptions Introduced

| Assumption | Where | Justification |
|---|---|---|
| `batch.n_id` contains unique node IDs | `searchsorted` mapping | Guaranteed by `NeighborLoader` / `LinkNeighborLoader` |
| `block_nodes[ae, :]` distinct per row | `_sample_block_nodes` | Guaranteed by duplicate-row fallback |
| `tgt_edge_inc` has exactly one nonzero per column | Scaled-original step | True for clean incidence; preserved by perturbation (only adds new entries, never removes the original) |

### Files Touched

| File | Change |
|---|---|
| `robust_multignn/attacks/aml.py` | Added `BlockAML._sample_block_nodes`, `BlockAML.perturb_tgt_edge_inc` overrides |
| `robust_multignn/models/line_graph_gnn_scatter.py` | New scatter-based `LineGraphGNNModel` |
| `tests/test_perturb_tgt_edge_inc.py` | New equivalence + gradient tests |
