# PR-BCD Attack on AML: Optimization and Final Attack

The attack has two phases. The first optimizes a **continuous distribution** over possible perturbations using gradients. The second **samples a single discrete perturbation** from that distribution and uses it as the final attack.

## Setup

| Symbol | Meaning |
|---|---|
| `block_edges` | A sampled subset of eligible (test-period) edges. The attack only considers perturbing these. |
| `block_nodes[e, :]` | `k` candidate target nodes per block edge, sampled uniformly from the graph. |
| `edge_probs[e]` ∈ [0,1] | Probability that block edge `e` will be perturbed. Trainable. |
| `target_logits[e, :]` | Logits over the `k` candidate targets for block edge `e`. Trainable. `softmax → target_probs`. |
| `num_pert` | Hard budget: total edges allowed to be perturbed (= `attack_ratio × num_eligible`). |

## Phase 1 — Gradient optimization

**Goal:** find values of `edge_probs` and `target_logits` such that the model's PR-AUC on the test split is minimized.

For each epoch, for each batch (from `LinkNeighborLoader`):

1. **Build a soft incidence**. `perturb_tgt_edge_inc` produces a sparse `tgt_edge_inc` where active block edge columns have fractional entries:
   - Original target row: weight `1 - edge_probs[e]`
   - Candidate target rows in the batch: weight `edge_probs[e] * target_probs[e, j]`

2. **Forward pass**. The model runs as usual but with the soft incidence. Each perturbed edge's feature `edge_attr[e]` is delivered to multiple target nodes in fractional amounts.

3. **Loss**. `CrossEntropyLoss` on the model's predictions for test-period edges in the batch.

4. **Backward**. Gradients flow back through the soft incidence to `edge_probs` and `target_logits`.

5. **Optimizer step**. `AdamW` updates both parameters.

6. **Project `edge_probs`** onto a capped simplex with budget `num_pert` (i.e., `sum(edge_probs) ≤ num_pert`, each value in `[0,1]`). This keeps the soft distribution inside the feasible region.

Between epochs:

- **Resample low-prob block edges**: replace the bottom-half edges (by `edge_probs`) with fresh ones drawn from the eligible pool. Expands the search beyond the initial block.
- **Track best state** by lowest PR-AUC across epochs.
- **Stop resampling and load best state** when PR-AUC has not improved for `patience` epochs.

After the loop ends, the attack has learned:
- A soft distribution `edge_probs` over which edges to perturb.
- A soft distribution `target_probs` over which of the `k` candidates each edge should redirect to.

## Phase 2 — Discrete sampling (final attack)

**Goal:** commit to a single discrete graph edit and return it.

Repeat `sample_size` times:

1. **Hard-pick `num_pert` edges** from `edge_probs` via `multinomial(replacement=False)`.

2. **Hard-pick one target per chosen edge** from `softmax(target_logits)` via `multinomial(num_samples=1)`.

3. **Apply**: `edge_index[1, chosen] = new_target`. Each chosen edge now has exactly one new target node.

4. **Evaluate** the model on this fully discrete perturbed graph; record PR-AUC.

Return the perturbed `edge_index` from the sample with lowest PR-AUC.

## Why two phases

| | Why continuous (Phase 1) | Why discrete (Phase 2) |
|---|---|---|
| **Optimization** | Gradients only flow through continuous variables. The soft incidence is a differentiable surrogate for "which target does this edge point to." | — |
| **Real attack** | — | A graph has discrete edges. The model must be evaluated on a graph it could actually receive at inference. |
| **Sampling stochasticity** | — | The discrete sample is random; running it `sample_size` times finds a strong sample within the learned distribution. |

## One-sentence summary

Phase 1 optimizes a soft distribution over perturbations using gradients through a fractional incidence matrix; Phase 2 draws a single discrete graph edit from that distribution and returns the strongest sample.
