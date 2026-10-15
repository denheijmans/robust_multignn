"""AML (anti-money-laundering) variants of the structural attacks.

Mirrors the non-AML attacks in
:mod:`robust_multignn.attacks.uniform`,
:mod:`robust_multignn.attacks.prbcd`
but uses :class:`~torch_geometric.loader.LinkNeighborLoader` (via
:func:`robust_multignn.data.build_edge_neighbor_loader`), the edge-level
attack/validation epoch helpers (:func:`robust_multignn.utils.attack_edge_epoch`,
:func:`robust_multignn.utils.val_edge_epoch`), :class:`torch.nn.CrossEntropyLoss`,
and restricts the structural perturbation budget to test-period edges
(``graph.split_mask``).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator, List

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange

import wandb
from robust_multignn.aml.exchange_rates import (
    _AMOUNT_COL,
    convert_rewired_edge_currencies,
)
from robust_multignn.aml.utils import (
    criminal_accounts,
    illicit_edges,
    index_to_mask,
    mask_to_index,
)
from robust_multignn.attacks.prbcd import Block, _resolve_block_size
from robust_multignn.data import GraphData, GraphDataBatch, build_edge_neighbor_loader
from robust_multignn.utils import (
    TanhMarginLoss,
    attack_edge_epoch,
    compute_binary_metrics,
    log_epoch_metrics,
    val_edge_epoch,
)

_BYTES_PER_FLOAT32 = 4
_BYTES_PER_GIB = 1024**3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_eligible(graph: GraphData, eval_inds: Tensor | None) -> Tensor:
    """Return the indices of edges eligible for perturbation.

    Defaults to ``mask_to_index(graph.split_mask)`` (the test-period edges in a
    cumulative AML graph) when ``eval_inds`` is not provided.
    """
    if eval_inds is not None:
        return eval_inds
    if getattr(graph, "split_mask", None) is None:
        raise ValueError(
            "AML attacks require either an explicit `eval_inds` tensor or a "
            "`graph.split_mask` to determine which edges are eligible for "
            "perturbation."
        )
    return mask_to_index(graph.split_mask)


THREAT_MODELS = ("unrestricted", "criminal")


def _resolve_threat_model(
    graph: GraphData,
    eval_inds: Tensor,
    threat_model: str,
) -> tuple[Tensor, Tensor | None]:
    """Which edges the attacker may rewire, and onto which accounts.

    * ``"unrestricted"``: every edge in ``eval_inds``, onto any account
      (``candidate_pool=None``). The original setting.
    * ``"criminal"``: the launderers rewire only their own transactions --
      the *illicit* edges in ``eval_inds`` -- and only onto accounts they
      control: every account that sent or received an illicit transaction in
      any period of ``graph`` (see :func:`criminal_accounts`).

    The budget is a fraction of the returned eligible set, so under the
    criminal model ``attack_ratio`` is a fraction of the illicit edges.
    """
    if threat_model == "unrestricted":
        return eval_inds, None
    if threat_model != "criminal":
        raise ValueError(
            f"Unknown threat_model {threat_model!r}; expected one of {THREAT_MODELS}"
        )
    y = graph.y.cpu()
    eligible = illicit_edges(eval_inds.cpu(), y)
    if eligible.numel() == 0:
        raise ValueError("threat_model='criminal' but the split has no illicit edges")
    return eligible, criminal_accounts(graph.edge_index.cpu(), y)


@contextmanager
def _temporary_seed_split(model: nn.Module, name: str, inds: Tensor) -> Iterator[None]:
    """Let the model resolve seeds of a loader restricted to ``inds``.

    The edge heads map a batch's seeds as ``{split}_inds[batch.input_id]``,
    where ``input_id`` indexes the loader's seed list. A loader built with a
    ``seed_mask`` subset therefore needs that subset's indices under the split
    name it is called with. They are set as a plain attribute -- not a buffer
    -- so they never reach ``state_dict``, and removed again on exit.
    """
    head = model.classifier
    attr = f"{name}_inds"
    if hasattr(head, attr):
        raise ValueError(f"model.classifier already has {attr!r}")
    device = next(model.parameters()).device
    setattr(head, attr, inds.to(device))
    try:
        yield
    finally:
        delattr(head, attr)


def _sample_uniform_targets_for_edges(
    edge_index: Tensor,
    pert_edge_idx: Tensor,
    num_nodes: int,
    device: torch.device,
    candidate_pool: Tensor | None = None,
) -> Tensor:
    """Uniformly sample new target nodes per edge, excluding current src/tgt.

    Targets are drawn from all ``num_nodes`` accounts, or only from
    ``candidate_pool`` when given (the criminal threat model).
    """
    src = edge_index[0, pert_edge_idx].to(device)
    tgt = edge_index[1, pert_edge_idx].to(device)

    new_tgt = torch.empty_like(tgt)
    if candidate_pool is not None:
        pool = candidate_pool.to(device)
        for i in range(int(pert_edge_idx.numel())):
            valid = pool[(pool != src[i]) & (pool != tgt[i])]
            if valid.numel() == 0:
                raise ValueError(
                    "candidate_pool has no account other than the edge's own endpoints"
                )
            new_tgt[i] = valid[torch.randint(valid.numel(), (1,), device=device)]
        return new_tgt

    for i in range(int(pert_edge_idx.numel())):
        s_i = int(src[i])
        t_i = int(tgt[i])
        while True:
            candidate = torch.randint(num_nodes, (1,), device=device).item()
            if candidate != s_i and candidate != t_i:
                new_tgt[i] = candidate
                break
    return new_tgt


def _eval_loader_params(loader_params: dict[str, Any]) -> dict[str, Any]:
    """Loader params used for validation/sampling: identical except shuffle off."""
    return {**loader_params, "shuffle": False}


def _apply_currency_conversion(
    graph: GraphData,
    pert_edge_idx: Tensor,
    new_targets: Tensor,
    rate_matrix: Tensor | None,
    generator: torch.Generator | None = None,
) -> None:
    """Update ``graph.edge_attr``/``graph.currency`` in place so rewired edges
    stay plausible at their new target node (see
    :func:`robust_multignn.aml.exchange_rates.convert_rewired_edge_currencies`).

    A no-op if ``rate_matrix`` is ``None`` (e.g. currency-unaware callers, or
    datasets without a built rate table) -- ``graph.edge_attr``/``currency``
    are left exactly as they were, matching pre-existing behaviour.

    The index tensors arrive on the model's device while ``graph`` stays on
    CPU (``experiment_attack`` only moves the model), so they are moved onto
    the graph's device before being used to index into it.
    """
    if rate_matrix is None:
        return
    device = graph.edge_attr.device
    graph.edge_attr, graph.currency = convert_rewired_edge_currencies(
        edge_attr=graph.edge_attr,
        currency=graph.currency,
        pert_edge_idx=pert_edge_idx.to(device),
        new_targets=new_targets.to(device),
        node_currency_support=graph.node_currency_support,
        rate_matrix=rate_matrix.to(device),
        edge_attr_mean=graph.edge_attr_mean,
        edge_attr_std=graph.edge_attr_std,
        generator=generator,
    )



@torch.no_grad()
def sample_perturbed_edge_index_from_block_aml(
    model: nn.Module,
    graph: GraphData,
    block: Block,
    num_pert: int,
    sample_size: int,
    loader_params: dict[str, Any],
    eval_inds: Tensor,
    seed_split: str = "test",
    rate_matrix: Tensor | None = None,
) -> Tensor:
    """Sample hard perturbations from a PR-BCD block without densifying targets.

    ``Block.to_full_probs`` materialises a ``[num_edges, num_nodes]`` target
    matrix, which is infeasible on AML graphs whose node-id space is much
    larger than the number of active nodes. This helper keeps the block-sized
    target logits and samples edge probabilities directly in block coordinates.

    Sampling over the block (rather than a scattered ``num_edges``-long
    distribution) also keeps the multinomial input under torch's 2**24 category
    cap, which a full-graph distribution would exceed on large AML graphs.

    ``rate_matrix``, if given, makes rewired edges that were already
    cross-currency consistent with their new target node's currency before
    each candidate is scored (see :func:`_apply_currency_conversion`); the
    winning candidate's converted ``edge_attr``/``currency`` are left on
    ``graph`` when this returns. This only affects which hard sample is
    selected and what its features look like -- the block's learned
    ``edge_probs``/``target_logits`` (the actual PR-BCD optimization) are
    untouched.
    """
    device = next(model.parameters()).device
    eval_loader_kwargs = _eval_loader_params(loader_params)

    learned_target_probs = torch.softmax(block.target_logits, dim=1)

    best_pr_auc = float("inf")
    best_edge_index: Tensor | None = None
    best_edge_attr: Tensor | None = None
    best_currency: Tensor | None = None
    base_edge_index = graph.edge_index.clone()
    base_edge_attr = graph.edge_attr.clone()
    base_currency = graph.currency.clone()

    for _ in trange(sample_size, leave=False, desc="Perturbation Samples"):
        edge_index = base_edge_index.clone().to(device)
        graph.edge_attr = base_edge_attr.clone()
        graph.currency = base_currency.clone()

        # Sample block-local positions directly from the block edge probs, then
        # map to global edge ids. block.edge_probs has length block_num_edges
        # (<= n_eligible), so it never approaches the multinomial category cap.
        local_pos = torch.multinomial(
            block.edge_probs,
            num_pert,
            replacement=False,
        )
        pert_edge_idx = block.block_edges[local_pos]

        pert_node_idx = block.block_nodes[local_pos].gather(
            1,
            learned_target_probs[local_pos].multinomial(num_samples=1),
        ).squeeze(1)
        edge_index[1, pert_edge_idx] = pert_node_idx
        _apply_currency_conversion(graph, pert_edge_idx, pert_node_idx, rate_matrix)

        graph.edge_index = edge_index.cpu()
        loader = build_edge_neighbor_loader(
            graph,
            eval_inds=eval_inds,
            bidirectional=False,
            ego_dim=getattr(model, "ego_dim", 0),
            **eval_loader_kwargs,
        )
        _, pr_auc, _, _ = val_edge_epoch(model, loader, split=seed_split)

        if pr_auc < best_pr_auc:
            best_pr_auc = pr_auc
            best_edge_index = edge_index
            best_edge_attr = graph.edge_attr
            best_currency = graph.currency

    assert best_edge_index is not None, "sample_size must be >= 1"
    graph.edge_attr = best_edge_attr
    graph.currency = best_currency

    num_pert_actual = (best_edge_index != base_edge_index.to(device)).sum().item()
    assert num_pert_actual == num_pert, (
        f"Expected {num_pert} perturbed edges, but got {num_pert_actual}. "
        "This indicates an issue with the sampling process."
    )

    return best_edge_index


# ---------------------------------------------------------------------------
# Uniform attack
# ---------------------------------------------------------------------------


def uniform_attack_aml(
    model: nn.Module,
    graph: GraphData,
    attack_ratio: float,
    sample_size: int,
    loader_params: dict[str, Any],
    eval_inds: Tensor | None = None,
    regularizers: List[tuple] | None = None,
    rate_matrix: Tensor | None = None,
    threat_model: str = "unrestricted",
    **_,
) -> Tensor:
    """Perform a uniform structural attack on the AML test graph, with the
    perturbation budget restricted to test-period edges.

    ``regularizers`` is accepted for a uniform call signature with the
    gradient-based attacks but is a no-op here: the uniform attack samples
    perturbations rather than optimising a differentiable objective, so there is
    no loss to regularise. Structural change is still measured post-hoc via the
    ``test/cycle_loss`` / ``test/degree_loss`` metrics in ``experiment_attack``.

    Parameters
    ----------
    model : nn.Module
        Edge-level model to attack (e.g. ``LineGraphGNNModel``).
    graph : GraphData
        Cumulative AML test graph (must carry ``split_mask`` if ``eval_inds``
        is not provided).
    attack_ratio : float
        Fraction of test-period edges to perturb. ``0`` returns ``graph.edge_index``
        unchanged.
    sample_size : int
        Number of i.i.d. samples drawn from the (uniform) perturbation
        distribution; the one with lowest PR-AUC is returned.
    loader_params : dict
        ``LinkNeighborLoader`` kwargs (forwarded to
        :func:`robust_multignn.data.build_edge_neighbor_loader`).
    eval_inds : Tensor, optional
        Local edge indices eligible for perturbation. Defaults to
        ``mask_to_index(graph.split_mask)``.
    rate_matrix : Tensor, optional
        ``[num_currencies, num_currencies]`` exchange rate lookup (see
        :func:`robust_multignn.aml.exchange_rates.build_dataset_rate_matrix`).
        If given, rewired edges that were already cross-currency are made
        consistent with their new target node's currency before each
        candidate is scored; the winner's converted ``edge_attr``/``currency``
        are left on ``graph``. If omitted, ``edge_attr``/``currency`` are
        left untouched by rewiring, matching prior behaviour.
    threat_model : str, optional
        ``"unrestricted"`` (default): any eligible edge onto any account.
        ``"criminal"``: only illicit edges, onto criminal accounts, with the
        budget a fraction of the illicit edges (see
        :func:`_resolve_threat_model`).

    Returns
    -------
    Tensor
        Perturbed ``edge_index`` of shape ``[2, num_edges]``.
    """
    device = next(model.parameters()).device

    if attack_ratio == 0.0:
        return graph.edge_index

    eval_inds = _resolve_eligible(graph, eval_inds)
    eligible, candidate_pool = _resolve_threat_model(graph, eval_inds, threat_model)
    eligible = eligible.to(device)
    num_eligible = int(eligible.numel())

    num_pert = max(1, int(attack_ratio * num_eligible))
    logging.info(
        f"Perturbation budget: {num_pert}/{num_eligible} "
        f"{'illicit ' if candidate_pool is not None else ''}test-period edges"
    )

    eval_loader_kwargs = _eval_loader_params(loader_params)
    best_pr_auc = float("inf")
    best_edge_index: Tensor | None = None
    best_edge_attr: Tensor | None = None
    best_currency: Tensor | None = None
    base_edge_index = graph.edge_index.clone()
    base_edge_attr = graph.edge_attr.clone()
    base_currency = graph.currency.clone()

    for _ in trange(sample_size, leave=False, desc="Perturbation Samples"):
        edge_index = base_edge_index.clone().to(device)
        graph.edge_attr = base_edge_attr.clone()
        graph.currency = base_currency.clone()

        perm = torch.randperm(num_eligible, device=device)[:num_pert]
        pert_edge_idx = eligible[perm]
        new_targets = _sample_uniform_targets_for_edges(
            edge_index=edge_index,
            pert_edge_idx=pert_edge_idx,
            num_nodes=graph.num_nodes,
            device=device,
            candidate_pool=candidate_pool,
        )
        edge_index[1, pert_edge_idx] = new_targets
        _apply_currency_conversion(graph, pert_edge_idx, new_targets, rate_matrix)

        graph.edge_index = edge_index.cpu()
        loader = build_edge_neighbor_loader(
            graph,
            eval_inds=eval_inds,
            bidirectional=False,
            ego_dim=getattr(model, "ego_dim", 0),
            **eval_loader_kwargs,
        )
        _, pr_auc, _, _ = val_edge_epoch(model, loader, split="test")
        if pr_auc < best_pr_auc:
            best_pr_auc = pr_auc
            best_edge_index = edge_index
            best_edge_attr = graph.edge_attr
            best_currency = graph.currency

    assert best_edge_index is not None, "sample_size must be >= 1"
    graph.edge_attr = best_edge_attr
    graph.currency = best_currency

    num_pert_actual = (best_edge_index != base_edge_index.to(device)).sum().item()
    assert num_pert_actual == num_pert, (
        f"Expected {num_pert} perturbed edges, but got {num_pert_actual}. "
        "This indicates an issue with the sampling process."
    )
    return best_edge_index


# ---------------------------------------------------------------------------
# PR-BCD attack
# ---------------------------------------------------------------------------


class BlockAML(Block):
    """:class:`Block` restricted to a subset of eligible edges.

    Overrides :meth:`Block.sample` and :meth:`Block.resample` to draw block
    edges only from ``eligible_edges`` (typically the test-period edges of the
    cumulative AML graph). Target masking and soft-incidence construction are
    inherited unchanged; final hard sampling uses
    :func:`sample_perturbed_edge_index_from_block_aml` instead of
    :meth:`Block.to_full_probs` so AML graphs are not densified to
    ``[num_edges, num_nodes]``.
    """

    def __init__(
        self,
        graph: GraphData,
        size: tuple[int, int],
        num_pert: int,
        device: torch.device,
        eligible_edges: Tensor,
        rate_matrix: Tensor | None = None,
        candidate_pool: Tensor | None = None,
    ):
        # Set eligible_edges before super().__init__ since the parent's
        # constructor calls self.sample() (which is the overridden version).
        # Store them sorted so resample() can map kept block edges back into
        # eligible-edge coordinates via searchsorted. Ordering is irrelevant
        # elsewhere (sample() draws a random permutation over the pool).
        self.eligible_edges = torch.sort(eligible_edges.to(device)).values
        if int(self.eligible_edges.numel()) < num_pert:
            raise ValueError(
                f"Number of eligible edges ({int(self.eligible_edges.numel())}) is "
                f"less than the perturbation budget ({num_pert})."
            )
        # Only set when the attack optimizes against currency-aware features
        # (see perturb_edge_attr); None leaves the optimization exactly as it
        # was before currency conversion existed.
        self.rate_matrix = rate_matrix.to(device) if rate_matrix is not None else None
        # Accounts the new targets are drawn from (criminal threat model);
        # None draws from every node, as before.
        self.candidate_pool = (
            candidate_pool.to(device) if candidate_pool is not None else None
        )
        super().__init__(graph=graph, size=size, num_pert=num_pert, device=device)

    def _sample_block_nodes(self, num_edges: int) -> Tensor:
        if self.candidate_pool is not None:
            return self._sample_block_nodes_from_pool(num_edges)
        k, n = self.block_num_nodes, self.graph.num_nodes
        samples = torch.randint(n, (num_edges, k), device=self.device)
        # Fix the rare rows that contain duplicate node IDs (expected rate: k(k-1)/2n).
        sorted_s, _ = samples.sort(dim=1)
        dup_rows = (sorted_s[:, 1:] == sorted_s[:, :-1]).any(dim=1).nonzero(as_tuple=True)[0]
        for idx in dup_rows:
            samples[idx] = torch.randperm(n, device=self.device)[:k]
        return samples

    def _sample_block_nodes_from_pool(self, num_edges: int, chunk: int = 256) -> Tensor:
        """``block_num_nodes`` distinct candidates per edge, all from the pool.

        With ``k >= |pool|`` every edge gets the whole pool -- the setting for
        the small criminal pools, where the search covers every account the
        attacker controls. Otherwise each edge draws its own random subset;
        rows are processed in chunks so the ``[rows, |pool|]`` random matrix
        stays small on large pools.
        """
        pool = self.candidate_pool
        k, p = self.block_num_nodes, int(pool.numel())
        if k >= p:
            return pool.unsqueeze(0).expand(num_edges, p).clone()
        rows = [
            pool[torch.rand(min(chunk, num_edges - start), p, device=self.device)
                 .argsort(dim=1)[:, :k]]
            for start in range(0, num_edges, chunk)
        ]
        return torch.cat(rows, dim=0)

    def perturb_tgt_edge_inc(self, batch: GraphData) -> Tensor:
        target_probs = torch.softmax(self.target_logits, dim=1)
        batch_edge_lookup = self.edge_lookup[batch.e_id]
        active = batch_edge_lookup >= 0
        if not active.any():
            return batch.tgt_edge_inc

        inc = batch.tgt_edge_inc.coalesce()
        orig_rows = inc.indices()[0]
        orig_cols = inc.indices()[1]
        orig_vals = inc.values()

        local_edge_pos = batch_edge_lookup[active]
        active_batch_pos = active.nonzero(as_tuple=True)[0]
        edge_probs_active = self.edge_probs[local_edge_pos]

        # Scale active columns by (1 - p) in-place on the sparse values.
        # index_put is differentiable w.r.t. edge_probs_active.
        col_scale = torch.ones(batch.e_id.numel(), dtype=orig_vals.dtype, device=self.device)
        col_scale = col_scale.index_put((active_batch_pos,), 1.0 - edge_probs_active)
        scaled_vals = orig_vals * col_scale[orig_cols]

        # New entries: map each candidate node to its batch row via searchsorted.
        # Memory: O(num_active * k), no dense [B × E] matrix.
        edge_block_nodes = self.block_nodes[local_edge_pos]  # [num_active, k]
        edge_target_probs = target_probs[local_edge_pos]     # [num_active, k]
        batch_node_ids = batch.n_id.to(self.device)          # [B]

        sorted_batch, sort_idx = batch_node_ids.sort()
        flat_candidates = edge_block_nodes.reshape(-1)
        ins_pos = torch.searchsorted(sorted_batch, flat_candidates).clamp(
            max=sorted_batch.numel() - 1
        )
        in_batch = sorted_batch[ins_pos] == flat_candidates
        valid_idx = in_batch.nonzero(as_tuple=True)[0]

        if valid_idx.numel() > 0:
            k = self.block_num_nodes
            ae_idx = valid_idx // k
            new_col = active_batch_pos[ae_idx]
            new_row = sort_idx[ins_pos[valid_idx]]
            new_val = edge_probs_active[ae_idx] * edge_target_probs.reshape(-1)[valid_idx]

            all_rows = torch.cat([orig_rows, new_row])
            all_cols = torch.cat([orig_cols, new_col])
            all_vals = torch.cat([scaled_vals, new_val])
        else:
            all_rows, all_cols, all_vals = orig_rows, orig_cols, scaled_vals

        return torch.sparse_coo_tensor(
            torch.stack([all_rows, all_cols]),
            all_vals,
            size=inc.size(),
            device=self.device,
        ).coalesce()

    def perturb_edge_attr(self, batch: GraphDataBatch) -> Tensor:
        """Scale ``Amount Received`` by the conversion the perturbation implies.

        ``perturb_tgt_edge_inc`` soft-perturbs *where* an edge points, but an
        edge has a single feature row shared by all k candidate targets, so
        the optimizer never sees that rewiring would change the amount. This
        folds that in, using the same weights as the incidence perturbation:

            amount_i *= (1 - p_i) + p_i * sum_j q_ij * ratio_ij

        where ``ratio_ij`` converts the amount into the currency candidate j
        would receive it in. ``p`` and ``q`` are the learnable block
        parameters, so the gradient reaches ``edge_probs``/``target_logits``
        and the search becomes currency-aware.

        Two approximations, both inherent to keeping one row per edge:
        the blended amount is a probability-weighted mixture and so is not a
        realizable transaction (the hard-sampling path applies the exact
        per-candidate conversion instead), and the currency *column* is left
        alone because averaging categorical ids would name a currency no
        candidate uses.
        """
        edge_attr = batch.edge_attr
        if self.rate_matrix is None:
            return edge_attr  # currency-unaware optimization: original formulation

        # Block membership: which of this batch's edges the attack may rewire.
        batch_edge_lookup = self.edge_lookup[batch.e_id]
        active = batch_edge_lookup >= 0
        if not active.any():
            return edge_attr

        local_edge_pos = batch_edge_lookup[active]      # rows of the block tensors
        active_batch_pos = active.nonzero(as_tuple=True)[0]  # rows of batch.edge_attr

        p = self.edge_probs[local_edge_pos]                       # [n] P(rewire edge)
        q = torch.softmax(self.target_logits[local_edge_pos], 1)  # [n,k] P(pick candidate)
        candidates = self.block_nodes[local_edge_pos]             # [n,k] candidate targets

        sent = batch.currency[active_batch_pos, 0]      # [n] currency sent in
        recv = batch.currency[active_batch_pos, 1]      # [n] currency received in today

        support = self.graph.node_currency_support.to(self.device)
        cand_supported = support[candidates]            # [n,k,C] currencies each candidate takes
        # Candidates that already handle this edge's currency imply no change.
        already_ok = cand_supported.gather(2, recv.view(-1, 1, 1).expand(-1, q.size(1), 1)).squeeze(2)
        has_any = cand_supported.any(dim=2)             # [n,k] candidate has any history?

        # Draw each candidate's target currency uniformly from the ones it
        # handles -- the same rule the hard path uses, so the optimizer is
        # steered by the conversion the attack will actually perform.
        weights = cand_supported.to(edge_attr.dtype).reshape(-1, cand_supported.size(2))
        # multinomial rejects all-zero rows (candidates with no history); give
        # them a placeholder, their draw is discarded by `has_any` below.
        weights = torch.where(
            weights.sum(dim=1, keepdim=True) > 0, weights, torch.ones_like(weights)
        )
        chosen = torch.multinomial(weights, num_samples=1).reshape(cand_supported.shape[:2])

        # The amount implies a sent amount of amount/old_rate; re-applying the
        # candidate's rate to it gives amount * (new_rate/old_rate).
        old_rate = self.rate_matrix[sent, recv].unsqueeze(1)                    # [n,1]
        new_rate = self.rate_matrix[sent.unsqueeze(1).expand_as(chosen), chosen]  # [n,k]
        # Convert only where it is both needed and possible; else leave as-is.
        usable = (old_rate != 0) & (new_rate != 0) & has_any & (~already_ok)
        ratio = torch.where(usable, new_rate / old_rate.clamp(min=1e-12),
                            torch.ones_like(new_rate))          # [n,k]

        expected_ratio = (q * ratio).sum(dim=1)         # [n] average over candidates
        blended_ratio = (1.0 - p) + p * expected_ratio  # [n] mix with "stays put"; 1.0 at p=0

        # edge_attr is z-scored, so convert in raw space and re-normalize.
        mean_a = batch.edge_attr_mean[0, _AMOUNT_COL]
        std_a = batch.edge_attr_std[0, _AMOUNT_COL]

        edge_attr = edge_attr.clone()
        raw = edge_attr[active_batch_pos, _AMOUNT_COL] * std_a + mean_a
        edge_attr[active_batch_pos, _AMOUNT_COL] = (raw * blended_ratio - mean_a) / std_a
        return edge_attr

    def sample(self) -> None:
        """Sample a fresh block of edges drawn only from the eligible set."""
        n_eligible = int(self.eligible_edges.numel())
        block_n_edges = min(self.block_num_edges, n_eligible)
        perm = torch.randperm(n_eligible, device=self.device)
        self.block_edges = self.eligible_edges[perm[:block_n_edges]]
        self.block_nodes = self._sample_block_nodes(self.block_num_edges)
        self.edge_probs = torch.zeros(
            (self.block_num_edges,),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        self.target_logits = torch.zeros(
            (self.block_num_edges, self.block_num_nodes),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        self._build_edge_lookup()
        self._mask_block_targets(
            torch.arange(self.block_num_edges, device=self.device),
            reset_values=False,
        )

    def resample(self) -> None:
        """Resample low-probability block edges from the eligible pool."""
        num_nonzero_edges = (self.edge_probs > 0).sum().item()
        if num_nonzero_edges < self.edge_majority:
            return

        resample_idx = torch.topk(
            self.edge_probs,
            self.edge_majority,
            largest=False,
        ).indices

        keep_mask = torch.ones(
            self.block_num_edges,
            device=self.device,
            dtype=torch.bool,
        )
        keep_mask[resample_idx] = False

        # Draw replacements uniformly from the eligible pool, excluding the
        # eligible edges already kept in the block. We work in eligible-edge
        # coordinates (a length-``n_eligible`` mask) rather than a dense
        # ``num_edges``-long distribution: block_edges store global edge ids, but
        # sampling over a full-graph pool would exceed torch.multinomial's 2**24
        # category cap on large graphs (e.g. the medium AML dataset).
        n_eligible = int(self.eligible_edges.numel())
        eligible_mask = torch.ones(
            n_eligible, device=self.device, dtype=torch.bool
        )

        # block_edges is always a subset of the sorted eligible_edges, so
        # searchsorted recovers each kept edge's exact position in the pool.
        kept_pos = torch.searchsorted(
            self.eligible_edges, self.block_edges[keep_mask]
        )
        eligible_mask[kept_pos] = False

        candidate_pos = eligible_mask.nonzero(as_tuple=True)[0]
        choice = torch.randperm(
            candidate_pos.numel(), device=self.device
        )[: len(resample_idx)]
        new_edges = self.eligible_edges[candidate_pos[choice]]

        with torch.no_grad():
            self.edge_probs[resample_idx] = 0
            self.target_logits[resample_idx] = 0
            self.block_edges[resample_idx] = new_edges
            self.block_nodes[resample_idx] = self._sample_block_nodes(
                len(resample_idx)
            )
            self._mask_block_targets(resample_idx, reset_values=True)
            self._build_edge_lookup()


def prbcd_attack_aml(
    model: nn.Module,
    graph: GraphData,
    max_epochs: int,
    lr: float,
    attack_ratio: float,
    sample_size: int,
    class_names: List[str],
    loader_params: dict[str, Any],
    max_resamples: int,
    eval_inds: Tensor | None = None,
    block_size: tuple[int, int] | None = None,
    min_epochs: int = 0,
    patience: int = 0,
    display_interval: int | None = None,
    seed_split: str = "test",
    regularizers: List[tuple] | None = None,
    rate_matrix: Tensor | None = None,
    currency_aware_optimization: bool = False,
    threat_model: str = "unrestricted",
    **_,
) -> Tensor:
    """Perform a PR-BCD structural attack on the AML test graph, with the
    perturbation budget restricted to test-period edges.

    Mirrors :mod:`robust_multignn.attacks.prbcd` but plugs in :class:`BlockAML`
    (eligible-edge-restricted block sampler), the edge-level loader/epoch
    helpers, ``CrossEntropyLoss``, and PR-AUC-based best tracking.

    ``rate_matrix``, if given, is forwarded to the final hard-sampling step
    (:func:`sample_perturbed_edge_index_from_block_aml`), so each hard
    candidate is currency-converted before being scored.

    ``currency_aware_optimization`` additionally feeds expected-value,
    currency-converted features into the optimization loop via
    :meth:`BlockAML.perturb_edge_attr`, so the learned
    ``edge_probs``/``target_logits`` account for what rewiring does to an
    edge's amount. Off by default: with it off the optimization is bit-for-bit
    the original formulation, which is what reproducing earlier results
    requires.

    ``threat_model="criminal"`` restricts the attack to the launderers' own
    moves (see :func:`_resolve_threat_model`): only illicit edges are
    rewired, only onto criminal accounts, and ``block_size`` is clamped to
    those pools. The objective becomes evasion: the tanh margin
    (:class:`~robust_multignn.utils.TanhMarginLoss`) maximized over the
    illicit edges only, so the optimization loop iterates just those seeds.
    PR-AUC is undefined on illicit-only batches, so the best epoch is the one
    with the highest loss, as in Geisler et al. (2021). The final hard sample
    is still chosen by the lowest PR-AUC over the whole split.
    """
    if attack_ratio == 0.0:
        return graph.edge_index

    device = next(model.parameters()).device

    eval_inds = _resolve_eligible(graph, eval_inds)
    eligible, candidate_pool = _resolve_threat_model(graph, eval_inds, threat_model)
    criminal = candidate_pool is not None
    eligible = eligible.to(device)
    num_eligible = int(eligible.numel())

    num_pert = max(1, int(attack_ratio * num_eligible))
    logging.info(
        f"Perturbation budget: {num_pert}/{num_eligible} "
        f"{'illicit ' if criminal else ''}test-period edges"
    )

    # Resolve and clamp block size against the eligible pool rather than the
    # full graph (the parent helper uses graph.num_edges), and the candidate
    # count against the criminal accounts when the targets are restricted.
    bn_nodes, bn_edges = _resolve_block_size(graph, num_pert, block_size)
    bn_edges = max(min(bn_edges, num_eligible), num_pert)
    if criminal:
        bn_nodes = min(bn_nodes, int(candidate_pool.numel()))
    resolved_block_size = (bn_nodes, bn_edges)
    logging.info(
        "Block size: %s nodes / %s edges",
        resolved_block_size[0],
        resolved_block_size[1],
    )

    block = BlockAML(
        graph=graph,
        size=resolved_block_size,
        num_pert=num_pert,
        device=device,
        eligible_edges=eligible,
        rate_matrix=rate_matrix if currency_aware_optimization else None,
        candidate_pool=candidate_pool,
    )
    optimizer = torch.optim.AdamW([block.edge_probs, block.target_logits], lr=lr)

    if criminal:
        # Evasion objective over the illicit edges only: optimize on just
        # those seeds, under a split name that maps the subset loader's seeds.
        criterion = TanhMarginLoss()
        seed_mask = index_to_mask(eligible.cpu(), size=graph.edge_index.size(1))
        opt_split = "attack"
        seed_scope = _temporary_seed_split(model, opt_split, mask_to_index(seed_mask))
    else:
        criterion = torch.nn.CrossEntropyLoss()
        seed_mask = None
        opt_split = seed_split
        seed_scope = nullcontext()

    loader = build_edge_neighbor_loader(
        graph,
        eval_inds=eval_inds,
        bidirectional=False,
        ego_dim=getattr(model, "ego_dim", 0),
        seed_mask=seed_mask,
        **loader_params,
    )

    # Best epoch: lowest PR-AUC, or under the criminal model (illicit-only
    # batches, so no PR-AUC) the highest loss. Kept as "lower is better".
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    # The criminal block holds every illicit edge, so there is nothing new
    # to resample; resampling would only reset the low-probability half.
    resampling = not (criminal and bn_edges >= num_eligible)

    with seed_scope:
        for epoch in trange(max_epochs, leave=False, desc="Epochs"):
            avg_loss, pred_np, gt_np = attack_edge_epoch(
                model,
                loader,
                criterion=criterion,
                optimizer=optimizer,
                split=opt_split,
                tgt_edge_inc_builder=block.perturb_tgt_edge_inc,
                edge_attr_builder=(
                    block.perturb_edge_attr if currency_aware_optimization else None
                ),
                post_step_hook=lambda: block.project_probs(),
                regularizers=regularizers,
            )

            if criminal:
                # Batches hold only illicit seeds: no PR-AUC to compute.
                score = -avg_loss
                metrics = {"loss": avg_loss}
                display = {"attack_loss": avg_loss}
            else:
                f1, pr_auc, precision, recall = compute_binary_metrics(pred_np, gt_np)
                score = pr_auc
                metrics = {
                    "loss": avg_loss,
                    "attack/f1": f1,
                    "attack/pr_auc": pr_auc,
                    "attack/precision": precision,
                    "attack/recall": recall,
                }
                display = {"attack_loss": avg_loss, "attack_pr_auc": pr_auc}

            wandb.log(metrics, step=epoch)

            if display_interval and epoch % display_interval == 0:
                log_epoch_metrics(epoch, display)

            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = block.save_state()
            elif patience and epoch >= best_epoch + patience:
                if resampling:
                    resampling = False
                    best_epoch = epoch
                    if best_state is not None:
                        block.load_state(best_state)
                    continue
                elif epoch >= min_epochs:
                    break

            if resampling:
                block.resample()
                if epoch >= max_resamples:
                    resampling = False

    if best_state is None:
        best_state = block.save_state()

    with torch.no_grad():
        block.load_state(best_state)
        block.project_probs()

        return sample_perturbed_edge_index_from_block_aml(
            model=model,
            graph=graph,
            block=block,
            num_pert=num_pert,
            sample_size=sample_size,
            loader_params=loader_params,
            eval_inds=eval_inds,
            seed_split=seed_split,
            rate_matrix=rate_matrix,
        )

__all__ = [
    "BlockAML",
    "prbcd_attack_aml",
    "uniform_attack_aml",
]
