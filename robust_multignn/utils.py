import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import sklearn.metrics
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm, trange

if TYPE_CHECKING:
    from robust_multignn.attacks.regularizers import LossCallback
from robust_multignn.data import (
    GraphData,
    GraphDataBatch,
    augment_with_ego_ids,
    build_neighbor_loader,
    compute_batch_ports,
)

def compute_binary_metrics(
    pred: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute binary classification metrics from two-class probabilities.

    Parameters
    ----------
    pred : np.ndarray
        Class probabilities (typically softmax output), shape ``[N, 2]``.
    labels : np.ndarray
        Binary ground-truth labels, shape ``[N]`` or ``[N, 1]``.

    Returns
    -------
    Tuple[float, float, float, float]
        ``(f1, pr_auc, precision, recall)`` where ``pr_auc`` is the area under
        the precision-recall curve.
    """
    probs = pred[:,1]
    preds_binary = pred.argmax(axis=-1)
    labels = labels.astype(int).flatten()

    precisions, recalls, _ = sklearn.metrics.precision_recall_curve(labels, probs)
    pr_auc = sklearn.metrics.auc(recalls, precisions)
    f1 = sklearn.metrics.f1_score(labels, preds_binary, zero_division=0)
    precision = sklearn.metrics.precision_score(labels, preds_binary, zero_division=0)
    recall = sklearn.metrics.recall_score(labels, preds_binary, zero_division=0)
    return f1, pr_auc, precision, recall
    

class BoundaryLoss(Module):
    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        y_tilde = targets * 2 - 1
        margin = y_tilde * logits
        loss = 0.5 * (torch.tanh(self.beta * margin) + 1.0)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class TanhMarginLoss(Module):
    """Tanh margin loss for global attacks (Geisler et al., NeurIPS 2021).

    ``tanh(max_{c != y} z_c - z_y)``, averaged: positive once a prediction is
    wrong, negative while it is right. The edge-level attack loop *maximizes*
    it. Unlike cross-entropy it saturates, so an attack stops spending budget
    on predictions that are already confidently wrong and focuses on those
    still near the decision boundary. Computed on logits, as in the authors'
    implementation (``rgnn_at_scale/attacks/base_attack.py``, ``tanhMargin``).
    """

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.long()
        true_logit = logits.gather(1, targets.view(-1, 1)).squeeze(1)
        other = logits.scatter(1, targets.view(-1, 1), float("-inf"))
        margin = other.amax(dim=1) - true_logit
        return torch.tanh(margin).mean()


def train_epoch(
    model: Module,
    loader: NeighborLoader,
    criterion: Module,
    optimizer: Optimizer,
) -> Tuple[float, Tensor]:
    """Runs training epoch.

    Parameters
    ----------
    model : Module
        The model to train.
    loader : NeighborLoader
        The data loader for the epoch.
    criterion : Module
        The loss function.
    optimizer : Optimizer
        The optimizer for training.

    Returns
    -------
    Tuple[float, Tensor]
        The average loss and per-class AUPRC scores.
    """
    model.train()
    return _run_epoch(
        model,
        loader,
        criterion,
        optimizer,
        batch_desc="Training Batches",
    )


@torch.no_grad()
def val_epoch(
    model: Module,
    loader: NeighborLoader,
    batch_desc: str = "Validation Batches",
) -> Tensor:
    """Runs validation epoch.

    Parameters
    ----------
    model : Module
        The model to validate.
    loader : NeighborLoader
        The data loader for the epoch.
    batch_desc : str, optional
        Label shown for the tqdm batch progress bar, by default "Validation Batches"

    Returns
    -------
    Tensor
        Per-class AUPRC scores.
    """
    model.eval()
    return _run_epoch(model, loader, batch_desc=batch_desc)


def train_edge_epoch(
    model: Module,
    loader: LinkNeighborLoader,
    criterion: Module,
    optimizer: Optimizer,
    split: str = "train",
) -> Tuple[float, "np.ndarray", "np.ndarray"]:
    """Run one training epoch for edge-level classification.

    Parameters
    ----------
    model : Module
        Edge-level GNN whose ``forward(batch, split)`` returns logits for seed
        edges only (masking is handled inside the model).
    loader : LinkNeighborLoader
        Edge-level mini-batch loader.
    criterion : Module
        Loss function (e.g. ``BCEWithLogitsLoss``).
    optimizer : Optimizer
        Optimiser for training.
    split : str, optional
        Split name forwarded to the model (default ``"train"``).

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        ``(avg_loss, sigmoid_probabilities, ground_truth_labels)`` where the
        numpy arrays have shape ``[N]``.
    """
    model.train()
    return _run_edge_epoch(
        model, loader, split, criterion, optimizer, batch_desc="Training Batches"
    )


@torch.no_grad()
def val_edge_epoch(
    model: Module,
    loader: LinkNeighborLoader,
    split: str = "val",
    batch_desc: str = "Validation Batches",
) -> Tuple[float, float, float, float]:
    """Run one validation/test epoch for edge-level classification.

    Parameters
    ----------
    model : Module
        Edge-level GNN whose ``forward(batch, split)`` returns logits for seed
        edges only.
    loader : LinkNeighborLoader
        Edge-level mini-batch loader.
    split : str, optional
        Split name forwarded to the model (``"val"`` or ``"test"``).
    batch_desc : str, optional
        Label for the tqdm progress bar.

    Returns
    -------
    Tuple[float, float, float, float]
        ``(f1, pr_auc, precision, recall)`` binary classification metrics.
    """
    model.eval()
    return _run_edge_epoch(model, loader, split, batch_desc=batch_desc)


def build_adj_from_incidence(src_inc: Tensor, tgt_inc: Tensor) -> Tensor:
    """``src_inc @ tgt_inc.T`` as sparse COO without a sparse-sparse matmul.

    ``src_inc`` has exactly one nonzero per edge column (each edge has one
    source node), so the product is a pure re-indexing of ``tgt_inc``'s
    entries: ``A[src(e), j] += src_val(e) * tgt_inc[j, e]``. This matters for
    autograd: torch's backward through ``torch.sparse.mm(sparse, sparse)``
    materializes dense ``[num_nodes, num_nodes]`` intermediates (~47GB on an
    AML batch), while the backward of this construction is a cheap gather.
    """
    src_inc = src_inc.coalesce()
    s_node, s_edge = src_inc.indices()
    num_edges = src_inc.size(1)
    src_of_edge = torch.empty(num_edges, dtype=torch.long, device=s_node.device)
    src_of_edge[s_edge] = s_node
    src_val_of_edge = torch.zeros(
        num_edges, dtype=src_inc.values().dtype, device=s_node.device
    )
    src_val_of_edge[s_edge] = src_inc.values()

    tgt_inc = tgt_inc.coalesce()
    t_node, t_edge = tgt_inc.indices()
    num_nodes = src_inc.size(0)
    return torch.sparse_coo_tensor(
        torch.stack([src_of_edge[t_edge], t_node]),
        tgt_inc.values() * src_val_of_edge[t_edge],
        (num_nodes, num_nodes),
    ).coalesce()


def _run_edge_epoch(
    model: Module,
    loader: LinkNeighborLoader,
    split: str,
    criterion: Module | None = None,
    optimizer: Optimizer | None = None,
    maximize_loss: bool = False,
    tgt_edge_inc_builder: Callable[[GraphDataBatch], Tensor] | None = None,
    edge_attr_builder: Callable[[GraphDataBatch], Tensor] | None = None,
    post_step_hook: Callable[[], None] | None = None,
    regularizers: List[Tuple["LossCallback", float]] | None = None,
    batch_desc: str = "Batches",
):
    """Core loop for edge-level training / validation / attack.

    Seed-edge masking is handled inside the model (via
    :class:`~robust_multignn.models.line_graph_gnn.EdgePredictionHead` which
    uses ``batch.e_id`` and ``batch.input_id``).  The model returns logits
    **only for seed edges**, matching ``batch.edge_label`` directly — no
    ID-column or external masking is needed here.

    Parameters
    ----------
    model : Module
        Edge-level GNN whose ``forward(batch, split, tgt_edge_inc)`` returns
        seed-edge logits.
    loader : LinkNeighborLoader
        Edge-level mini-batch loader.
    split : str
        One of ``'train'``, ``'val'``, ``'test'``; forwarded to the model.
    criterion : Module, optional
        Loss function; required for training / attack.
    optimizer : Optimizer, optional
        Optimiser; required for training / attack.
    maximize_loss : bool, optional
        If ``True``, performs gradient ascent (attack mode).
    tgt_edge_inc_builder : Callable[[GraphDataBatch], Tensor] | None, optional
        Callback that returns a perturbed target-edge incidence matrix for each
        batch (PGD-style attacks).  When ``None``, the model falls back to
        ``batch.tgt_edge_inc``.
    post_step_hook : Callable[[], None] | None, optional
        Called after each ``optimizer.step()`` to enforce perturbation budgets.
    regularizers : List[Tuple[LossCallback, float]] | None, optional
        ``(callback, weight)`` tuples for incidence-based regularization. Applied
        only when ``tgt_edge_inc_builder`` is set (i.e. gradient-based attacks).
    batch_desc : str
        tqdm label.

    Returns
    -------
    Training / attack mode : ``Tuple[float, np.ndarray, np.ndarray]``
        ``(avg_loss, sigmoid_probs, ground_truth)``
    Validation mode : ``Tuple[float, float, float, float]``
        ``(f1, pr_auc, precision, recall)``
    """
    training = optimizer is not None
    if training and criterion is None:
        raise ValueError("criterion is required when optimizer is provided")

    device = next(model.parameters()).device
    total_loss = 0.0
    total_examples = 0
    preds: List[Tensor] = []
    ground_truths: List[Tensor] = []

    # Debug-only accounting for regularizer contribution, aggregated over the
    # epoch (per-batch logging here would flood the log across many batches
    # x many epochs). Reported once at the end of the epoch.
    reg_ce_sum = 0.0
    reg_raw_sums: dict[str, float] = {}
    reg_weighted_sums: dict[str, float] = {}
    reg_batches = 0
    cycle_stat_sums: dict[str, float] = {}

    for batch in tqdm(loader, total=len(loader), leave=False, desc=batch_desc):
        if training:
            optimizer.zero_grad()

        batch = batch.to(device, non_blocking=True)

        # Optionally inject a perturbed target-edge incidence matrix (e.g. PGD attack)
        tgt_edge_inc = (
            tgt_edge_inc_builder(batch)
            if tgt_edge_inc_builder is not None
            else None
        )
        # Optionally replace the batch's edge features with perturbation-aware
        # ones (e.g. currency-converted amounts for edges the attack may
        # rewire). Models read batch.edge_attr directly, so this is set on the
        # batch rather than passed through the forward signature.
        if edge_attr_builder is not None:
            batch.edge_attr = edge_attr_builder(batch)
        # Forward pass: model returns logits for seed edges only
        pred, ground_truth = model(batch, split, tgt_edge_inc=tgt_edge_inc)

        if training:
            loss: Tensor = criterion(pred, ground_truth.long())
            objective = -loss if maximize_loss else loss

            # Incidence-based regularization (e.g. cycle/degree loss). The
            # attacker ascends the model loss but should keep structural
            # perturbations small, so the penalty is *added* to the (already
            # sign-flipped) objective and thus minimised.
            if tgt_edge_inc_builder is not None and regularizers:
                pert_adj = torch.sparse.mm(batch.src_edge_inc, tgt_edge_inc.T)
                orig_adj = torch.sparse.mm(batch.src_edge_inc, batch.tgt_edge_inc.T)
                ce_loss = float(loss.detach())
                reg_ce_sum += ce_loss
                reg_batches += 1
                for cb, weight in regularizers:
                    cb_loss = cb(pert_adj, orig_adj)
                    objective = objective + weight * cb_loss
                    name = getattr(cb, "__name__", str(cb))
                    reg_raw_sums[name] = reg_raw_sums.get(name, 0.0) + float(
                        cb_loss.detach()
                    )
                    reg_weighted_sums[name] = reg_weighted_sums.get(
                        name, 0.0
                    ) + weight * float(cb_loss.detach())
                    if name == "cycle_loss_sparse":
                        from robust_multignn.attacks.regularizers import (
                            cycle_stats,
                        )

                        for stat_name, val in cycle_stats(
                            pert_adj, orig_adj
                        ).items():
                            cycle_stat_sums[stat_name] = (
                                cycle_stat_sums.get(stat_name, 0.0) + val
                            )

            objective.backward()
            optimizer.step()
            if post_step_hook is not None:
                post_step_hook()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        preds.append(pred.detach().cpu())
        ground_truths.append(ground_truth.detach().cpu())

    if training and reg_batches:
        avg_ce = reg_ce_sum / reg_batches
        for name in reg_raw_sums:
            avg_raw = reg_raw_sums[name] / reg_batches
            avg_weighted = reg_weighted_sums[name] / reg_batches
            ratio = avg_weighted / avg_ce if avg_ce != 0 else float("inf")
            logging.info(
                "[reg %s] avg_raw=%.3e avg_weighted=%.3e | avg_ce=%.3e | "
                "avg_weighted/avg_ce=%.3e (over %d batches)",
                name,
                avg_raw,
                avg_weighted,
                avg_ce,
                ratio,
                reg_batches,
            )

        if cycle_stat_sums:
            avg_num_nodes = cycle_stat_sums["num_nodes"] / reg_batches
            avg_pert_closed = cycle_stat_sums["pert_closed_2walk_nodes"] / reg_batches
            avg_orig_closed = cycle_stat_sums["orig_closed_2walk_nodes"] / reg_batches
            logging.info(
                "[cycle stats] avg_num_nodes=%.1f | closed-2-walk nodes: "
                "pert=%.2f orig=%.2f (of %.1f) | avg_trace2 pert=%.3e orig=%.3e | "
                "avg_trace3 pert=%.3e orig=%.3e (over %d batches)",
                avg_num_nodes,
                avg_pert_closed,
                avg_orig_closed,
                avg_num_nodes,
                cycle_stat_sums["pert_trace2"] / reg_batches,
                cycle_stat_sums["orig_trace2"] / reg_batches,
                cycle_stat_sums["pert_trace3"] / reg_batches,
                cycle_stat_sums["orig_trace3"] / reg_batches,
                reg_batches,
            )

    pred_np = torch.cat(preds, dim=0).numpy()
    gt_np = torch.cat(ground_truths, dim=0).numpy()

    if training:
        return total_loss / total_examples, pred_np, gt_np

    f1, pr_auc, precision, recall = compute_binary_metrics(pred_np, gt_np)
    return f1, pr_auc, precision, recall


def attack_epoch(
    model: Module,
    loader: NeighborLoader,
    criterion: Module,
    optimizer: Optimizer,
    tgt_edge_inc_builder: Callable[[GraphDataBatch], Tensor] | None = None,
    post_step_hook: Callable[[], None] | None = None,
    batch_desc: str = "Attack Batches",
    regularizers: List[Tuple["LossCallback", float]] | None = None,
) -> Tuple[float, Tensor]:
    """Runs one epoch of the attack.

    Parameters
    ----------
    model : Module
        The model to attack.
    loader : NeighborLoader
        The data loader for the graph data.
    criterion : Module
        The loss function.
    optimizer : Optimizer
        The optimizer for the attack.
    tgt_edge_inc_builder : Callable[[GraphDataBatch], Tensor] | None, optional
        A callback to build the target-edge incidence matrix for a batch,
        by default None
    post_step_hook : Callable[[], None] | None, optional
        A callback executed after each optimization step, by default None
    batch_desc : str, optional
        Label shown for the tqdm batch progress bar, by default "Attack Batches"
    regularizers : List[Tuple["LossCallback", float]] | None, optional
        A list of (callback, weight) tuples for incidence-based regularization.

    Returns
    -------
    Tuple[float, Tensor]
        The average loss and per-class AUPRC scores.
    """
    model.eval()
    return _run_epoch(
        model,
        loader,
        criterion,
        optimizer,
        batch_desc=batch_desc,
        tgt_edge_inc_builder=tgt_edge_inc_builder,
        post_step_hook=post_step_hook,
        regularizers=regularizers,
    )


def attack_edge_epoch(
    model: Module,
    loader: LinkNeighborLoader,
    criterion: Module,
    optimizer: Optimizer,
    split: str = "train",
    tgt_edge_inc_builder: Callable[[GraphDataBatch], Tensor] | None = None,
    edge_attr_builder: Callable[[GraphDataBatch], Tensor] | None = None,
    post_step_hook: Callable[[], None] | None = None,
    regularizers: List[Tuple["LossCallback", float]] | None = None,
    batch_desc: str = "Attack Batches",
) -> Tuple[float, "np.ndarray", "np.ndarray"]:
    """Run one epoch of an edge-level attack (gradient ascent on the loss).

    Mirrors :func:`attack_epoch` for the node-level case.  Passes
    ``maximize_loss=True`` to :func:`_run_edge_epoch` so the optimiser
    performs gradient ascent, allowing PGD-style attacks to maximise the
    model's loss with respect to learnable perturbation parameters.

    Parameters
    ----------
    model : Module
        Edge-level GNN to attack (kept in eval mode so BatchNorm etc. use
        running statistics rather than batch statistics).
    loader : LinkNeighborLoader
        Edge-level mini-batch loader.
    criterion : Module
        Loss function whose value is **maximised** during the attack.
    optimizer : Optimizer
        Optimiser that updates the perturbation parameters (not model weights).
    split : str, optional
        Split name forwarded to the model (default ``"train"``).
    tgt_edge_inc_builder : Callable[[GraphDataBatch], Tensor] | None, optional
        Callback that returns a soft perturbed target-edge incidence matrix for
        the given batch (e.g. ``perturb_tgt_edge_inc`` from PGD).
    post_step_hook : Callable[[], None] | None, optional
        Callback executed after each ``optimizer.step()`` to enforce
        perturbation constraints.
    regularizers : List[Tuple[LossCallback, float]] | None, optional
        ``(callback, weight)`` tuples for incidence-based regularization,
        forwarded to :func:`_run_edge_epoch`.
    batch_desc : str, optional
        Label for the tqdm progress bar.

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        ``(avg_loss, sigmoid_probs, ground_truth)`` accumulated over the epoch.
    """
    model.eval()
    return _run_edge_epoch(
        model,
        loader,
        split,
        criterion,
        optimizer,
        maximize_loss=True,
        tgt_edge_inc_builder=tgt_edge_inc_builder,
        edge_attr_builder=edge_attr_builder,
        post_step_hook=post_step_hook,
        regularizers=regularizers,
        batch_desc=batch_desc,
    )


def _iter_batch_loader(
    model: Module,
    loader: NeighborLoader,
    batch_desc: str,
) -> Iterator[tuple[GraphDataBatch, int, Tensor]]:
    """Iterates over batches from the loader, moves them to the model's device,
    and applies ego ID augmentation.

    Parameters
    ----------
    model : Module
        The model whose device will be used for the batches.
    loader : NeighborLoader
        The data loader for the batches.
    batch_desc : str
        Label shown for the tqdm batch progress bar.

    Yields
    ------
    Iterator[tuple[GraphDataBatch, int, Tensor]]
        The batch, batch size, and augmented features.
    """
    device = next(model.parameters()).device
    ego_dim = getattr(model, "ego_dim", 0)

    batch: GraphDataBatch
    for batch in tqdm(loader, total=len(loader), leave=False, desc=batch_desc):
        batch.to(device, non_blocking=True)
        batch_size = batch.batch_size
        x_aug = augment_with_ego_ids(batch.x, batch_size, ego_dim)
        yield batch, batch_size, x_aug


def _run_epoch(
    model: Module,
    loader: NeighborLoader,
    criterion: Module | None = None,
    optimizer: Optimizer | None = None,
    batch_desc: str = "Batches",
    tgt_edge_inc_builder: Callable[[GraphDataBatch], Tensor] | None = None,
    post_step_hook: Callable[[], None] | None = None,
    regularizers: List[Tuple["LossCallback", float]] | None = None,
) -> Tuple[float, Tensor] | Tensor:
    """Runs one epoch of training or validation based on the presence of an optimizer.

    Parameters
    ----------
    model : Module
        The model to train or validate.
    loader : NeighborLoader
        The data loader for the epoch.
    criterion : Module | None, optional
        The loss function. Required when training.
    optimizer : Optimizer | None, optional
        The optimizer for training, by default None
    batch_desc : str, optional
        Label shown for the tqdm batch progress bar, by default "Batches"
    tgt_edge_inc_builder : Callable[[GraphDataBatch], Tensor] | None, optional
        Optional callback to build the target-edge incidence matrix for a batch.
    post_step_hook : Callable[[], None] | None, optional
        Optional callback executed after optimizer.step() in training mode.
    regularizers : List[Tuple["LossCallback", float]] | None, optional
        A list of (callback, weight) tuples for incidence-based regularization.

    Returns
    -------
    Tuple[float, Tensor] | Tensor
        In training mode, returns average loss and per-class AUPRC scores.
        In validation mode, returns only per-class AUPRC scores.

    Raises
    ------
    ValueError
        If optimizer is provided without a criterion.
    """
    # Determine if we are in training mode based on the presence of an optimizer
    training = optimizer is not None
    if training and criterion is None:
        raise ValueError("criterion is required when optimizer is provided")

    use_ports = getattr(model, "use_ports", False)

    # Iterate over batches
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []
    for batch, batch_size, x_aug in _iter_batch_loader(model, loader, batch_desc):
        # Reset gradients in training mode
        if training:
            optimizer.zero_grad()

        # Recompute port IDs from the sampled subgraph so they reflect local
        # timestamp ordering (correct after perturbations and prevents overfitting
        # to fixed port ID patterns seen during full-graph pre-computation).
        if use_ports:
            batch.in_ports, batch.out_ports = compute_batch_ports(
                batch.edge_index, batch.time, batch.num_nodes
            )

        # Optionally build a custom target-edge incidence matrix (e.g., for attacks)
        if tgt_edge_inc_builder is not None:
            tgt_edge_inc = tgt_edge_inc_builder(batch)
        else:
            tgt_edge_inc = batch.tgt_edge_inc

        # Forward pass through the model.
        logits = model(
            x=x_aug,
            src_edge_inc=batch.src_edge_inc,
            tgt_edge_inc=tgt_edge_inc,
            edge_attr=batch.edge_attr,
            in_ports=batch.in_ports if use_ports else None,
            out_ports=batch.out_ports if use_ports else None,
        )

        # Extract logits and labels for the current batch
        batch_logits: Tensor = logits[:batch_size]
        batch_labels: Tensor = batch.y[:batch_size]

        # Compute loss and backpropagate in training mode
        if training:
            batch_loss: Tensor = criterion(batch_logits, batch_labels)

            if tgt_edge_inc_builder is not None and regularizers:
                # Compute sparse directed adjacency matrices: src @ tgt.T
                pert_adj = torch.sparse.mm(batch.src_edge_inc, tgt_edge_inc.T)
                orig_adj = torch.sparse.mm(batch.src_edge_inc, batch.tgt_edge_inc.T)

                ce_loss = float(batch_loss.detach())
                for cb, weight in regularizers:
                    cb_loss = cb(pert_adj, orig_adj)
                    batch_loss = batch_loss + weight * cb_loss
                    raw = float(cb_loss.detach())
                    weighted = weight * raw
                    ratio = weighted / ce_loss if ce_loss != 0 else float("inf")
                    logging.info(
                        "[reg %s] raw=%.3e weight=%.3e weighted=%.3e | "
                        "ce=%.3e | weighted/ce=%.3e",
                        getattr(cb, "__name__", str(cb)),
                        raw,
                        weight,
                        weighted,
                        ce_loss,
                        ratio,
                    )

            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * batch_size
            total_samples += batch_size

        # Execute post-step hook if provided (e.g., enforcing attack constraints)
        if post_step_hook is not None:
            post_step_hook()

        # Store logits and labels for metric computation
        all_logits.append(batch_logits.detach())
        all_labels.append(batch_labels.detach())

    # Compute per-class AUPRC scores for the entire epoch
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    predictions = torch.sigmoid(logits)
    auprc_scores = per_class_auprc(labels, predictions)

    # Additionally return average loss if in training mode
    if training:
        avg_loss = total_loss / total_samples
        return avg_loss, auprc_scores

    # In validation mode, only return AUPRC scores
    return auprc_scores


def per_class_auprc(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Computes per-class AUPRC scores for multi-label classification. If the positive
    class froms the majority of samples, the labels and predictions are flipped to
    ensure the metric focuses on the minority class.

    Parameters
    ----------
    y_true : Tensor
        Binary ground-truth labels of shape :obj:`[num_samples, num_classes]`.
    y_pred : Tensor
        Predicted probabilities for the positive class
        of shape :obj:`[num_samples, num_classes]`.

    Returns
    -------
    Tensor
        Per-class AUPRC scores of shape :obj:`[num_classes]`.
    """
    # Flip labels and predictions for positive majority classes
    minority_mask = 2 * y_true.sum(dim=0) < y_true.size(0)
    y_true = torch.where(minority_mask, y_true, 1 - y_true)
    y_pred = torch.where(minority_mask, y_pred, 1 - y_pred)

    # Calculate per-class AUPRC scores
    num_classes = y_true.size(1)
    metric = MultilabelAveragePrecision(num_labels=num_classes, average=None)
    return metric(y_pred.float(), y_true.int())


def auprc_metrics(
    prefix: str,
    scores: Tensor,
    class_names: List[str],
) -> Dict[str, float]:
    """Formats AUPRC scores into a dictionary with keys for macro and per-class scores.

    Parameters
    ----------
    prefix : str
        A prefix to add to the metric keys (e.g., "train" or "val").
    scores : Tensor
        Per-class AUPRC scores of shape :obj:`[num_classes]`.
    class_names : List[str]
        The names of the classes corresponding to the scores.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping metric names to their values.
    """
    return {
        f"{prefix}/auprc_macro": scores.mean().item(),
        **{
            f"{prefix}/auprc_{class_name}": scores[i].item()
            for i, class_name in enumerate(class_names)
        },
    }


def log_table(
    title: str,
    names: List[str],
    values: Tensor,
):
    """Logs a table of values with a title and corresponding names.

    Parameters
    ----------
    title : str
        The title of the table.
    names : List[str]
        The names of the rows in the table.
    values : Tensor
        The values to display in the table.
    """
    logging.info("%s", f" {title} ".center(25, "—"))
    for i, name in enumerate(names):
        logging.info("%-16s : %.4f", name, values[i].item())


def log_epoch_metrics(
    epoch: int,
    metrics: Dict[str, float],
):
    """Logs epoch metrics to the console in a compact format.

    Parameters
    ----------
    epoch : int
        The current epoch number.
    metrics : Dict[str, float]
        A dictionary mapping metric names to their values.
    """
    parts = []
    for key, value in metrics.items():
        parts.append(f"{key}={value:.4f}")
    logging.info("epoch %03d: %s", epoch, " | ".join(parts))


def compute_class_weights(labels: Tensor) -> Tensor:
    """Computes positive class weights for imbalanced multi-label classification.

    Parameters
    ----------
    labels : Tensor
        Binary ground-truth labels of shape :obj:`[num_samples, num_classes]`.

    Returns
    -------
    Tensor
        Positive class weights of shape :obj:`[num_classes]`.

    """
    pos_counts = labels.sum(dim=0)
    neg_counts = labels.size(0) - pos_counts
    weights = neg_counts / pos_counts.clamp(min=1)
    return weights


def project_onto_capped_simplex_(
    input: Tensor,
    s: float,
    surface_only: bool,
    **kwargs,
):
    """Project onto the capped simplex defined by {x | sum(x) <= s, 0 <= x <= 1}.

    This enforces a fixed L1 budget (sum constraint) while keeping all entries within
    [0, 1], which is useful for distributing a limited perturbation budget across edges.
    The implementation follows the root-finding approach for capped simplex projection
    described by Ang et. al. (https://arxiv.org/abs/2110.08471).

    Parameters
    ----------
    input : Tensor
        The input tensor to be projected.
    s : float
        Sum constraint for the capped simplex.
    surface_only : bool
        If True, only project onto the surface of the simplex (sum(x) = s).
    """
    from scipy import optimize

    # Check if the input is already within the capped simplex
    if not surface_only and torch.clamp(input, 0, 1).sum().item() <= s:
        input.copy_(torch.clamp(input, 0, 1))
        return

    # Gamma bounds for root finding
    gamma_low = torch.min(input).item() - 1
    gamma_high = torch.max(input).item()

    # Function to find the root for projection onto the capped simplex
    def omega_prime(gamma, y, s):
        return s - torch.sum(torch.clamp(y - gamma, min=0, max=1)).item()

    # Use a root-finding method to find the optimal gamma for projection
    func = partial(omega_prime, y=input, s=s)
    gamma_star = optimize.brentq(func, gamma_low, gamma_high, **kwargs)
    result = torch.clamp(input - gamma_star, min=0, max=1)
    input.copy_(result)


@torch.no_grad()
def sample_perturbed_edge_index(
    model: Module,
    graph: GraphData,
    num_pert: int,
    edge_probs: Tensor,
    target_probs: Tensor,
    sample_size: int,
    loader_params: dict[str, Any],
) -> Tensor:
    """Samples hard perturbations from edge and target probabilities,
    and returns the best perturbed edge index based on validation performance.

    Parameters
    ----------
    model : Module
        The model to evaluate the perturbations on.
    graph : GraphData
        The graph data for the attack.
    num_pert : int
        The number of edges to perturb.
    edge_probs : Tensor
        The probabilities of perturbing each edge.
    target_probs : Tensor
        The probabilities for selecting new target nodes for each edge.
    sample_size : int
        The number of samples to use for the attack.
    loader_params : dict[str, Any]
        The parameters for the data loader.

    Returns
    -------
    Tensor
        The best perturbed edge index based on validation performance.
    """
    device = next(model.parameters()).device

    # Keep track of the best perturbation so far based on macro PR-AUC
    best_auprc_macro = float("inf")
    best_edge_index = None
    base_edge_index = graph.edge_index.clone()

    # Sample i.i.d. edge index perturbations
    for _ in trange(sample_size, leave=False, desc="Perturbation Samples"):
        edge_index = base_edge_index.clone().to(device)

        # Samples edges to perturb and their new target nodes
        pert_edge_idx = torch.multinomial(edge_probs, num_pert, replacement=False)
        pert_node_idx = target_probs[pert_edge_idx].multinomial(num_samples=1).flatten()
        edge_index[1, pert_edge_idx] = pert_node_idx

        # Run validation epoch with the perturbed edge index
        graph.edge_index = edge_index.cpu()
        loader = build_neighbor_loader(graph, **loader_params)
        val_auprc = val_epoch(model=model, loader=loader)
        val_auprc_macro = val_auprc.mean().item()

        # Keep the perturbation with the best (lowest) macro AUPRC score
        if val_auprc_macro < best_auprc_macro:
            best_auprc_macro = val_auprc_macro
            best_edge_index = edge_index

    # Verify that we have found a valid perturbation
    num_pert_actual = (best_edge_index != base_edge_index.to(device)).sum().item()
    assert num_pert_actual == num_pert, (
        f"Expected {num_pert} perturbed edges, but got {num_pert_actual}. "
        "This indicates an issue with the sampling process."
    )

    return best_edge_index
