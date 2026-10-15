import copy
import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from tqdm import trange

import wandb
from robust_multignn.profiling import measure
from robust_multignn.utils import (
    auprc_metrics,
    log_epoch_metrics,
    train_edge_epoch,
    train_epoch,
    val_edge_epoch,
    val_epoch,
)

# Sentinel used for best-model tracking in train_edge
_NEG_INF = float("-inf")


def _checkpoint_best(
    on_new_best: Optional[Callable[[Dict[str, Any]], None]],
    state_dict: Dict[str, Any],
) -> None:
    """Persist a new best checkpoint via ``on_new_best``, without letting a
    checkpointing failure (e.g. a transient W&B upload error) crash training.
    """
    if on_new_best is None:
        return
    try:
        on_new_best(state_dict)
    except Exception:
        logging.exception("Failed to persist new-best checkpoint; continuing training.")


def train(
    model: Module,
    train_loader: NeighborLoader,
    val_loader: NeighborLoader,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: Tensor,
    class_names: List[str],
    min_epochs: int = 0,
    patience: int = 0,
    scheduler_params: Dict[str, Any] = None,
    display_interval: int = 0,
    on_new_best: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Trains model and returns state of  best model based on validation AUPRC.

    Parameters
    ----------
    model : Module
        The model to train.
    train_loader : NeighborLoader
        The data loader for the training set.
    val_loader : NeighborLoader
        The data loader for the validation set.
    max_epochs : int
        The maximum number of epochs to train for.
    lr : float
        The learning rate for the optimizer.
    weight_decay : float
        The weight decay for the optimizer.
    class_weights : Tensor
        The weights for each class in the loss function with shape :obj:`[num_classes]`.
    class_names : List[str]
        The names of the classes.
    min_epochs : int, optional
        The minimum number of epochs to train for, by default 0.
    patience : int, optional
        The number of epochs to wait before stopping early, by default 0.
        If set to 0, early stopping is not applied.
    scheduler_params : Dict[str, Any], optional
        The parameters for the learning rate scheduler, by default None.
    display_interval : int, optional
        The interval at which to display training metrics, by default 0.
        If set to 0, no metrics are displayed during training.
    on_new_best : Callable[[Dict[str, Any]], None], optional
        Called with the model's state dict every time a new best-validation
        checkpoint is found, so it can be persisted immediately (e.g. as a
        W&B artifact) instead of only living in memory until training ends.
        A failure inside this callback is logged and does not interrupt
        training.

    Returns
    -------
    Dict[str, Any]
        The state dictionary of the best model.
    """
    # Set up training components
    device = next(model.parameters()).device
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set up learning rate scheduler
    scheduler = None
    if scheduler_params:
        scheduler = CosineAnnealingWarmRestarts(optimizer, **scheduler_params)

    # Run training loop
    best_macro_auprc = 0.0
    best_epoch = 0
    best_state = None
    for epoch in trange(max_epochs, leave=False, desc="Epochs"):
        # Do one training epoch
        train_loss, train_auprc = train_epoch(model, train_loader, criterion, optimizer)
        train_macro_auprc = train_auprc.mean().item()

        # Do one validation epoch
        val_auprc = val_epoch(model, val_loader)
        val_macro_auprc = val_auprc.mean().item()

        # Log all metrics to Weights & Biases
        metrics = {
            "loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **auprc_metrics("training", train_auprc, class_names),
            **auprc_metrics("validation", val_auprc, class_names),
        }
        wandb.log(metrics, step=epoch)

        # Log loss and macro scores to console at display intervals
        if display_interval and epoch % display_interval == 0:
            log_epoch_metrics(
                epoch,
                {
                    "train_loss": train_loss,
                    "train_macro_auprc": train_macro_auprc,
                    "val_macro_auprc": val_macro_auprc,
                },
            )

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Store best model and stop early when patience is exceeded
        if val_macro_auprc > best_macro_auprc:
            best_macro_auprc = val_macro_auprc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            _checkpoint_best(on_new_best, best_state)
        elif patience and epoch >= min_epochs and epoch >= best_epoch + patience:
            break

    return best_state


def train_aml(
    model: Module,
    train_loader: LinkNeighborLoader,
    val_loader: LinkNeighborLoader,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: Tensor,
    class_names: List[str],
    min_epochs: int = 0,
    patience: int = 0,
    scheduler_params: Dict[str, Any] = None,
    display_interval: int = 0,
    on_new_best: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Train an edge-level model; return the best state dict by validation PR-AUC.

    Seed-edge masking is handled inside the model (via
    :class:`~robust_multignn.models.line_graph_gnn.EdgePredictionHead`).
    Best model is selected by the area under the precision-recall curve on the
    validation split.

    Parameters
    ----------
    model : Module
        Edge-level GNN (e.g. ``LineGraphGNNModel``).
    train_loader : LinkNeighborLoader
        Edge-level loader for the training split.
    val_loader : LinkNeighborLoader
        Edge-level loader for the validation split (seeds restricted to
        val-period edges via ``edge_label_index``).
    max_epochs : int
        Maximum number of training epochs.
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight-decay coefficient.
    class_weights : Tensor
        Positive-class weight for ``BCEWithLogitsLoss`` (shape ``[1]``).
    class_names : List[str]
        Class names used for logging (typically ``["is_laundering"]``).
    min_epochs : int, optional
        Minimum epochs before early stopping is considered.
    patience : int, optional
        Early-stopping patience in epochs; 0 disables early stopping.
    scheduler_params : Dict[str, Any], optional
        ``CosineAnnealingWarmRestarts`` keyword arguments; ``None`` disables.
    display_interval : int, optional
        Console logging interval in epochs; 0 disables console output.
    on_new_best : Callable[[Dict[str, Any]], None], optional
        Called with the model's state dict every time a new best-validation
        checkpoint is found, so it can be persisted immediately (e.g. as a
        W&B artifact) instead of only living in memory until training ends.
        A failure inside this callback is logged and does not interrupt
        training.

    Returns
    -------
    Dict[str, Any]
        State dictionary of the best model checkpoint.
    """
    device = next(model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, class_weights]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if scheduler_params:
        scheduler = CosineAnnealingWarmRestarts(optimizer, **scheduler_params)

    best_val_pr_auc = _NEG_INF
    best_epoch = 0
    best_state = None

    device = next(model.parameters()).device

    for epoch in trange(max_epochs, leave=False, desc="Epochs"):
        # Timed so a standard epoch is directly comparable with an adversarial
        # one (same keys, logged by experiment_adv_train.log_epoch_cost); their
        # ratio is the adversarial-training overhead.
        with measure("train_epoch", device) as epoch_cost:
            train_loss, _, _ = train_edge_epoch(
                model, train_loader, criterion, optimizer, split="train"
            )
        val_f1, val_pr_auc, val_precision, val_recall = val_edge_epoch(
            model, val_loader, split="val"
        )

        metrics = {
            "loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "cost/epoch_seconds": epoch_cost.seconds,
            "cost/epoch_host_peak_gb": epoch_cost.host_peak_bytes / 1e9,
            **(
                {"cost/epoch_gpu_peak_gb": epoch_cost.gpu_peak_bytes / 1e9}
                if epoch_cost.gpu_peak_bytes is not None
                else {}
            ),
            "val/f1": val_f1,
            "val/pr_auc": val_pr_auc,
            "val/precision": val_precision,
            "val/recall": val_recall,
        }
        wandb.log(metrics, step=epoch)

        if display_interval and epoch % display_interval == 0:
            log_epoch_metrics(
                epoch,
                {
                    "train_loss": train_loss,
                    "val_pr_auc": val_pr_auc,
                    "val_f1": val_f1,
                },
            )

        if scheduler is not None:
            scheduler.step()

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            _checkpoint_best(on_new_best, best_state)
        elif patience and epoch >= min_epochs and epoch >= best_epoch + patience:
            break

    return best_state
