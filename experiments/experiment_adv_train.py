import copy
import logging
from pathlib import Path
from typing import Any, Dict, Union

import torch
import torch.nn.functional as F
from tqdm import trange

import wandb
from experiments.common import check_availability, set_seed, wandb_disabled
from robust_multignn.aml import AML_DATASET_NAMES
from robust_multignn.aml.exchange_rates import build_dataset_rate_matrix
from robust_multignn.aml.utils import mask_to_index
from robust_multignn.attacks import get_attack
from robust_multignn.attacks.aml import THREAT_MODELS
from robust_multignn.data import build_edge_neighbor_loader
from robust_multignn.loading import (
    GraphDataset,
    create_config_name,
    load_dataset,
    load_model_artifact,
    load_resume_state,
    resume_state_path,
    save_model_artifact,
    save_resume_state,
    validate_baseline_artifact_metadata,
)
from robust_multignn.models import create_model
from robust_multignn.profiling import Measurement, measure
from robust_multignn.train import train_aml
from robust_multignn.utils import (
    log_epoch_metrics,
    log_table,
    train_edge_epoch,
    val_edge_epoch,
)


def log_epoch_cost(
    epoch_cost: Measurement,
    attack_cost: Measurement | None = None,
    *,
    epoch: int | None = None,
) -> None:
    """Log wall-clock and peak memory for one training epoch to W&B.

    Standard and adversarial epochs log the same keys, so the two are directly
    comparable and their ratio is the adversarial-training overhead.
    ``cost/epoch_attack_seconds`` isolates the attack's share of an adversarial
    epoch.
    """
    metrics = {
        "cost/epoch_seconds": epoch_cost.seconds,
        "cost/epoch_host_peak_gb": epoch_cost.host_peak_bytes / 1e9,
    }
    if epoch_cost.gpu_peak_bytes is not None:
        metrics["cost/epoch_gpu_peak_gb"] = epoch_cost.gpu_peak_bytes / 1e9
    if attack_cost is not None:
        metrics["cost/epoch_attack_seconds"] = attack_cost.seconds
    if epoch is not None:
        metrics["cost/epoch_index"] = epoch
    # Log at the epoch's step. Without an explicit step, wandb.log advances the
    # run's step counter, so the per-epoch metrics logged afterwards with
    # step=epoch land on a step W&B considers already passed and are dropped
    # ("Steps must be monotonically increasing").
    wandb.log(metrics, step=epoch)


def _clone_or_none(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Clone a tensor, passing ``None`` through for graphs that lack it."""
    return None if tensor is None else tensor.clone()


def _checkpoint_best(
    state_dict: Dict[str, Any],
    artifact_name: str,
    metadata: Dict[str, Any],
    save_artifacts: bool = True,
) -> None:
    """Persist a new best checkpoint immediately (e.g. as a W&B artifact) so
    multi-day adversarial training runs don't lose progress if the process
    crashes before reaching the end of the loop. A failure here is logged but
    does not interrupt training.

    ``save_artifacts=False`` skips the upload entirely. Cost-measurement runs
    train for only a few epochs but resolve to the same artifact name as the
    real run, so without this they would publish an under-trained model as the
    new ``:latest`` that attack configs then load.
    """
    if not save_artifacts:
        logging.info(
            "save_artifacts=False; not publishing checkpoint '%s'", artifact_name
        )
        return
    try:
        save_model_artifact(state_dict, artifact_name=artifact_name, metadata=metadata)
    except Exception:
        logging.exception("Failed to persist new-best checkpoint; continuing training.")


def _checkpoint_resume(
    payload: Dict[str, Any],
    path: Path,
    save_artifacts: bool = True,
) -> None:
    """Persist full training state for this epoch so the run can be continued.

    Companion to :func:`_checkpoint_best`, which uploads only the best weights
    to W&B in the format every attack config reads. This writes the live
    weights plus optimizer moments and counters to a local file, so a crashed
    multi-day run continues rather than restarts.

    Skipped when ``save_artifacts`` is off, like ``_checkpoint_best``:
    cost-measurement runs share the real run's artifact name, and therefore its
    resume path, so they would otherwise overwrite the real run's state.

    As with ``_checkpoint_best``, a failure here is logged and training goes on:
    losing a resume point is worse than losing an epoch, but not worth aborting
    a multi-day run over.
    """
    if not save_artifacts:
        return
    try:
        save_resume_state(payload, path)
    except Exception:
        logging.exception("Failed to persist resume state; continuing training.")


def run(**kwargs):
    """Dispatch to the AML edge-level or the synthetic node-level adversarial
    training loop based on the configured dataset name."""
    dataset = kwargs.get("dataset")
    if dataset in AML_DATASET_NAMES:
        return _run_aml(**kwargs)
    return _run_synthetic(**kwargs)


def _run_aml(
    data_dir: str,
    dataset: str,
    data_seed: int,
    seed: int,
    device: Union[str, int],
    model_type: str,
    train_type: str,
    model_params: Dict[str, Any],
    loader_params: Dict[str, Any],
    train_params: Dict[str, Any],
    attack_type: str,
    attack_params: Dict[str, Any],
    pretrain_epochs: int = 0,
    init_artifact: str | None = None,
    init_artifact_version: str = "latest",
    display_interval: int = 0,
    save_artifacts: bool = True,
    convert_rewired_currency: bool = False,
    artifact_name: str | None = None,
    resume: bool = False,
    resume_dir: str | None = None,
    threat_model: str = "unrestricted",
):
    """AML edge-level adversarial training with per-epoch PR-BCD regeneration.

    Mirrors the node-level loop: each epoch we (1) reset the train graph to its
    clean edge index, (2) run the configured AML attack on the train split to
    obtain a perturbed edge index, (3) take one optimisation step on the
    perturbed graph, and (4) re-attack the validation split for the early-
    stopping metric. Best checkpoint is selected by validation PR-AUC on the
    perturbed val graph.

    Parameters
    ----------
    convert_rewired_currency : bool, optional
        If ``True``, the per-epoch attack that generates the training
        perturbation also patches each rewired edge's Received Currency/Amount
        Received to stay consistent with its new target node, exactly as
        ``experiment_attack`` does. Without it a rewired training edge carries a
        currency its new target has never received, which the model can learn to
        detect instead of learning from the perturbed topology. By default
        ``False``, reproducing the original adversarial training behaviour.
    artifact_name : str, optional
        Explicit name for the saved checkpoint. Defaults to the name
        :func:`create_config_name` derives from the run config, which has no
        slot for ``convert_rewired_currency`` -- so a converted run would
        otherwise publish over the unconverted checkpoint of the same
        configuration. Converted runs must set this.
    resume : bool, optional
        If ``True``, continue a previous run of the same ``artifact_name`` from
        its resume file: model weights, optimizer moments, epoch counter and
        best-so-far metric are all restored, so ``max_epochs`` keeps meaning the
        total epoch budget rather than that many *more* epochs. Starts fresh
        with a warning if no resume file exists yet, which makes it safe to
        leave on in a config that may be submitted repeatedly. By default
        ``False``.
    resume_dir : str, optional
        Folder for the per-epoch resume file, written to
        ``{resume_dir}/{artifact_name}/resume.pt`` and read back when ``resume``
        is set. Local only; nothing here is uploaded to W&B. Defaults to
        ``checkpoints/`` in the current working directory. For runs that may
        continue on another machine, point it at shared storage visible to
        both.
    threat_model : str, optional
        Threat model of the per-epoch attacks (see
        :func:`robust_multignn.attacks.aml.prbcd_attack_aml`). ``"unrestricted"``
        (default) is the original setting. ``"criminal"`` rewires only the
        illicit edges of the attacked split, onto accounts with an illicit
        transaction in *that split's graph* -- the train graph holds only
        train-period edges, so no later labels leak into training. Like
        ``convert_rewired_currency``, it requires an explicit ``artifact_name``.
    """
    assert train_type == "adv", (
        f"AML adversarial training only supports train_type='adv', got {train_type!r}"
    )

    check_availability(device)
    config = locals()

    graphs: GraphDataset = load_dataset(dataset, data_dir, data_seed)
    set_seed(seed)

    edge_dim = (
        graphs.train.edge_attr.size(1) if graphs.train.edge_attr is not None else 0
    )
    hyperparams = dict(model_params)
    hyperparams.update(
        {
            "in_dim": graphs.train.x.size(1),
            "out_dim": 2 if graphs.train.y.dim() == 1 else graphs.train.y.size(1),
            "edge_dim": edge_dim,
        }
    )

    train_local_inds = mask_to_index(graphs.train.split_mask)
    val_local_inds = mask_to_index(graphs.val.split_mask)
    test_local_inds = mask_to_index(graphs.test.split_mask)

    train_params = dict(train_params)
    pos_weight = train_params.pop("pos_weight", 1.0)

    # Snapshot clean edge indices so each adversarial round starts from the
    # same baseline (the attack mutates ``graph.edge_index`` in place).
    clean_train_edge_index = graphs.train.edge_index.clone()
    clean_val_edge_index = graphs.val.edge_index.clone()

    # Currency conversion rewrites ``edge_attr``/``currency`` in place, and the
    # attack runs once per outer epoch, so without a clean copy each epoch would
    # convert the previous epoch's already-converted amounts and the rate ratios
    # would compound multiplicatively. Snapshotting is a no-op when conversion
    # is off (the tensors are never modified), so it is unconditional.
    clean_train_edge_attr = _clone_or_none(getattr(graphs.train, "edge_attr", None))
    clean_train_currency = _clone_or_none(getattr(graphs.train, "currency", None))
    clean_val_edge_attr = _clone_or_none(getattr(graphs.val, "edge_attr", None))
    clean_val_currency = _clone_or_none(getattr(graphs.val, "currency", None))

    def reset_train_graph() -> None:
        """Restore the train split to its pre-attack state."""
        graphs.train.edge_index = clean_train_edge_index.clone()
        if clean_train_edge_attr is not None:
            graphs.train.edge_attr = clean_train_edge_attr.clone()
        if clean_train_currency is not None:
            graphs.train.currency = clean_train_currency.clone()

    def reset_val_graph() -> None:
        """Restore the val split to its pre-attack state."""
        graphs.val.edge_index = clean_val_edge_index.clone()
        if clean_val_edge_attr is not None:
            graphs.val.edge_attr = clean_val_edge_attr.clone()
        if clean_val_currency is not None:
            graphs.val.currency = clean_val_currency.clone()

    eval_loader_params = {**loader_params, "shuffle": False}

    if init_artifact and pretrain_epochs > 0:
        raise ValueError(
            "init_artifact and pretrain_epochs > 0 are mutually exclusive; "
            "set pretrain_epochs to 0 when loading a baseline checkpoint"
        )

    if init_artifact:
        model, artifact_metadata = load_model_artifact(
            init_artifact,
            version=init_artifact_version,
            train_inds=train_local_inds,
            val_inds=val_local_inds,
            test_inds=test_local_inds,
        )
        for warning in validate_baseline_artifact_metadata(
            artifact_metadata,
            init_artifact=init_artifact,
            model_type=model_type,
            dataset=dataset,
            data_seed=data_seed,
            seed=seed,
            hyperparams=hyperparams,
            loader_params=loader_params,
            train_params=train_params,
        ):
            logging.warning(warning)
        model = model.to(device)
    else:
        model = create_model(
            model_type,
            hyperparams,
            train_inds=train_local_inds,
            val_inds=val_local_inds,
            test_inds=test_local_inds,
        ).to(device)

    device_obj = next(model.parameters()).device
    ego_dim = getattr(model, "ego_dim", 0)

    def make_train_loader():
        return build_edge_neighbor_loader(
            graphs.train, bidirectional=False, ego_dim=ego_dim, **loader_params
        )

    def make_val_loader():
        return build_edge_neighbor_loader(
            graphs.val,
            eval_inds=val_local_inds,
            bidirectional=False,
            ego_dim=ego_dim,
            **eval_loader_params,
        )

    if init_artifact:
        _, pre_val_pr_auc, _, _ = val_edge_epoch(model, make_val_loader(), split="val")
        logging.info(f"Init val PR-AUC (clean): {pre_val_pr_auc:.4f}")
    elif pretrain_epochs > 0:
        pretrain_params = train_params.copy()
        pretrain_params["max_epochs"] = pretrain_epochs
        with wandb_disabled():
            best_state = train_aml(
                model=model,
                train_loader=make_train_loader(),
                val_loader=make_val_loader(),
                class_weights=pos_weight,
                class_names=graphs.class_names,
                display_interval=display_interval,
                **pretrain_params,
            )
        if best_state is not None:
            model.load_state_dict(best_state)
        _, pre_val_pr_auc, _, _ = val_edge_epoch(model, make_val_loader(), split="val")
        logging.info(f"Pretrain val PR-AUC (clean): {pre_val_pr_auc:.4f}")

    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor([1.0, pos_weight]).to(device_obj)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )

    attack_fn = get_attack(attack_type)
    max_epochs = train_params["max_epochs"]
    min_epochs = train_params.get("min_epochs", 0)
    patience = train_params.get("patience", 0)

    # Built once per run and passed to every per-epoch attack. ``None`` disables
    # the patch (the attacks treat it as a no-op), reproducing the original
    # currency-inconsistent adversarial examples.
    rate_matrix = (
        build_dataset_rate_matrix(dataset, data_dir)
        if convert_rewired_currency
        else None
    )

    # Build the artifact name/metadata up front so every new-best checkpoint
    # found during training can be persisted immediately, rather than only
    # living in memory until the (possibly days-long) loop finishes.
    # An explicit ``artifact_name`` overrides the derived one. The derived name
    # has no slot for ``convert_rewired_currency``, so a converted run would
    # otherwise resolve to the same name as the unconverted run of the same
    # configuration and publish over it as the new ``:latest`` -- silently
    # invalidating every attack config that reloads that checkpoint.
    if threat_model not in THREAT_MODELS:
        raise ValueError(
            f"Unknown threat_model {threat_model!r}; expected one of {THREAT_MODELS}"
        )
    if artifact_name is None:
        if convert_rewired_currency:
            raise ValueError(
                "convert_rewired_currency=True requires an explicit "
                "artifact_name: the derived name does not encode the flag, so "
                "the run would overwrite the unconverted checkpoint of the same "
                "configuration."
            )
        if threat_model != "unrestricted":
            raise ValueError(
                f"threat_model={threat_model!r} requires an explicit "
                "artifact_name: the derived name does not encode the threat "
                "model, so the run would overwrite the unrestricted adversarially "
                "trained checkpoint of the same configuration."
            )
        artifact_name = create_config_name(
            arch=model_type,
            train_type=train_type,
            dataset=dataset,
            data_seed=data_seed,
            model_seed=seed,
            adv_train_params={
                "attack_type": attack_type,
                "attack_ratio": attack_params["attack_ratio"],
            },
        )
    metadata = {**config, "model_params": hyperparams}

    best_val_pr_auc_pert = float("-inf")
    best_epoch = 0
    best_state = None
    start_epoch = 0

    resume_path = resume_state_path(
        resume_dir if resume_dir is not None else Path.cwd() / "checkpoints",
        artifact_name,
    )
    logging.info("Resume state file: %s", resume_path)

    if resume:
        try:
            payload = load_resume_state(resume_path)
        except Exception:
            # No resume file yet (first submission), or it could not be
            # read. Starting fresh is the right fallback -- it makes ``resume``
            # safe to leave set in a config that may be submitted repeatedly.
            logging.warning(
                "resume=True but no usable resume state at %s; starting fresh.",
                resume_path,
            )
        else:
            # Restore the *live* model, not the best one: rolling back to the
            # best checkpoint would silently discard every epoch since the last
            # improvement, which under patience=10 can be ten epochs of work.
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            start_epoch = int(payload["epoch"]) + 1
            best_val_pr_auc_pert = payload.get(
                "best_val_pr_auc_pert", float("-inf")
            )
            best_epoch = int(payload.get("best_epoch", 0))
            best_state = payload.get("best_model")
            logging.info(
                "Resumed %r at epoch %d (best val PR-AUC %.4f at epoch %d)",
                artifact_name,
                start_epoch,
                best_val_pr_auc_pert,
                best_epoch,
            )
            if start_epoch >= max_epochs:
                logging.warning(
                    "Resume point (epoch %d) already reaches max_epochs=%d; "
                    "nothing left to train.",
                    start_epoch,
                    max_epochs,
                )

    for epoch in trange(
        start_epoch, max_epochs, initial=start_epoch, total=max_epochs,
        leave=False, desc="Adversarial Training",
    ):
        # --- Train-side attack: perturb training edges, then take one step. ---
        # Timed as one adversarial epoch (attack + step) for the cost ablation;
        # the separate attack/step split is logged so the attack's share of the
        # overhead is visible.
        reset_train_graph()
        with measure("adv_epoch", device) as epoch_cost:
            with measure("adv_attack", device) as attack_cost, wandb_disabled():
                adv_train_edge_index = attack_fn(
                    model=model,
                    graph=graphs.train,
                    eval_inds=train_local_inds,
                    class_names=graphs.class_names,
                    loader_params=loader_params,
                    seed_split="train",
                    display_interval=None,
                    rate_matrix=rate_matrix,
                    threat_model=threat_model,
                    **attack_params,
                )
            graphs.train.edge_index = adv_train_edge_index.cpu()
            train_loss, _, _ = train_edge_epoch(
                model, make_train_loader(), criterion, optimizer, split="train"
            )
        log_epoch_cost(epoch_cost, attack_cost, epoch=epoch)

        # --- Val-side attack: PR-AUC on perturbed val drives early stopping. ---
        reset_val_graph()
        with wandb_disabled():
            adv_val_edge_index = attack_fn(
                model=model,
                graph=graphs.val,
                eval_inds=val_local_inds,
                class_names=graphs.class_names,
                loader_params=eval_loader_params,
                seed_split="val",
                display_interval=None,
                rate_matrix=rate_matrix,
                threat_model=threat_model,
                **attack_params,
            )
        graphs.val.edge_index = adv_val_edge_index.cpu()
        val_f1_pert, val_pr_auc_pert, _, _ = val_edge_epoch(
            model, make_val_loader(), split="val"
        )

        # --- Clean val metric on the original val edges. ---
        reset_val_graph()
        val_f1_clean, val_pr_auc_clean, _, _ = val_edge_epoch(
            model, make_val_loader(), split="val"
        )

        if display_interval and epoch % display_interval == 0:
            log_epoch_metrics(
                epoch,
                {
                    "train_loss_pert": train_loss,
                    "val_pr_auc_clean": val_pr_auc_clean,
                    "val_pr_auc_pert": val_pr_auc_pert,
                },
            )

        wandb.log(
            {
                "loss/training (perturbed)": train_loss,
                "val/pr_auc_clean": val_pr_auc_clean,
                "val/pr_auc_pert": val_pr_auc_pert,
                "val/f1_clean": val_f1_clean,
                "val/f1_pert": val_f1_pert,
            },
            step=epoch,
        )

        if val_pr_auc_pert > best_val_pr_auc_pert:
            best_val_pr_auc_pert = val_pr_auc_pert
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            _checkpoint_best(best_state, artifact_name, metadata, save_artifacts)
            improved = True
        else:
            improved = False

        # Written every epoch, not only on a new best: late in training the gap
        # between improvements is exactly when a crash costs the most. Saved
        # before the early-stop check below so the last epoch is covered too.
        _checkpoint_resume(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_pr_auc_pert": best_val_pr_auc_pert,
                "best_epoch": best_epoch,
                "best_model": best_state,
            },
            resume_path,
            save_artifacts,
        )

        if (
            not improved
            and patience
            and epoch >= min_epochs
            and epoch >= best_epoch + patience
        ):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test on the clean test graph.
    test_loader = build_edge_neighbor_loader(
        graphs.test,
        eval_inds=test_local_inds,
        bidirectional=False,
        ego_dim=ego_dim,
        **eval_loader_params,
    )
    te_f1, te_pr_auc, te_precision, te_recall = val_edge_epoch(
        model, test_loader, split="test", batch_desc="Test Batches"
    )
    log_table(
        "Test Binary Metrics",
        ["f1", "pr_auc", "precision", "recall"],
        torch.tensor([te_f1, te_pr_auc, te_precision, te_recall]),
    )
    wandb.log(
        {
            "test/f1": te_f1,
            "test/pr_auc": te_pr_auc,
            "test/precision": te_precision,
            "test/recall": te_recall,
        }
    )

    # `best_state` was already persisted as a W&B artifact via
    # `_checkpoint_best` the last time it improved during training, so no
    # further save is needed here.


def _run_synthetic(
    data_dir: str,
    dataset: str,
    device: Union[str, int],
    as_sparse: bool,
    train_split: float,
    val_split: float,
    seed: int,
    model_type: str,
    train_type: str,
    model_params: Dict[str, Any],
    pretrain_epochs: int,
    train_params: Dict[str, Any],
    attack_type: str,
    attack_params: Dict[str, Any],
    display_interval: int,
    pseudo_epochs: int = None,
    save_artifacts: bool = True,
):
    """Initiates and runs the adversarial training experiment with the given configuration.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the dataset.
    dataset : str
        Name of the dataset to use.
    device : Union[str, int]
        Device to run the experiment on.
    as_sparse : bool
        Whether to use sparse tensors.
    train_split : float
        Fraction of the dataset to use for training.
    val_split : float
        Fraction of the dataset to use for validation.
    seed : int
        Random seed for reproducibility.
    model_type : str
        The type of the trained model to be attacked. Supported model types are:
            - incgnn
    train_type : str
        The training type of the model to be attacked. Supported training types are:
            - baseline: Model trained without any adversarial training.
            - adv: Model trained with adversarial training using a specific attack.
            - self_adv: Model trained with self-adversarial training.
    model_params : Dict[str, Any]
        Hyperparameters for the model.
    pretrain_epochs : int
        Number of epochs to pretrain the model before adversarial training.
    train_params : Dict[str, Any]
        Training hyperparameters.
    attack_type : str
        Type of adversarial attack to use.
    attack_params : Dict[str, Any]
        Parameters for the adversarial attack.
    display_interval : int
        Interval (in epochs) at which to display training progress.
    pseudo_epochs : int, optional
        Number of epochs to run pseudo-labeling for self-training. Only used if train_type is "self_adv". By default None.
    """
    # Lazy imports so the AML branch is not affected if these modules are missing.
    from robust_multignn.data import generate_synthetic_graph, split
    from robust_multignn.helper.utils import f1_scores
    from robust_multignn.train import train

    assert train_type in ["adv", "self_adv"], f"Unsupported train_type: {train_type}"
    if train_type == "self_adv":
        assert pseudo_epochs is not None, (
            "pseudo_epochs must be provided when train_type is 'self_adv'"
        )

    # Prepare experiment (set device, seed, etc.)
    check_availability(device=device, seed=seed)

    # Store configuration for artifact metadata
    config = locals()

    # Prepare dataset
    attr, inc, labels = generate_synthetic_graph(
        name=dataset,
        dataset_root=data_dir,
        device=device,
        as_sparse=as_sparse,
    )

    # Split dataset
    train_idx, val_idx, test_idx = split(
        labels=labels,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
    )

    # Create model
    hyperparams = dict(model_params)
    model_dims = {
        "n_features": attr.size(1),
        "n_classes": labels.max().item() + 1,
    }
    hyperparams.update(model_dims)
    model = create_model(model_type, hyperparams).to(device)

    # Optionally pretrain the model before adversarial training
    if pretrain_epochs > 0:
        pretrain_params = train_params.copy()
        pretrain_params["max_epochs"] = pretrain_epochs
        with wandb_disabled():
            train(
                model=model,
                attr=attr,
                inc=inc,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                **pretrain_params,
            )
        logits = model(attr, inc)
        pretrain_acc = f1_scores(logits, labels, test_idx)
        logging.info(f"Pretraining Accuracy: {pretrain_acc:.4f}")

    # Optionally perform self-training to generate pseudo-labels for adversarial training
    if train_type == "self_adv":
        # Train baseline model
        baseline_model = create_model(model_type, hyperparams).to(device)
        pseudo_params = train_params.copy()
        pseudo_params["max_epochs"] = pseudo_epochs
        pseudo_params["early_stopping"] = False
        with wandb_disabled():
            train(
                model=baseline_model,
                attr=attr,
                inc=inc,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                **pseudo_params,
            )

        # Generate pseudo-labels for the entire dataset using the baseline model
        pseudo_logits: torch.Tensor = baseline_model(attr, inc)
        pseudo_labels = pseudo_logits.argmax(dim=1)
        pseudo_labels[train_idx] = labels[train_idx]
        train_idx = torch.cat([train_idx, test_idx])
        train_labels = pseudo_labels

        # Log pseudo-labeling accuracy on the test set
        pseudo_acc = f1_scores(pseudo_logits, labels, test_idx)
        logging.info(f"Pseudo-labeling Accuracy: {pseudo_acc:.4f}")
    else:
        # Otherwise, use the original labels for training
        train_labels = labels

    # Set up attack function and optimizer for adversarial training
    attack_fn = get_attack(attack_type)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )

    # Build the artifact name/metadata up front so every new-best checkpoint
    # found during training can be persisted immediately, rather than only
    # living in memory until the loop finishes.
    artifact_name = create_config_name(
        arch=model_type,
        train_type=train_type,
        dataset=dataset,
        seed=seed,
        adv_train_params={
            "attack_type": attack_type,
            "attack_ratio": attack_params["attack_ratio"],
        },
    )
    metadata = {**config, "model_params": hyperparams}

    # Initialize variables for early stopping
    best_acc = 0.0
    best_epoch = 0
    best_state = None

    # Adversarial training loop
    for epoch in trange(
        train_params["max_epochs"],
        leave=False,
        desc="Adversarial Training",
    ):
        # Generate adversarial example for the training set
        model.eval()
        with wandb_disabled():
            adv_attr, adv_inc = attack_fn(
                model=model,
                attr=attr,
                inc=inc,
                labels=train_labels,
                as_sparse=as_sparse,
                attack_idx=train_idx,
                **attack_params,
            )

        # Train model on adversarial example
        model.train()
        optimizer.zero_grad()
        logits_train_pert = model(adv_attr, adv_inc)
        loss = F.cross_entropy(logits_train_pert[train_idx], train_labels[train_idx])
        loss.backward()
        optimizer.step()

        # Evaluate model on clean graph and adversarial example
        train_acc_pert = f1_scores(logits_train_pert, train_labels, train_idx)
        logits_clean = model(attr, inc)
        train_acc_clean = f1_scores(logits_clean, train_labels, train_idx)
        val_acc_clean = f1_scores(logits_clean, labels, val_idx)

        # TODO: Only do validation attack every few epochs
        # Generate another adversarial example for the validation set
        model.eval()
        with wandb_disabled():
            val_adv_attr, val_adv_inc = attack_fn(
                model=model,
                attr=attr,
                inc=inc,
                labels=labels,
                as_sparse=as_sparse,
                attack_idx=val_idx,
                **attack_params,
            )

        # Evaluate model on adversarial validation example
        logits_val_pert = model(val_adv_attr, val_adv_inc)
        val_acc_pert = f1_scores(logits_val_pert, labels, val_idx)

        # Display training progress at specified intervals
        if display_interval and epoch % display_interval == 0:
            log_epoch_metrics(
                epoch,
                {
                    "train_loss_pert": loss.item(),
                    "train_acc_clean": train_acc_clean,
                    "train_acc_pert": train_acc_pert,
                    "val_acc_clean": val_acc_clean,
                    "val_acc_pert": val_acc_pert,
                },
            )

        wandb.log(
            {
                "loss/training (perturbed)": loss.item(),
                "accuracy/training (clean)": train_acc_clean,
                "accuracy/training (perturbed)": train_acc_pert,
                "accuracy/validation (clean)": val_acc_clean,
                "accuracy/validation (perturbed)": val_acc_pert,
            },
            step=epoch,
        )

        if val_acc_pert > best_acc:
            best_acc = val_acc_pert
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            _checkpoint_best(best_state, artifact_name, metadata, save_artifacts)
        elif (
            train_params["early_stopping"]
            and epoch >= best_epoch + train_params["patience"]
        ):
            break

    # Load best model state after training. When early stopping is enabled,
    # `best_state` was already persisted as a W&B artifact via
    # `_checkpoint_best` the last time it improved, so no further save is
    # needed for that case.
    used_best_state = train_params["early_stopping"] and best_state is not None
    if used_best_state:
        model.load_state_dict(best_state)

    # Test trained model
    with torch.no_grad():
        model.eval()
        logits = model(attr, inc)
        test_accuracy = f1_scores(logits, labels, test_idx)

    # Log final test accuracy
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    wandb.log({"accuracy/test": test_accuracy})

    if not used_best_state and save_artifacts:
        # Early stopping disabled (or no epoch ever improved): the final
        # weights weren't covered by the incremental best-checkpointing
        # above, so persist them explicitly.
        save_model_artifact(
            model.state_dict(),
            artifact_name=artifact_name,
            metadata=metadata,
        )
