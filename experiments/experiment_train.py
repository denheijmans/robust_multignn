from functools import partial
from typing import Any, Dict, Union

import wandb
import torch
from experiments.common import check_availability, set_seed
from robust_multignn.aml import AML_DATASET_NAMES
from robust_multignn.aml.utils import mask_to_index
from robust_multignn.data import build_edge_neighbor_loader, build_neighbor_loader
from robust_multignn.loading import (
    GraphDataset,
    create_config_name,
    load_dataset,
    save_model_artifact,
)
from robust_multignn.models import create_model
from robust_multignn.train import train, train_aml
from robust_multignn.utils import (
    auprc_metrics,
    compute_class_weights,
    log_table,
    val_edge_epoch,
    val_epoch,
)


def run(
    data_dir: str,
    dataset: str,
    data_seed: int,
    seed: int,
    device: Union[str, int],
    model_type: str,
    model_params: Dict[str, Any],
    loader_params: dict[str, Any],
    train_params: Dict[str, Any],
    display_interval: int = 0,
    save_artifacts: bool = True,
):
    """Runs the training and evaluation pipeline for a given configuration. The best
    model based on validation AUPRC is saved as a wandb artifact.

    Parameters
    ----------
    data_dir : str
        The directory where the dataset is stored.
    dataset : str
        The name of the dataset to load.
    data_seed : int
        The random seed for data generation and splitting.
    seed : int
        The random seed for reproducibility.
    device : Union[str, int]
        The device on which to run training and evaluation (e.g., "cpu", "cuda:0").
    model_type : str
        The type of model to create.
    model_params : Dict[str, Any]
        The parameters for initializing the model.
    loader_params : dict[str, Any]
        The parameters for building the neighbor sampling data loaders.
    train_params : Dict[str, Any]
        The parameters for training the model.
    display_interval : int, optional
        The interval at which to display training progress, by default 0.
        If set to 0, no intermediate metrics are displayed during training.
    """
    # Verify device availability
    check_availability(device)

    # Store configurations to later store with model artifact
    config = locals()

    # Load dataset with train/val/test splits
    graphs: GraphDataset = load_dataset(dataset, data_dir, data_seed)

    # Set random seed for reproducibility
    set_seed(seed)

    edge_dim = 0
    if graphs.train.edge_attr is not None:
        edge_dim = graphs.train.edge_attr.size(1)
    model_dims = {
        "in_dim": graphs.train.x.size(1),
        "out_dim": 2 if graphs.train.y.dim() == 1 else graphs.train.y.size(1),
        "edge_dim": edge_dim,
    }
    hyperparams = dict(model_params)
    hyperparams.update(model_dims)

    # Build the artifact name/metadata up front so every new-best checkpoint
    # found during training can be persisted immediately (see `on_new_best`
    # below), rather than only living in memory until training finishes.
    artifact_name = create_config_name(
        arch=model_type,
        train_type="baseline",
        dataset=dataset,
        data_seed=data_seed,
        model_seed=seed,
    )
    metadata = {**config, "model_params": hyperparams}
    # save_artifacts=False skips publishing entirely. Cost-measurement runs
    # train for only a few epochs but resolve to the same artifact name as the
    # real run, so without this they would publish an under-trained model as
    # the new ``:latest`` that attack configs then load.
    checkpoint_on_new_best = (
        partial(save_model_artifact, artifact_name=artifact_name, metadata=metadata)
        if save_artifacts
        else None
    )

    if dataset in AML_DATASET_NAMES:
        # ------------------------------------------------------------------
        # AML edge-level path
        # ------------------------------------------------------------------
        pos_weight = train_params.pop("pos_weight", 1.)
        train_local_inds = mask_to_index(graphs.train.split_mask)
        val_local_inds   = mask_to_index(graphs.val.split_mask)
        test_local_inds  = mask_to_index(graphs.test.split_mask)

        model = create_model(
            model_type, hyperparams,
            train_inds=train_local_inds,
            val_inds=val_local_inds,
            test_inds=test_local_inds,
        ).to(device)

        ego_dim = getattr(model, "ego_dim", 0)
        train_loader = build_edge_neighbor_loader(
            graphs.train, bidirectional=False, ego_dim=ego_dim, **loader_params
        )
        loader_params.update({"shuffle": False})
        val_loader = build_edge_neighbor_loader(
            graphs.val, eval_inds=val_local_inds, bidirectional=False,
            ego_dim=ego_dim, **loader_params
        )
        test_loader = build_edge_neighbor_loader(
            graphs.test, eval_inds=test_local_inds, bidirectional=False,
            ego_dim=ego_dim, **loader_params
        )

        best_state = train_aml(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=pos_weight,
            class_names=graphs.class_names,
            display_interval=display_interval,
            on_new_best=checkpoint_on_new_best,
            **train_params,
        )
        model.load_state_dict(best_state)

        te_f1, te_pr_auc, te_precision, te_recall = val_edge_epoch(
            model, test_loader, split="test", batch_desc="Test Batches"
        )

        # Log test metrics
        log_table(
            "Test Binary Metrics",
            ["f1", "pr_auc", "precision", "recall"],
            torch.tensor([te_f1, te_pr_auc, te_precision, te_recall]),
        )
        wandb.log({
            "test/f1": te_f1,
            "test/pr_auc": te_pr_auc,
            "test/precision": te_precision,
            "test/recall": te_recall,
        })
    else:
        model = create_model(model_type, hyperparams).to(device)

        # Compute class weights for imbalanced classification
        pos_weight = compute_class_weights(graphs.train.y)

        train_loader = build_neighbor_loader(graphs.train, **loader_params)
        val_loader = build_neighbor_loader(graphs.val, **loader_params)
        test_loader = build_neighbor_loader(graphs.test, **loader_params)

        best_state = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=pos_weight,
            class_names=graphs.class_names,
            display_interval=display_interval,
            on_new_best=checkpoint_on_new_best,
            **train_params,
        )
        model.load_state_dict(best_state)

        test_auprc = val_epoch(model, test_loader, batch_desc="Test Batches")

        # Log test metrics
        log_table("Test AUPRC Scores", graphs.class_names, test_auprc)
        metrics = auprc_metrics("test", test_auprc, graphs.class_names)
        wandb.log(metrics)

    # `best_state` was already persisted as a W&B artifact via `on_new_best`
    # the last time it improved during training (see `checkpoint_on_new_best`
    # above), so no further save is needed here.
