import logging
from typing import Any, Dict, Union

import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import trange

from experiments.common import check_availability, set_seed, wandb_disabled
from robust_multignn.attacks import get_attack
from robust_multignn.attacks.regularizers import build_incidence_loss_callbacks
from robust_multignn.data import build_neighbor_loader
from robust_multignn.loading import (
    GraphDataset,
    create_config_name,
    load_dataset,
    save_model_artifact,
)
from robust_multignn.models import create_model
from robust_multignn.train import train
from robust_multignn.utils import (
    auprc_metrics,
    compute_class_weights,
    log_epoch_metrics,
    log_table,
    train_epoch,
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
    loader_params: Dict[str, Any],
    pretrain_params: Dict[str, Any],
    adv_train_params: Dict[str, Any],
    attack_type: str,
    attack_params: Dict[str, Any],
    display_interval: int = 0,
):
    check_availability(device)
    config = locals()

    graphs: GraphDataset = load_dataset(dataset, data_dir, data_seed)
    set_seed(seed)

    edge_dim = graphs.train.edge_attr.size(1) if graphs.train.edge_attr is not None else 0
    hyperparams = dict(model_params)
    hyperparams.update({
        "in_dim": graphs.train.x.size(1),
        "out_dim": 2 if graphs.train.y.dim() == 1 else graphs.train.y.size(1),
        "edge_dim": edge_dim,
    })
    model = create_model(model_type, hyperparams).to(device)

    pos_weight = compute_class_weights(graphs.train.y)
    train_loader = build_neighbor_loader(graphs.train, **loader_params)
    val_loader = build_neighbor_loader(graphs.val, **loader_params)
    test_loader = build_neighbor_loader(graphs.test, **loader_params)

    # ---- Warmup phase: clean training with cosine schedule ----
    if pretrain_params.get("max_epochs", 0) > 0:
        with wandb_disabled():
            best_warmup_state = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=pos_weight,
                class_names=graphs.class_names,
                display_interval=display_interval,
                **pretrain_params,
            )
        if best_warmup_state is not None:
            model.load_state_dict(best_warmup_state)
        logging.info("Warmup complete.")

    # ---- Adversarial training phase ----
    attack_fn = get_attack(attack_type)
    attack_kwargs = dict(attack_params)
    regularizer_terms = attack_kwargs.pop("regularizer_terms", None)
    regularizers = build_incidence_loss_callbacks(regularizer_terms)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=adv_train_params["lr"],
        weight_decay=adv_train_params["weight_decay"],
    )
    scheduler = None
    if "scheduler_params" in adv_train_params:
        scheduler = CosineAnnealingWarmRestarts(optimizer, **adv_train_params["scheduler_params"])

    max_adv_epochs = adv_train_params["max_epochs"]

    orig_train_edge_index = graphs.train.edge_index.clone()

    for outer_epoch in trange(max_adv_epochs, leave=False, desc="Adversarial Training"):
        # Attack: generate discrete perturbed training graph
        model.eval()
        with wandb_disabled():
            perturbed_train_edge_index = attack_fn(
                model=model,
                graph=graphs.train,
                class_names=graphs.class_names,
                loader_params=loader_params,
                display_interval=0,
                regularizers=regularizers,
                **attack_kwargs,
            )

        # Train one epoch on the perturbed graph
        graphs.train.edge_index = perturbed_train_edge_index.cpu()
        pert_train_loader = build_neighbor_loader(graphs.train, **loader_params)
        train_loss, train_auprc = train_epoch(model, pert_train_loader, criterion, optimizer)
        graphs.train.edge_index = orig_train_edge_index

        # Evaluate on clean validation graph
        val_auprc = val_epoch(model, val_loader)
        val_macro_auprc = val_auprc.mean().item()

        metrics = {
            "loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **auprc_metrics("training", train_auprc, graphs.class_names),
            **auprc_metrics("validation", val_auprc, graphs.class_names),
        }

        wandb.log(metrics, step=outer_epoch)

        if display_interval and outer_epoch % display_interval == 0:
            log_epoch_metrics(
                outer_epoch,
                {
                    "train_loss": train_loss,
                    "train_macro_auprc": train_auprc.mean().item(),
                    "val_macro_auprc": val_macro_auprc,
                },
            )

        if scheduler is not None:
            scheduler.step()

    # Final test evaluation
    test_auprc = val_epoch(model, test_loader, batch_desc="Test Batches")
    log_table("Test AUPRC Scores", graphs.class_names, test_auprc)
    wandb.log(auprc_metrics("test", test_auprc, graphs.class_names))

    artifact_name = create_config_name(
        arch=model_type,
        train_type="adv",
        dataset=dataset,
        data_seed=data_seed,
        model_seed=seed,
        adv_train_params={
            "attack_type": attack_type,
            "attack_ratio": attack_kwargs.get("attack_ratio"),
        },
    )
    metadata = {**config, "model_params": hyperparams}
    save_model_artifact(model.state_dict(), artifact_name=artifact_name, metadata=metadata)
