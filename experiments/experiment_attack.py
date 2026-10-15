from typing import Any, Dict, Union

import torch
from torch import Tensor

import wandb
from experiments.common import check_availability, set_seed
from robust_multignn.aml import AML_DATASET_NAMES
from robust_multignn.aml.exchange_rates import build_dataset_rate_matrix
from robust_multignn.aml.utils import mask_to_index
from robust_multignn.attacks import get_attack
from robust_multignn.attacks.aml import THREAT_MODELS
from robust_multignn.attacks.regularizers import (
    build_incidence_loss_callbacks,
    cycle_loss,
    cycle_loss_sparse,
    degree_loss,
)
from robust_multignn.data import build_edge_neighbor_loader, build_neighbor_loader
from robust_multignn.profiling import Measurement, measure, predicted_block_bytes
from robust_multignn.loading import (
    GraphDataset,
    create_config_name,
    load_dataset,
    load_model_artifact,
)
from robust_multignn.utils import (
    auprc_metrics,
    log_table,
    val_edge_epoch,
    val_epoch,
)


def log_attack_cost(
    cost: Measurement,
    block_size: Any = None,
) -> None:
    """Log wall-clock and peak memory for one attack to W&B.

    Recorded per attack rather than relying on W&B's system metrics, which
    sample the whole process and so cannot separate the attack from data
    loading or evaluation. ``cost/predicted_block_gb`` is the analytic size of
    PR-BCD's learnable block state, which depends on the block and not on the
    graph -- the quantity the scalability claim is about.
    """
    metrics = {
        "cost/attack_seconds": cost.seconds,
        "cost/host_peak_gb": cost.host_peak_bytes / 1e9,
    }
    if cost.gpu_peak_bytes is not None:
        metrics["cost/gpu_peak_gb"] = cost.gpu_peak_bytes / 1e9
    if cost.gpu_reserved_bytes is not None:
        metrics["cost/gpu_reserved_gb"] = cost.gpu_reserved_bytes / 1e9
    if block_size is not None:
        block_num_nodes, block_num_edges = block_size
        metrics["cost/block_num_nodes"] = block_num_nodes
        metrics["cost/block_num_edges"] = block_num_edges
        metrics["cost/predicted_block_gb"] = (
            predicted_block_bytes(block_num_edges, block_num_nodes) / 1e9
        )
    wandb.log(metrics)


def run(
    data_dir: str,
    dataset: str,
    data_seed: int,
    model_seed: int,
    seed: int,
    device: Union[str, int],
    model_type: str,
    train_type: str,
    attack_type: str,
    attack_params: Dict[str, Any],
    loader_params: Dict[str, Any],
    display_interval: int = 0,
    adv_train_params: Dict[str, Any] = None,
    model_dataset: str | None = None,
    convert_rewired_currency: bool = False,
    currency_aware_optimization: bool = False,
    artifact_name: str | None = None,
    threat_model: str = "unrestricted",
):
    """Runs the specified attack on the model and evaluates
    its performance on the perturbed graph.

    Parameters
    ----------
    data_dir : str
        The directory where the dataset is stored.
    dataset : str
        The name of the dataset.
    data_seed : int
        The random seed for data shuffling.
    model_seed : int
        The random seed for model initialization.
    seed : int
        The random seed for reproducibility.
    device : Union[str, int]
        The device on which to run the model.
    model_type : str
        The type of the model.
    train_type : str
        The training type for the model.
    attack_type : str
        The type of attack to perform.
    attack_params : Dict[str, Any]
        The parameters for the attack.
    loader_params : Dict[str, Any]
        The parameters for the data loader.
    display_interval : int, optional
        The interval at which to display progress, by default 0
    adv_train_params : Dict[str, Any], optional
        The parameters for adversarial training, by default None
    convert_rewired_currency : bool, optional
        AML only. If ``True``, rewired edges that were already cross-currency
        have their Received Currency/Amount Received patched to stay
        consistent with their new target node's currency (see
        :func:`robust_multignn.aml.exchange_rates.build_dataset_rate_matrix`).
        By default ``False``, matching the original behaviour where a
        rewired edge keeps its pre-rewire currency/amount unchanged.
    currency_aware_optimization : bool, optional
        AML PR-BCD only, and only meaningful with ``convert_rewired_currency``
        on. If ``True``, the attack also optimizes against expected-value
        currency-converted features, so its learned perturbation distribution
        accounts for the conversion instead of only the final hard samples.
        By default ``False``, which leaves the PR-BCD optimization exactly as
        originally formulated (required to reproduce earlier results).
    artifact_name : str, optional
        Explicit name of the model artifact to attack. Defaults to the name
        derived from the run config by :func:`create_config_name`; pass it to
        reach a checkpoint that format cannot express, such as an adversarially
        trained model saved with ``convert_rewired_currency`` on.
    threat_model : str, optional
        AML only. ``"unrestricted"`` (default): any test edge may be rewired
        onto any account, the original setting. ``"criminal"``: the
        launderers rewire only their own (illicit) test transactions, onto
        accounts that sent or received illicit funds in any period;
        ``attack_ratio`` is then a fraction of the illicit test edges, and
        PR-BCD optimizes an evasion objective (see
        :func:`robust_multignn.attacks.aml.prbcd_attack_aml`).
    """
    if threat_model not in THREAT_MODELS:
        raise ValueError(
            f"Unknown threat_model {threat_model!r}; expected one of {THREAT_MODELS}"
        )
    if threat_model != "unrestricted" and dataset not in AML_DATASET_NAMES:
        raise ValueError("threat_model is only supported for AML datasets")

    # Verify device availability
    check_availability(device)

    # Load dataset with train/val/test splits
    graphs: GraphDataset = load_dataset(dataset, data_dir, data_seed)
    artifact_dataset = model_dataset or dataset

    # Set random seed for reproducibility
    set_seed(seed)

    # Build artifact name shared across both branches. An explicit
    # ``artifact_name`` in the config wins, which is how attacks target
    # checkpoints whose name the derived format cannot express -- e.g. the
    # ``-cc`` models adversarially trained with currency conversion.
    if artifact_name is None:
        artifact_name = create_config_name(
            arch=model_type,
            train_type=train_type,
            dataset=artifact_dataset,
            data_seed=data_seed,
            model_seed=model_seed,
            adv_train_params=adv_train_params,
        )
    print(f"[experiment_attack] Loading artifact: {artifact_name}")

    # Resolve attack function and regularizers
    attack_fn = get_attack(attack_type)
    attack_kwargs = dict(attack_params)
    regularizer_terms = attack_kwargs.pop("regularizer_terms", None)
    regularizers = build_incidence_loss_callbacks(regularizer_terms)

    if dataset in AML_DATASET_NAMES:
        # ------------------------------------------------------------------
        # AML edge-level path
        # ------------------------------------------------------------------
        train_local_inds = mask_to_index(graphs.train.split_mask)
        val_local_inds = mask_to_index(graphs.val.split_mask)
        test_local_inds = mask_to_index(graphs.test.split_mask)

        # Reconstruct the line-graph model with split index buffers; their
        # values are overwritten by load_state_dict below.
        model, _ = load_model_artifact(
            artifact_name,
            train_inds=train_local_inds,
            val_inds=val_local_inds,
            test_inds=test_local_inds,
        )
        model = model.to(device)

        orig_edge_index = graphs.test.edge_index.clone()

        # Rewired edges whose new target has never received in the edge's
        # currency get their Received Currency/Amount Received converted into
        # one the target does use -- otherwise a rewired edge trivially
        # reveals itself by carrying a currency its new target never uses.
        # Built once per run. Applied to the hard samples; with
        # ``currency_aware_optimization`` also folded into PR-BCD's
        # optimization. ``rate_matrix=None`` disables the patch (attack_fn
        # treats it as a no-op), reproducing the old, currency-inconsistent
        # behaviour.
        rate_matrix = (
            build_dataset_rate_matrix(dataset, data_dir)
            if convert_rewired_currency
            else None
        )

        # Run attack restricted to test-period edges. Wrapped in ``measure`` so
        # the cost ablation gets a per-attack wall-clock and peak-memory figure;
        # W&B's own system metrics are process-wide samples and cannot be
        # attributed to the attack alone.
        with measure("attack", device) as attack_cost:
            adv_edge_index: Tensor = attack_fn(
                model=model,
                graph=graphs.test,
                eval_inds=test_local_inds,
                class_names=graphs.class_names,
                loader_params=loader_params,
                display_interval=display_interval,
                regularizers=regularizers,
                rate_matrix=rate_matrix,
                currency_aware_optimization=currency_aware_optimization,
                threat_model=threat_model,
                **attack_kwargs,
            )
        log_attack_cost(attack_cost, attack_kwargs.get("block_size"))

        # Evaluate the model on the perturbed graph
        graphs.test.edge_index = adv_edge_index.cpu()
        eval_loader_params = {**loader_params, "shuffle": False}
        test_loader = build_edge_neighbor_loader(
            graphs.test,
            eval_inds=test_local_inds,
            bidirectional=False,
            ego_dim=getattr(model, "ego_dim", 0),
            **eval_loader_params,
        )
        te_f1, te_pr_auc, te_precision, te_recall = val_edge_epoch(
            model, test_loader, split="test", batch_desc="Test Batches"
        )

        # Log test metrics on console and to Weights & Biases
        log_table(
            "Test Binary Metrics",
            ["f1", "pr_auc", "precision", "recall"],
            torch.tensor([te_f1, te_pr_auc, te_precision, te_recall]),
        )

        graph_device = next(model.parameters()).device
        num_nodes = int(graphs.test.num_nodes)
        orig_edge_index = orig_edge_index.to(device=graph_device, dtype=torch.long)
        adv_edge_index = adv_edge_index.to(device=graph_device, dtype=torch.long)
        orig_vals = torch.ones(orig_edge_index.size(1), device=graph_device)
        pert_vals = torch.ones(adv_edge_index.size(1), device=graph_device)
        orig_adj = torch.sparse_coo_tensor(
            orig_edge_index,
            orig_vals,
            (num_nodes, num_nodes),
            device=graph_device,
        ).coalesce()
        pert_adj = torch.sparse_coo_tensor(
            adv_edge_index,
            pert_vals,
            (num_nodes, num_nodes),
            device=graph_device,
        ).coalesce()
        metrics = {
            "test/f1": te_f1,
            "test/pr_auc": te_pr_auc,
            "test/precision": te_precision,
            "test/recall": te_recall,
        }
        # Always report the discrete cycle-count error as a noticeability
        # metric, independent of whether the regularizer was active during
        # the attack — needed to compare regularized vs. unregularized runs.
        metrics["test/cycle_loss"] = float(
            cycle_loss_sparse(pert_adj, orig_adj, normalize="clamp").item()
        )
        wandb.log(metrics)
    else:
        # ------------------------------------------------------------------
        # Non-AML node-level path
        # ------------------------------------------------------------------
        model, _ = load_model_artifact(artifact_name)
        model = model.to(device)

        orig_edge_index = graphs.test.edge_index.clone()

        # Run attack to perturb the graph structure
        adv_edge_index = attack_fn(
            model=model,
            graph=graphs.test,
            class_names=graphs.class_names,
            loader_params=loader_params,
            display_interval=display_interval,
            regularizers=regularizers,
            **attack_kwargs,
        )

        # Evaluate the model on the perturbed graph
        graphs.test.edge_index = adv_edge_index.cpu()
        test_loader = build_neighbor_loader(graphs.test, **loader_params)
        test_auprc = val_epoch(model, test_loader, batch_desc="Test Batches")

        # Log test metrics on console and to Weights & Biases
        metrics = auprc_metrics("test", test_auprc, graphs.class_names)
        log_table("Test AUPRC Scores", graphs.class_names, test_auprc)

        graph_device = next(model.parameters()).device
        num_nodes = int(graphs.test.num_nodes)
        orig_edge_index = orig_edge_index.to(device=graph_device, dtype=torch.long)
        adv_edge_index = adv_edge_index.to(device=graph_device, dtype=torch.long)
        orig_vals = torch.ones(orig_edge_index.size(1), device=graph_device)
        pert_vals = torch.ones(adv_edge_index.size(1), device=graph_device)
        orig_adj = torch.sparse_coo_tensor(
            orig_edge_index,
            orig_vals,
            (num_nodes, num_nodes),
            device=graph_device,
        ).coalesce()
        pert_adj = torch.sparse_coo_tensor(
            adv_edge_index,
            pert_vals,
            (num_nodes, num_nodes),
            device=graph_device,
        ).coalesce()
        metrics["test/cycle_loss"] = float(
            cycle_loss(pert_adj, orig_adj, normalize="clamp").item()
        )
        metrics["test/degree_loss"] = float(degree_loss(pert_adj, orig_adj).item())

        wandb.log(metrics)
