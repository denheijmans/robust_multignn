import logging
from typing import Any, List

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange

import wandb
from robust_multignn.attacks.regularizers import LossCallback
from robust_multignn.data import GraphData, build_neighbor_loader
from robust_multignn.attacks.regularizers import LossCallback
from robust_multignn.utils import (
    BoundaryLoss,
    attack_epoch,
    auprc_metrics,
    log_epoch_metrics,
    project_onto_capped_simplex_,
    sample_perturbed_edge_index,
)


def pgd_attack(
    model: nn.Module,
    graph: GraphData,
    max_epochs: int,
    lr: float,
    attack_ratio: float,
    sample_size: int,
    class_names: List[str],
    loader_params: dict[str, Any],
    min_epochs: int = 0,
    patience: int = 0,
    display_interval: int = None,
    regularizers: list[tuple[LossCallback, float]] | None = None,
    **_,
) -> Tensor:
    """Performs a PGD attack on the given model and data loader.

    Parameters
    ----------
    model : nn.Module
        The model to attack.
    graph : GraphData
        The graph data for the attack.
    max_epochs : int
        The maximum number of epochs to run the attack.
    lr : float
        The learning rate for the optimizer.
    attack_ratio : float
        The ratio of edges to perturb.
    sample_size : int
        The number of samples to use for the attack.
    class_names : List[str]
        The names of the classes.
    min_epochs : int, optional
        The minimum number of epochs to run the attack, by default 0
    patience : int, optional
        The number of epochs to wait before stopping the attack, by default 0
    loader_params : dict, optional
        The parameters for the data loader, by default None
    display_interval : int, optional
        The interval at which to display attack progress, by default None

    Returns
    -------
    Tensor
        The perturbed edge index after the attack.
    """
    # Get the graph data and device from the model
    device = next(model.parameters()).device

    # Return original edge index if no attack
    if attack_ratio == 0.0:
        return graph.edge_index

    # Calculate the perturbation budget
    num_pert = max(1, int(attack_ratio * graph.num_edges))  # Perturb at least one edge
    logging.info(f"Perturbation budget: {num_pert}/{graph.num_edges} edges")

    # Per edge probabilities of perturbation (initialized to zero for all edges)
    edge_probs: Tensor = torch.zeros(
        (graph.num_edges,),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    # Target logits for all possible target nodes
    target_logits: Tensor = torch.zeros(
        (graph.num_edges, graph.num_nodes),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    with torch.no_grad():
        edge_ids = torch.arange(graph.num_edges, device=device)
        target_logits[edge_ids, graph.edge_index[0]] = float("-inf")
        target_logits[edge_ids, graph.edge_index[1]] = float("-inf")

    # Optimize edge probabilities and target logits
    optimizer = torch.optim.AdamW([edge_probs, target_logits], lr=lr)
    criterion = BoundaryLoss()

    # Create a data loader for the attack epochs
    loader = build_neighbor_loader(graph, **loader_params)

    def perturb_tgt_edge_inc(batch: GraphData) -> Tensor:
        """Construct a soft perturbed target-edge incidence matrix for the given batch
        based on current edge probabilities and target logits.

        Parameters
        ----------
        batch : GraphData
            The input batch for which to construct the perturbation

        Returns
        -------
        Tensor
            Perturbed target-edge incidence matrix
            with shape :obj:`[num_nodes, num_edges]`.
        """
        # Compute the edge and target probabilities for the current batch
        target_probs = torch.softmax(target_logits, dim=1)
        b_target_probs = target_probs[batch.e_id][:, batch.n_id]
        b_edge_probs = edge_probs[batch.e_id].view(1, -1)

        # Compute probabilities for the original and new target nodes
        orig_tgt_probs = (1 - b_edge_probs) * batch.tgt_edge_inc
        new_tgt_probs = b_edge_probs * b_target_probs.T

        # Combine all probs in a soft perturbed target-edge incidence matrix
        return new_tgt_probs + orig_tgt_probs

    def project_edge_probs():
        """Constrain edge probabilities to satisfy perturbation budget"""
        with torch.no_grad():
            project_onto_capped_simplex_(edge_probs, s=num_pert, surface_only=True)

    # Attack training loop
    best_macro_auprc = float("inf")
    best_epoch = 0
    best_edge_probs = None
    best_target_logits = None
    for epoch in trange(max_epochs, leave=False, desc="Epochs"):
        # Do one attack epoch
        avg_loss, attack_auprc = attack_epoch(
            model,
            loader,
            criterion=criterion,
            optimizer=optimizer,
            tgt_edge_inc_builder=perturb_tgt_edge_inc,
            post_step_hook=project_edge_probs,
            regularizers=regularizers,
        )
        attack_macro_auprc = attack_auprc.mean().item()

        # Log all metrics to Weights & Biases
        metrics = {
            "loss": avg_loss,
            **auprc_metrics("attack", attack_auprc, class_names),
        }
        wandb.log(metrics, step=epoch)

        # Log loss and macro scores to console at display intervals
        if display_interval and epoch % display_interval == 0:
            log_epoch_metrics(
                epoch,
                {
                    "attack_loss": avg_loss,
                    "attack_macro_auprc": attack_macro_auprc,
                },
            )

        # Check for best attack and apply early stopping based on patience
        if attack_macro_auprc < best_macro_auprc:
            best_macro_auprc = attack_macro_auprc
            best_epoch = epoch
            best_edge_probs = edge_probs.clone()
            best_target_logits = target_logits.clone()
        elif patience and epoch >= min_epochs and epoch >= best_epoch + patience:
            break

    # Project and sample hard perturbations from the best attack parameters.
    with torch.no_grad():
        project_onto_capped_simplex_(best_edge_probs, s=num_pert, surface_only=True)
        best_target_probs = torch.softmax(best_target_logits, dim=1)
        perturbed_edge_index = sample_perturbed_edge_index(
            model=model,
            graph=graph,
            num_pert=num_pert,
            edge_probs=best_edge_probs,
            target_probs=best_target_probs,
            sample_size=sample_size,
            loader_params=loader_params,
        )

    return perturbed_edge_index
