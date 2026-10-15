import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from robust_multignn.data import GraphData
from robust_multignn.utils import (
    project_onto_capped_simplex_,
    sample_perturbed_edge_index,
)


def uniform_attack(
    model: nn.Module,
    graph: GraphData,
    attack_ratio: float,
    sample_size: int,
    loader_params: dict[str, Any],
    **_,
) -> Tensor:
    """Performs a uniform attack on the given model and graph data.

    Parameters
    ----------
    model : nn.Module
        The model to attack.
    graph : GraphData
        The graph data for the attack.
    attack_ratio : float
        The ratio of edges to perturb.
    sample_size : int
        The number of samples to use for the attack.
    loader_params : dict, optional
        The parameters for the data loader, by default None

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

    # Per edge probabilities of perturbation
    edge_probs: Tensor = torch.zeros(
        (graph.num_edges,),
        device=device,
        dtype=torch.float32,
    )

    # Target logits for all possible target nodes
    target_logits: Tensor = torch.zeros(
        (graph.num_edges, graph.num_nodes),
        device=device,
        dtype=torch.float32,
    )
    edge_ids = torch.arange(graph.num_edges, device=device)
    target_logits[edge_ids, graph.edge_index[0]] = float("-inf")
    target_logits[edge_ids, graph.edge_index[1]] = float("-inf")

    # Project and sample hard perturbations from the best attack parameters.
    project_onto_capped_simplex_(edge_probs, s=num_pert, surface_only=True)
    target_probs = torch.softmax(target_logits, dim=1)
    perturbed_edge_index = sample_perturbed_edge_index(
        model=model,
        graph=graph,
        num_pert=num_pert,
        edge_probs=edge_probs,
        target_probs=target_probs,
        sample_size=sample_size,
        loader_params=loader_params,
    )

    return perturbed_edge_index
