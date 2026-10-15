import logging
import math
from typing import Any, List

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange

import wandb
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


class Block:
    def __init__(
        self,
        graph: GraphData,
        size: tuple[int, int],
        num_pert: int,
        device: torch.device,
    ):
        self.graph = graph
        self.device = device
        self.block_num_nodes, self.block_num_edges = size
        self.block_num_nodes = min(self.block_num_nodes, graph.num_nodes)
        self.block_num_edges = min(self.block_num_edges, graph.num_edges)
        self.num_pert = num_pert
        self.edge_majority = (self.block_num_edges + 1) // 2

        if self.block_num_edges < self.num_pert:
            raise ValueError(
                "block_size[1] must be at least the perturbation budget "
                f"({self.block_num_edges} < {self.num_pert})"
            )

        self.sample()

    def _build_edge_lookup(self) -> None:
        self.edge_lookup = torch.full(
            (self.graph.num_edges,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self.edge_lookup[self.block_edges] = torch.arange(
            self.block_num_edges,
            device=self.device,
        )

    def _sample_block_nodes(self, num_edges: int) -> Tensor:
        weights = torch.ones(
            (num_edges, self.graph.num_nodes),
            device=self.device,
            dtype=torch.float32,
        )
        return torch.multinomial(
            weights,
            num_samples=self.block_num_nodes,
            replacement=False,
        )

    def _mask_block_targets(self, edge_pos: Tensor, reset_values: bool = False) -> None:
        # Check this function
        with torch.no_grad():
            global_edges = self.block_edges[edge_pos].cpu()
            src_nodes = self.graph.edge_index[0, global_edges].to(self.device)
            tgt_nodes = self.graph.edge_index[1, global_edges].to(self.device)
            edge_block_nodes = self.block_nodes[edge_pos]
            if reset_values:
                self.target_logits[edge_pos] = 0
            src_matches = (edge_block_nodes == src_nodes.unsqueeze(1)).nonzero(
                as_tuple=False
            )
            tgt_matches = (edge_block_nodes == tgt_nodes.unsqueeze(1)).nonzero(
                as_tuple=False
            )
            if src_matches.numel() > 0:
                self.target_logits[
                    edge_pos[src_matches[:, 0]],
                    src_matches[:, 1],
                ] = float("-inf")
            if tgt_matches.numel() > 0:
                self.target_logits[
                    edge_pos[tgt_matches[:, 0]],
                    tgt_matches[:, 1],
                ] = float("-inf")

    def sample(self) -> None:
        """Sample a new block of edges and reset the learnable block state."""
        self.block_edges = torch.randperm(self.graph.num_edges, device=self.device)[
            : self.block_num_edges
        ]
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

    def save_state(self) -> dict[str, Tensor]:
        return {
            "block_edges": self.block_edges.clone(),
            "block_nodes": self.block_nodes.clone(),
            "edge_probs": self.edge_probs.clone(),
            "target_logits": self.target_logits.clone(),
        }

    def load_state(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.block_edges = state["block_edges"].clone().to(self.device)
            self.block_nodes = state["block_nodes"].clone().to(self.device)
            self.edge_probs.copy_(state["edge_probs"])
            self.target_logits.copy_(state["target_logits"])
        self._build_edge_lookup()

    def project_probs(self) -> None:
        with torch.no_grad():
            project_onto_capped_simplex_(
                self.edge_probs,
                s=self.num_pert,
                surface_only=True,
            )

    def perturb_tgt_edge_inc(self, batch: GraphData) -> Tensor:
        target_probs = torch.softmax(self.target_logits, dim=1)
        batch_edge_lookup = self.edge_lookup[batch.e_id]
        active = batch_edge_lookup >= 0
        if not active.any():
            return batch.tgt_edge_inc

        dense_tgt_inc = batch.tgt_edge_inc.to_dense()
        local_edge_pos = batch_edge_lookup[active]
        edge_target_probs = target_probs[local_edge_pos]
        edge_block_nodes = self.block_nodes[local_edge_pos]
        batch_node_ids = batch.n_id.to(self.device)

        # Map edge-local node probabilities to batch-global node order.
        # Each edge has an independent block of candidate nodes.
        node_matches = edge_block_nodes.unsqueeze(2) == batch_node_ids.view(1, 1, -1)
        batch_target_probs = (edge_target_probs.unsqueeze(2) * node_matches).sum(dim=1)
        batch_edge_probs = self.edge_probs[local_edge_pos].view(1, -1)

        orig_tgt_probs = (1 - batch_edge_probs) * dense_tgt_inc[:, active]
        new_tgt_probs = batch_edge_probs * batch_target_probs.T
        dense_tgt_inc[:, active] = orig_tgt_probs + new_tgt_probs
        return dense_tgt_inc.to_sparse().coalesce()

    def resample(self) -> None:
        """Resample the lowest-probability edges in the block."""
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

        replacement_pool = torch.ones(
            self.graph.num_edges,
            device=self.device,
            dtype=torch.float32,
        )
        replacement_pool[self.block_edges[keep_mask]] = 0
        new_edges = torch.multinomial(
            replacement_pool,
            num_samples=len(resample_idx),
            replacement=False,
        )

        with torch.no_grad():
            self.edge_probs[resample_idx] = 0
            self.target_logits[resample_idx] = 0
            self.block_edges[resample_idx] = new_edges
            self.block_nodes[resample_idx] = self._sample_block_nodes(len(resample_idx))
            self._mask_block_targets(resample_idx, reset_values=True)
            self._build_edge_lookup()

    def to_full_probs(self) -> tuple[Tensor, Tensor]:
        full_edge_probs = torch.zeros(
            self.graph.num_edges,
            device=self.device,
            dtype=self.edge_probs.dtype,
        )
        full_edge_probs[self.block_edges] = self.edge_probs

        full_target_probs = torch.zeros(
            (self.graph.num_edges, self.graph.num_nodes),
            device=self.device,
            dtype=self.target_logits.dtype,
        )
        learned_target_probs = torch.softmax(self.target_logits, dim=1)
        row_idx = self.block_edges.unsqueeze(1).expand(-1, self.block_num_nodes)
        col_idx = self.block_nodes
        full_target_probs[row_idx, col_idx] = learned_target_probs
        return full_edge_probs, full_target_probs


def _resolve_block_size(
    graph: GraphData,
    num_pert: int,
    block_size: tuple[int, int] | None,
) -> tuple[int, int]:
    """Determine block size based on user input and graph properties.

    Parameters
    ----------
    graph : GraphData
        The graph data for the attack.
    num_pert : int
        The number of edges to perturb.
    block_size : tuple[int, int] | None
        The block size as (num_nodes, num_edges). If None, num_edge block is
        set to twice the perturbation budget and num_nodes to sqrt(|V|).

    Returns
    -------
    tuple[int, int]
        The resolved (block_num_nodes, block_num_edges), clamped to graph size
        and at least (1, num_pert).
    """
    if block_size is not None:
        block_num_nodes, block_num_edges = block_size
    else:
        block_num_edges = max(2 * num_pert, 1)
        block_num_nodes = int(math.sqrt(graph.num_nodes))

    block_num_edges = min(graph.num_edges, max(block_num_edges, num_pert))
    block_num_nodes = min(graph.num_nodes, max(block_num_nodes, 1))
    return block_num_nodes, block_num_edges


def prbcd_attack(
    model: nn.Module,
    graph: GraphData,
    max_epochs: int,
    lr: float,
    attack_ratio: float,
    sample_size: int,
    class_names: List[str],
    loader_params: dict[str, Any],
    max_resamples: int,
    block_size: tuple[int, int] | None = None,
    min_epochs: int = 0,
    patience: int = 0,
    display_interval: int = None,
    regularizers: list[tuple[LossCallback, float]] | None = None,
) -> Tensor:
    """Performs a PR-BCD attack in the inductive setting.

    The implementation follows the same inductive data flow as PGD: it optimizes
    soft edge-selection probabilities and target-node logits against batches from a
    NeighborLoader, then samples a discrete perturbed edge index from the best state.

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
        The number of samples to use for the final discrete perturbation.
    class_names : List[str]
        The names of the classes.
    loader_params : dict[str, Any]
        The parameters for the data loader.
    block_size : tuple[int, int] | None, optional
        Block size used for resampling. When omitted, the edge block is set to
        twice the perturbation budget and the node block uses a square-root
        heuristic based on graph size.
    max_resamples : int, optional
        Number of epochs during which block resampling is enabled, by default 150.
    min_epochs : int, optional
        Minimum number of epochs before early stopping can trigger, by default 0.
    patience : int, optional
        Number of epochs to wait before stopping the attack, by default 0.
    display_interval : int, optional
        Interval at which to display attack progress, by default None.

    Returns
    -------
    Tensor
        The perturbed edge index after the attack.
    """
    if attack_ratio == 0.0:
        return graph.edge_index

    device = next(model.parameters()).device

    num_pert = max(1, int(attack_ratio * graph.num_edges))
    logging.info(f"Perturbation budget: {num_pert}/{graph.num_edges} edges")

    resolved_block_size = _resolve_block_size(graph, num_pert, block_size)
    logging.info(
        "Block size: %s nodes / %s edges",
        resolved_block_size[0],
        resolved_block_size[1],
    )

    block = Block(
        graph=graph,
        size=resolved_block_size,
        num_pert=num_pert,
        device=device,
    )
    optimizer = torch.optim.AdamW([block.edge_probs, block.target_logits], lr=lr)
    criterion = BoundaryLoss()

    loader = build_neighbor_loader(graph, **loader_params)

    best_macro_auprc = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    resampling = True

    model.eval()
    for epoch in trange(max_epochs, leave=False, desc="Epochs"):
        avg_loss, attack_auprc = attack_epoch(
            model,
            loader,
            criterion=criterion,
            optimizer=optimizer,
            tgt_edge_inc_builder=block.perturb_tgt_edge_inc,
            post_step_hook=lambda: block.project_probs(),
            regularizers=regularizers,
        )
        attack_macro_auprc = attack_auprc.mean().item()

        metrics = {
            "loss": avg_loss,
            **auprc_metrics("attack", attack_auprc, class_names),
        }
        wandb.log(metrics, step=epoch)

        if display_interval and epoch % display_interval == 0:
            log_epoch_metrics(
                epoch,
                {
                    "attack_loss": avg_loss,
                    "attack_macro_auprc": attack_macro_auprc,
                },
            )

        if attack_macro_auprc < best_macro_auprc:
            best_macro_auprc = attack_macro_auprc
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
        full_edge_probs, full_target_probs = block.to_full_probs()

        perturbed_edge_index = sample_perturbed_edge_index(
            model=model,
            graph=graph,
            num_pert=num_pert,
            edge_probs=full_edge_probs,
            target_probs=full_target_probs,
            sample_size=sample_size,
            loader_params=loader_params,
        )

    return perturbed_edge_index
