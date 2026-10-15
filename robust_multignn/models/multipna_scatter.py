"""Scatter-based PNA message passing for AML edge classification.

The clean/hard-graph path implements the PNA aggregators exactly with scatter
reductions.  During a PR-BCD attack, the destination incidence contains
continuous weights.  Mean and standard deviation then become weighted moments,
while min/max use a weighted log-sum-exp relaxation so gradients can reach the
attack probabilities without materialising a dense node-by-edge matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import BatchNorm
from torch_geometric.typing import OptTensor

from robust_multignn.data import GraphDataBatch, compute_batch_ports
from robust_multignn.models.model_utils import (
    _aggregate_to_edges,
    _sparse_to_messages,
)
from robust_multignn.models.multigin_scatter import EdgePairPredictionHead


DEFAULT_AGGREGATORS = ("mean", "min", "max", "std")
DEFAULT_SCALERS = ("identity", "amplification", "attenuation")
SUPPORTED_AGGREGATORS = {"sum", "mean", "min", "max", "var", "std"}
SUPPORTED_SCALERS = {
    "identity",
    "amplification",
    "attenuation",
    "linear",
    "inverse_linear",
}


def compute_pna_degree_histograms(
    edge_index: Tensor,
    num_nodes: int,
) -> dict[str, list[int]]:
    """Compute full-training-graph degree histograms for directed PNA paths.

    The forward path aggregates at destinations and therefore uses the
    in-degree histogram.  Reverse message passing aggregates at sources and
    uses the out-degree histogram.  Returning plain lists keeps checkpoint
    metadata JSON-compatible and deterministic across devices.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    edge_index = edge_index.detach().cpu()

    def histogram(node_ids: Tensor) -> list[int]:
        degrees = torch.bincount(node_ids, minlength=num_nodes)
        return torch.bincount(degrees, minlength=1).tolist()

    return {
        "in_degree_histogram": histogram(edge_index[1]),
        "out_degree_histogram": histogram(edge_index[0]),
    }


def _as_names(
    values: Sequence[str] | str | None,
    default: tuple[str, ...],
    supported: set[str],
    label: str,
) -> tuple[str, ...]:
    if values is None:
        names = default
    elif isinstance(values, str):
        names = (values,)
    else:
        names = tuple(str(value) for value in values)
    if not names:
        raise ValueError(f"{label} must not be empty")
    unknown = sorted(set(names).difference(supported))
    if unknown:
        raise ValueError(
            f"Unsupported PNA {label}: {unknown}; supported values are "
            f"{sorted(supported)}"
        )
    return names


def _degree_averages(histogram: Sequence[int] | Tensor) -> tuple[float, float]:
    hist = torch.as_tensor(histogram, dtype=torch.float64)
    if hist.dim() != 1 or hist.numel() == 0:
        raise ValueError("A PNA degree histogram must be a non-empty 1-D sequence")
    if bool((hist < 0).any()) or float(hist.sum()) <= 0:
        raise ValueError("A PNA degree histogram must contain non-negative counts")
    degree = torch.arange(hist.numel(), dtype=torch.float64)
    count = hist.sum()
    avg_linear = float((degree * hist).sum() / count)
    avg_log = float((torch.log1p(degree) * hist).sum() / count)
    # Real AML training graphs contain edges, but keep degenerate toy graphs
    # numerically defined for unit tests and programmatic callers.
    return max(avg_linear, 1e-12), max(avg_log, 1e-12)


def _scatter_sum(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = values.new_zeros((dim_size, values.size(1)))
    return out.index_add(0, index, values)


def _scatter_extreme(
    messages: Tensor,
    index: Tensor,
    dim_size: int,
    reduce: str,
) -> Tensor:
    fill = float("-inf") if reduce == "amax" else float("inf")
    out = messages.new_full((dim_size, messages.size(1)), fill)
    expanded_index = index.view(-1, 1).expand_as(messages)
    out = out.scatter_reduce(
        0,
        expanded_index,
        messages,
        reduce=reduce,
        include_self=True,
    )
    counts = messages.new_zeros(dim_size).index_add(
        0, index, messages.new_ones(index.numel())
    )
    return out.masked_fill(counts.view(-1, 1) == 0, 0.0)


def _smooth_weighted_extreme(
    messages: Tensor,
    index: Tensor,
    weights: Tensor,
    dim_size: int,
    temperature: float,
    maximize: bool,
) -> Tensor:
    """Differentiable weighted approximation of a per-node min or max."""
    signed = messages if maximize else -messages
    eps = 1e-12
    positive_weights = weights.clamp_min(eps)
    scores = temperature * signed + positive_weights.log().unsqueeze(1)

    max_scores = scores.new_full((dim_size, scores.size(1)), float("-inf"))
    expanded_index = index.view(-1, 1).expand_as(scores)
    max_scores = max_scores.scatter_reduce(
        0,
        expanded_index,
        scores,
        reduce="amax",
        include_self=True,
    )
    exp_sum = _scatter_sum(
        torch.exp(scores - max_scores[index]),
        index,
        dim_size,
    )
    mass = weights.new_zeros(dim_size).index_add(0, index, weights)
    log_mass = mass.clamp_min(eps).log().unsqueeze(1)
    out = (max_scores + exp_sum.clamp_min(eps).log() - log_mass) / temperature
    out = out.masked_fill(mass.view(-1, 1) <= 0, 0.0)
    return out if maximize else -out


def weighted_pna_aggregate(
    messages: Tensor,
    index: Tensor,
    weights: Tensor,
    dim_size: int,
    aggregators: Sequence[str],
    scalers: Sequence[str],
    avg_degree_linear: Tensor,
    avg_degree_log: Tensor,
    soft_extrema_temperature: float,
) -> Tensor:
    """Aggregate weighted messages and apply the standard PNA degree scalers.

    For all-one, non-differentiable incidence weights this uses exact min/max,
    matching a hard graph.  Gradient-carrying weights select the differentiable
    relaxation used only by the soft PR-BCD optimization phase.
    """
    if messages.dim() != 2:
        raise ValueError("messages must have shape [num_messages, channels]")
    if index.dim() != 1 or weights.dim() != 1:
        raise ValueError("index and weights must be one-dimensional")
    if messages.size(0) != index.numel() or index.numel() != weights.numel():
        raise ValueError("messages, index, and weights must have equal length")
    if soft_extrema_temperature <= 0:
        raise ValueError("soft_extrema_temperature must be positive")

    degree = weights.new_zeros(dim_size).index_add(0, index, weights)
    denominator = degree.clamp_min(1e-12).unsqueeze(1)
    weighted_sum = _scatter_sum(messages * weights.unsqueeze(1), index, dim_size)

    need_mean = any(name in {"mean", "var", "std"} for name in aggregators)
    mean = weighted_sum / denominator if need_mean else None
    if mean is not None:
        mean = mean.masked_fill(degree.view(-1, 1) <= 0, 0.0)

    need_variance = any(name in {"var", "std"} for name in aggregators)
    variance = None
    if need_variance:
        second_moment = _scatter_sum(
            messages.square() * weights.unsqueeze(1), index, dim_size
        ) / denominator
        variance = second_moment - mean.square()

    exact_extrema = not (torch.is_grad_enabled() and weights.requires_grad)
    aggregated: list[Tensor] = []
    for aggregator in aggregators:
        if aggregator == "sum":
            out = weighted_sum
        elif aggregator == "mean":
            out = mean
        elif aggregator == "var":
            out = variance
        elif aggregator == "std":
            # Match PyG StdAggregation: avoid the undefined derivative at zero
            # and report an exact zero below its numerical threshold.
            out = variance.clamp(min=1e-5).sqrt()
            out = out.masked_fill(out <= 1e-5**0.5, 0.0)
        elif aggregator in {"min", "max"}:
            if exact_extrema:
                out = _scatter_extreme(
                    messages,
                    index,
                    dim_size,
                    "amin" if aggregator == "min" else "amax",
                )
            else:
                out = _smooth_weighted_extreme(
                    messages,
                    index,
                    weights,
                    dim_size,
                    soft_extrema_temperature,
                    maximize=aggregator == "max",
                )
        else:  # guarded in the model constructor
            raise ValueError(f"Unsupported PNA aggregator: {aggregator}")
        aggregated.append(out)

    combined = torch.cat(aggregated, dim=-1)
    avg_linear = avg_degree_linear.to(device=degree.device, dtype=degree.dtype)
    avg_log = avg_degree_log.to(device=degree.device, dtype=degree.dtype)
    log_degree = torch.log1p(degree)

    scaled: list[Tensor] = []
    for scaler in scalers:
        if scaler == "identity":
            out = combined
        elif scaler == "amplification":
            out = combined * (log_degree / avg_log).unsqueeze(1)
        elif scaler == "attenuation":
            factor = avg_log / torch.log1p(degree.clamp_min(1.0))
            out = combined * factor.unsqueeze(1)
        elif scaler == "linear":
            out = combined * (degree / avg_linear).unsqueeze(1)
        elif scaler == "inverse_linear":
            out = combined * (avg_linear / degree.clamp_min(1.0)).unsqueeze(1)
        else:  # guarded in the model constructor
            raise ValueError(f"Unsupported PNA scaler: {scaler}")
        scaled.append(out)
    return torch.cat(scaled, dim=-1)


def _mlp(in_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("PNA pre_layers and post_layers must be at least 1")
    modules: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    for _ in range(num_layers - 1):
        modules.extend([nn.ReLU(), nn.Linear(out_dim, out_dim)])
    return nn.Sequential(*modules)


class WeightedPNAPath(nn.Module):
    """One directed PNA convolution path over a sparse weighted incidence."""

    def __init__(
        self,
        hidden_dim: int,
        degree_histogram: Sequence[int] | Tensor,
        aggregators: Sequence[str],
        scalers: Sequence[str],
        towers: int,
        pre_layers: int,
        post_layers: int,
        soft_extrema_temperature: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregators = tuple(aggregators)
        self.scalers = tuple(scalers)
        self.towers = towers
        self.soft_extrema_temperature = soft_extrema_temperature
        tower_out_dim = hidden_dim // towers

        avg_linear, avg_log = _degree_averages(degree_histogram)
        self.register_buffer("avg_degree_linear", torch.tensor(avg_linear))
        self.register_buffer("avg_degree_log", torch.tensor(avg_log))

        # PNAConv applies a layer-local edge projection before its pre-MLPs,
        # even when the model-level edge embedding already has hidden_dim.
        self.edge_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.pre_nns = nn.ModuleList(
            [_mlp(3 * hidden_dim, hidden_dim, pre_layers) for _ in range(towers)]
        )
        post_in_dim = (len(self.aggregators) * len(self.scalers) + 1) * hidden_dim
        self.post_nns = nn.ModuleList(
            [_mlp(post_in_dim, tower_out_dim, post_layers) for _ in range(towers)]
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        node_attr: Tensor,
        edge_attr: Tensor,
        receiver_node: Tensor,
        receiver_edge: Tensor,
        receiver_weight: Tensor,
        sender_by_edge: Tensor,
        num_nodes: int,
    ) -> Tensor:
        receiver = node_attr[receiver_node]
        sender = sender_by_edge[receiver_edge]
        edge = self.edge_encoder(edge_attr[receiver_edge])
        outputs: list[Tensor] = []
        for pre_nn, post_nn in zip(self.pre_nns, self.post_nns):
            messages = pre_nn(torch.cat([receiver, sender, edge], dim=-1))
            aggregated = weighted_pna_aggregate(
                messages,
                receiver_node,
                receiver_weight,
                num_nodes,
                self.aggregators,
                self.scalers,
                self.avg_degree_linear,
                self.avg_degree_log,
                self.soft_extrema_temperature,
            )
            outputs.append(post_nn(torch.cat([node_attr, aggregated], dim=-1)))
        return self.lin(torch.cat(outputs, dim=-1))


class MultiPNALayer(nn.Module):
    """PNA node update with optional reverse message passing and edge update."""

    def __init__(
        self,
        hidden_dim: int,
        in_degree_histogram: Sequence[int] | Tensor,
        out_degree_histogram: Sequence[int] | Tensor,
        aggregators: Sequence[str],
        scalers: Sequence[str],
        towers: int,
        pre_layers: int,
        post_layers: int,
        soft_extrema_temperature: float,
        reverse_mp: bool,
        edge_updates: bool,
    ) -> None:
        super().__init__()
        self.reverse_mp = reverse_mp
        self.edge_updates = edge_updates
        self.fwd = WeightedPNAPath(
            hidden_dim,
            in_degree_histogram,
            aggregators,
            scalers,
            towers,
            pre_layers,
            post_layers,
            soft_extrema_temperature,
        )
        self.rev = None
        if reverse_mp:
            self.rev = WeightedPNAPath(
                hidden_dim,
                out_degree_histogram,
                aggregators,
                scalers,
                towers,
                pre_layers,
                post_layers,
                soft_extrema_temperature,
            )
        self.bn = BatchNorm(hidden_dim)
        if edge_updates:
            self.edge_mlp = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(
        self,
        node_attr: Tensor,
        edge_attr: Tensor,
        src_node: Tensor,
        src_edge: Tensor,
        src_weight: Tensor,
        dst_node: Tensor,
        dst_edge: Tensor,
        dst_weight: Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> Tuple[Tensor, Tensor]:
        source_by_edge = _aggregate_to_edges(
            src_weight, src_node, src_edge, node_attr, num_edges
        )
        target_by_edge = _aggregate_to_edges(
            dst_weight, dst_node, dst_edge, node_attr, num_edges
        )

        node_fwd = self.fwd(
            node_attr,
            edge_attr,
            dst_node,
            dst_edge,
            dst_weight,
            source_by_edge,
            num_nodes,
        )
        if self.rev is not None:
            node_rev = self.rev(
                node_attr,
                edge_attr,
                src_node,
                src_edge,
                src_weight,
                target_by_edge,
                num_nodes,
            )
            node_update = (node_fwd + node_rev) / 2
        else:
            node_update = node_fwd

        node_new = (node_attr + F.relu(self.bn(node_update))) / 2
        if self.edge_updates:
            edge_input = torch.cat(
                [
                    _aggregate_to_edges(
                        src_weight, src_node, src_edge, node_new, num_edges
                    ),
                    _aggregate_to_edges(
                        dst_weight, dst_node, dst_edge, node_new, num_edges
                    ),
                    edge_attr,
                ],
                dim=-1,
            )
            edge_new = edge_attr + self.edge_mlp(edge_input) / 2
        else:
            edge_new = edge_attr
        return node_new, edge_new


class MultiPNA(nn.Module):
    """Directed, scatter-based PNA backbone for AML seed-edge prediction."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        ego_dim: int,
        port_vocab_size: int,
        port_emb_dim: int,
        dropout: float,
        reverse_mp: bool,
        edge_updates: bool,
        in_degree_histogram: Sequence[int] | Tensor,
        out_degree_histogram: Sequence[int] | Tensor,
        train_inds: Tensor,
        val_inds: Tensor,
        test_inds: Tensor,
        towers: int = 4,
        aggregators: Sequence[str] | str | None = None,
        scalers: Sequence[str] | str | None = None,
        pre_layers: int = 1,
        post_layers: int = 1,
        soft_extrema_temperature: float = 10.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or towers <= 0 or hidden_dim % towers != 0:
            raise ValueError(
                "hidden_dim must be positive and divisible by the number of PNA towers"
            )
        aggregators = _as_names(
            aggregators,
            DEFAULT_AGGREGATORS,
            SUPPORTED_AGGREGATORS,
            "aggregators",
        )
        scalers = _as_names(
            scalers,
            DEFAULT_SCALERS,
            SUPPORTED_SCALERS,
            "scalers",
        )

        self.ego_dim = ego_dim
        self.port_vocab_size = port_vocab_size
        self.use_ports = bool(port_vocab_size) and bool(port_emb_dim)
        self.dropout = dropout
        self.port_emb = (
            nn.Embedding(port_vocab_size, port_emb_dim) if self.use_ports else None
        )
        self.node_emb = nn.Linear(in_dim + ego_dim, hidden_dim)
        encoded_edge_dim = edge_dim + (2 * port_emb_dim if self.use_ports else 0)
        self.edge_emb = nn.Linear(encoded_edge_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                MultiPNALayer(
                    hidden_dim,
                    in_degree_histogram,
                    out_degree_histogram,
                    aggregators,
                    scalers,
                    towers,
                    pre_layers,
                    post_layers,
                    soft_extrema_temperature,
                    reverse_mp,
                    edge_updates,
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = EdgePairPredictionHead(
            hidden_dim, out_dim, dropout, train_inds, val_inds, test_inds
        )

    def forward(
        self,
        batch: GraphDataBatch,
        split: str,
        tgt_edge_inc: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        num_nodes, num_edges = batch.num_nodes, batch.num_edges
        edge_index = batch.edge_index
        src_inc = batch.src_edge_inc
        dst_inc = tgt_edge_inc if tgt_edge_inc is not None else batch.tgt_edge_inc
        src_node, src_edge, src_weight = _sparse_to_messages(src_inc)
        dst_node, dst_edge, dst_weight = _sparse_to_messages(dst_inc)

        edge_attr = batch.edge_attr
        if self.use_ports:
            in_ports, out_ports = compute_batch_ports(
                edge_index, getattr(batch, "timestamps", None), num_nodes
            )
            in_ports = in_ports.clamp(max=self.port_vocab_size - 1)
            out_ports = out_ports.clamp(max=self.port_vocab_size - 1)
            parts = ([edge_attr] if edge_attr is not None else []) + [
                self.port_emb(in_ports),
                self.port_emb(out_ports),
            ]
            edge_attr = torch.cat(parts, dim=-1)
        elif edge_attr is None:
            edge_attr = batch.x.new_zeros(num_edges, 0)

        node_attr = self.node_emb(batch.x)
        edge_attr = self.edge_emb(edge_attr)
        for layer in self.layers:
            node_attr, edge_attr = layer(
                node_attr,
                edge_attr,
                src_node,
                src_edge,
                src_weight,
                dst_node,
                dst_edge,
                dst_weight,
                num_nodes,
                num_edges,
            )
            node_attr = F.dropout(node_attr, p=self.dropout, training=self.training)

        node_pair = torch.cat(
            [node_attr[edge_index[0]], node_attr[edge_index[1]]], dim=-1
        ).relu()
        edge_repr = torch.cat([node_pair, edge_attr], dim=-1)
        return self.classifier(edge_repr, batch, split)


__all__ = [
    "MultiPNA",
    "MultiPNALayer",
    "WeightedPNAPath",
    "compute_pna_degree_histograms",
    "weighted_pna_aggregate",
]
