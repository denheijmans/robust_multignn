import csv
import hashlib
import os
import random
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from robust_multignn.data import GraphData
from robust_multignn.synthetic.simulator import (
    GraphSimulator,
    deg_in,
    deg_out,
    fan_in,
    fan_out,
    to_lists,
)


def set_seed(seed: int):
    """
    Set seeds for Python, NumPy, and PyTorch.
    """
    # NumPy legacy seeding requires 0 <= seed < 2**32.
    seed = int(seed) % (2**32)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def split_seed(seed: int, split: str) -> int:
    """Derive a per-split seed that is stable across processes.

    The builtin ``hash`` of a tuple containing a string is randomized per
    process (``PYTHONHASHSEED``), so it cannot be used to derive seeds.
    """
    digest = hashlib.sha256(f"{seed}:{split}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _canonical_cycle(cyc: Tuple[int, ...]) -> Tuple[int, ...]:
    """Canonicalize by rotation so smallest node id is first (direction preserved)."""
    m = min(cyc)
    mi = cyc.index(m)
    return cyc[mi:] + cyc[:mi]


def Cn_check_with_witness(k: int, max_instances: Optional[int] = None):
    """
    Detects directed cycles of length k, labels all nodes that participate
    in at least one such cycle, and records the exact node tuples (witnesses)
    for each cycle instance it finds.

    Returns a function f(data)->(labels, witnesses)
      labels: [N,1] float 0/1 if node participates in at least one k-cycle
      witnesses: list of k-tuples (each tuple is a directed cycle instance)
    """
    assert 2 <= k <= 6

    def _fn(data: "GraphData"):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)

        out_sets = [set(g_out[i].tolist()) for i in range(num_nodes)]

        labels = torch.zeros((num_nodes, 1), dtype=torch.float32)
        witnesses: List[Tuple[int, ...]] = []
        seen = set()

        # bounded DFS from each start node
        for start in range(num_nodes):
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()

                if len(path) == k:
                    # close cycle: last -> start
                    if start in out_sets[node]:
                        cyc = tuple(path)
                        canon = _canonical_cycle(cyc)
                        if canon not in seen:
                            seen.add(canon)
                            witnesses.append(canon)
                            for u in canon:
                                labels[u, 0] = 1.0
                            if (
                                max_instances is not None
                                and len(witnesses) >= max_instances
                            ):
                                return labels, witnesses
                    continue

                for nxt in out_sets[node]:
                    if nxt in path:
                        continue
                    if nxt == start and len(path) + 1 < k:
                        continue
                    stack.append((nxt, path + [nxt]))

        return labels, witnesses

    return _fn


def SG2_check_with_witness(max_instances: Optional[int] = None):
    """
    Witness-returning version of SG2_check.

    Returns:
      labels: [N,1] 1 if node i is a 'gather' node under SG2 logic
      witnesses: list of 4-tuples (s, j1, j2, i)
        where i is gather, j1/j2 are two distinct predecessors of i,
        and s is a shared predecessor (source) of both j1 and j2 (excluding i).
    """

    def _fn(data: "GraphData"):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)

        labels = torch.zeros((num_nodes, 1), dtype=torch.float32)
        witnesses: List[Tuple[int, int, int, int]] = []

        for i in range(num_nodes):
            preds = g_in[i].unique().tolist()
            if len(preds) < 2:
                continue

            src_to_mids: Dict[int, List[int]] = defaultdict(list)
            for j in preds:
                if j == i:
                    continue
                pj = g_in[j].tolist()
                for s in pj:
                    if s == i:
                        continue
                    src_to_mids[int(s)].append(int(j))

            # pick one witness per gather node i
            found = False
            for s, mids in src_to_mids.items():
                mids_u = list(dict.fromkeys(mids))  # unique preserve order
                if len(mids_u) >= 2:
                    j1, j2 = mids_u[0], mids_u[1]
                    labels[i, 0] = 1.0
                    witnesses.append((int(s), int(j1), int(j2), int(i)))
                    found = True
                    if max_instances is not None and len(witnesses) >= max_instances:
                        return labels, witnesses
                    break
            if found:
                continue

        return labels, witnesses

    return _fn


def BP2_check_with_witness(max_instances: Optional[int] = None):
    """
    Witness-returning version of BP2_check.

    Returns:
      labels: [N,1] 1 if node i satisfies BP2 logic
      witnesses: list of 4-tuples (l1, l2, r, i)
        where i is the "right-side" gather-like node,
        l1,l2 are two distinct predecessors of i,
        and r is a node that both l1 and l2 point to (excluding i).
    """

    def _fn(data: "GraphData"):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)

        labels = torch.zeros((num_nodes, 1), dtype=torch.float32)
        witnesses: List[Tuple[int, int, int, int]] = []

        for i in range(num_nodes):
            lefts = g_in[i].unique().tolist()
            if len(lefts) < 2:
                continue

            right_to_lefts: Dict[int, List[int]] = defaultdict(list)
            for left in lefts:
                outs = g_out[int(left)].tolist()
                for r in outs:
                    if int(r) == i:
                        continue
                    right_to_lefts[int(r)].append(int(left))

            # one witness per i
            found = False
            for r, ls in right_to_lefts.items():
                ls_u = list(dict.fromkeys(ls))
                if len(ls_u) >= 2:
                    l1, l2 = ls_u[0], ls_u[1]
                    labels[i, 0] = 1.0
                    witnesses.append((int(l1), int(l2), int(r), int(i)))
                    found = True
                    if max_instances is not None and len(witnesses) >= max_instances:
                        return labels, witnesses
                    break

            if found:
                continue

        return labels, witnesses

    return _fn


def define_subtasks_thresholds_and_witness_builders(
    cycle_max_instances_per_k: Optional[int] = None,
    sg_max_instances: Optional[int] = None,
    bp_max_instances: Optional[int] = None,
):
    """
    Same task ordering with define_subtasks_and_thresholds() method of Beni,
    but for motif tasks we use witness-returning builders.
    For degree/fan tasks we reuse the existing functions and thresholds.
    """
    # Regular label-only tasks
    label_only_funcs = [deg_in, deg_out, fan_in, fan_out]
    label_only_thresholds = [3, 3, 3, 3]
    label_only_names = ["deg_in", "deg_out", "fan_in", "fan_out"]

    # Witness-returning motif tasks (cycles 2..6, SG2, BP2)
    motif_builders = [
        ("cycle2", Cn_check_with_witness(2, max_instances=cycle_max_instances_per_k)),
        ("cycle3", Cn_check_with_witness(3, max_instances=cycle_max_instances_per_k)),
        ("cycle4", Cn_check_with_witness(4, max_instances=cycle_max_instances_per_k)),
        ("cycle5", Cn_check_with_witness(5, max_instances=cycle_max_instances_per_k)),
        ("cycle6", Cn_check_with_witness(6, max_instances=cycle_max_instances_per_k)),
        ("scatter_gather", SG2_check_with_witness(max_instances=sg_max_instances)),
        ("biclique", BP2_check_with_witness(max_instances=bp_max_instances)),
    ]

    names = label_only_names + [n for (n, _) in motif_builders]
    thresholds = label_only_thresholds + [None] * len(motif_builders)
    return label_only_funcs, label_only_thresholds, motif_builders, names, thresholds


def set_y_with_labels_and_witnesses(
    data: "GraphData",
    label_only_funcs,
    label_only_thresholds,
    motif_builders,
):
    """
    Produces:
      data.y: [N, 11] float 0/1
      witnesses: dict with keys:
        "cycle2".."cycle6" -> List[Tuple[int,...]]
        "scatter_gather" -> List[Tuple[int,int,int,int]]
        "biclique" -> List[Tuple[int,int,int,int]]
    """
    ys = []
    witnesses: Dict[str, List[Tuple[int, ...]]] = {}

    # label-only tasks
    for f, th in zip(label_only_funcs, label_only_thresholds):
        y = f(data, th)
        ys.append(y.float())

    # motif tasks (label + witnesses)
    for name, builder in motif_builders:
        y, w = builder(data)
        ys.append(y.float())
        witnesses[name] = w

    data.y = torch.cat(ys, dim=1)
    return data, witnesses


def generate_synthetic_graph(
    seed: int,
) -> GraphData:
    """Generates a synthetic graph dataset with node labels based on various structural properties and motifs. The graph is generated using a simulator, and then labeled according to degree/fan thresholds and the presence of specific subgraph patterns (cycles, scatter-gather, biclique). Witnesses for the motif-based labels are also collected.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility of graph generation and labeling.

    Returns
    -------
    Data
        A synthetic graph dataset with node features, edge indices, and labels for various structural properties and motifs.
    """
    set_seed(seed)

    # Generate a synthetic graph
    graph: GraphData = GraphSimulator(
        num_nodes=8192,
        avg_degree=6,
        num_edges=None,
        network_type="type1",
        readout="node",
        node_feats=False,
        bidirectional=False,
        delta=11.1,
        num_graphs=1,
        generator="chordal",
    ).generate_pytorch_graph()
    graph.edge_attr = None

    # Define various subtasks
    (
        label_only_funcs,
        label_only_thresholds,
        motif_builders,
        task_names,
        thresholds,
    ) = define_subtasks_thresholds_and_witness_builders()

    # Label nodes based on subtasks
    graph, witnesses = set_y_with_labels_and_witnesses(
        graph,
        label_only_funcs,
        label_only_thresholds,
        motif_builders,
    )

    return graph


MULTIGRAPH_SPLITS = ("train", "val", "test")
MULTIGRAPH_CYCLE_TASKS = ("cycle2", "cycle3", "cycle4", "cycle5", "cycle6")

BALANCE_TYPES = (
    "strong_decreasing",
    "mild_decreasing",
    "flat_balanced",
    "mild_increasing",
    "strong_increasing",
)
CYCLE_BALANCE_DEFAULTS = {
    "n_pool": [3800, 3900, 4000, 4100, 4200],
    "d_pool": [4],
    "r_pool": [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        9.0,
        10.0,
    ],
    "generators": ["chordal"],
    "tasks": list(MULTIGRAPH_CYCLE_TASKS),
    "base_seed": 0,
    "selection_split": "train",
    "selection_slope_thresh": 0.015,
    "selection_strong_slope_thresh": 0.03,
}


def derive_seed(base_seed: int, namespace: str) -> int:
    data = f"{base_seed}:{namespace}".encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def dataset_id(n: int, d: int, r: float, generator: str) -> str:
    return f"data_{int(n)}_{int(d)}_{r}_{generator}"


def _sampled_param_grid(
    n_pool: Sequence[int],
    d_pool: Sequence[int],
    r_pool: Sequence[float],
    generators: Sequence[str],
    base_seed: int,
) -> list[tuple[int, int, float, str]]:
    all_combos = list(product(n_pool, d_pool, r_pool, generators))
    num_graphs = len(n_pool) * len(d_pool) * len(r_pool) * len(generators)

    if num_graphs > len(all_combos):
        raise ValueError(
            f"num_graphs={num_graphs} > total combos={len(all_combos)}. "
            "Reduce num_graphs or expand pools."
        )

    rng = np.random.default_rng(base_seed)
    chosen_idx = rng.choice(len(all_combos), size=num_graphs, replace=False)
    return [all_combos[i] for i in chosen_idx]


def generate_multigraph_graph(
    seed: int,
    n_pool: Sequence[int],
    d_pool: Sequence[int],
    r_pool: Sequence[float],
    generators: Sequence[str],
    tasks: Sequence[str] = MULTIGRAPH_CYCLE_TASKS,
    base_seed: int = 0,
    selected_graph_id: Optional[int] = None,
    selected_dataset_id: Optional[str] = None,
) -> GraphData:
    sampled_params = _sampled_param_grid(
        n_pool=n_pool,
        d_pool=d_pool,
        r_pool=r_pool,
        generators=generators,
        base_seed=base_seed,
    )

    if selected_graph_id is not None:
        combo_idx = int(selected_graph_id) % len(sampled_params)
    elif selected_dataset_id is not None:
        combo_idx = 0
        for idx, (n, d, r, generator) in enumerate(sampled_params):
            if dataset_id(n, d, r, generator) == str(selected_dataset_id):
                combo_idx = idx
                break
    else:
        combo_idx = int(seed) % len(sampled_params)

    n, d, r, generator = sampled_params[combo_idx]

    graph = make_multigraph_sim(
        n=n,
        d=d,
        r=r,
        generator=generator,
        seed=seed,
    ).generate_pytorch_graph()
    graph.edge_attr = None
    graph, _ = set_cycle_y_and_get_motifs(graph, tasks)
    return graph


def generate_named_multigraph_splits(
    seed: int,
    selected_family: str = "strong_decreasing",
    selected_family_index: int = 0,
) -> tuple[dict[str, GraphData], list[str], str]:
    cfg = CYCLE_BALANCE_DEFAULTS
    selected_graph_id = select_graph_id_by_family(
        family=selected_family,
        n_pool=cfg["n_pool"],
        d_pool=cfg["d_pool"],
        r_pool=cfg["r_pool"],
        generators=cfg["generators"],
        tasks=cfg["tasks"],
        base_seed=cfg["base_seed"],
        split=cfg["selection_split"],
        slope_thresh=cfg["selection_slope_thresh"],
        strong_slope_thresh=cfg["selection_strong_slope_thresh"],
        family_index=selected_family_index,
    )

    splits = {
        split_name: generate_multigraph_graph(
            seed=split_seed(seed, split_name),
            n_pool=cfg["n_pool"],
            d_pool=cfg["d_pool"],
            r_pool=cfg["r_pool"],
            generators=cfg["generators"],
            tasks=cfg["tasks"],
            base_seed=cfg["base_seed"],
            selected_graph_id=selected_graph_id,
            selected_dataset_id=None,
        )
        for split_name in MULTIGRAPH_SPLITS
    }

    return splits, list(cfg["tasks"]), selected_family


def _profile_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    m, _b = np.polyfit(x, y, deg=1)
    return float(m)


def _family_from_profile(
    profile: np.ndarray,
    slope_thresh: float,
    strong_slope_thresh: float,
) -> str:
    slope = _profile_slope(profile)

    if slope <= slope_thresh and slope >= -slope_thresh:
        return "flat_balanced"
    if slope > slope_thresh:
        if slope <= strong_slope_thresh:
            return "mild_increasing"
        return "strong_increasing"
    if slope < -slope_thresh:
        if slope > -strong_slope_thresh:
            return "mild_decreasing"
        return "strong_decreasing"
    return "error"


def select_graph_id_by_family(
    family: str,
    n_pool: Sequence[int],
    d_pool: Sequence[int],
    r_pool: Sequence[float],
    generators: Sequence[str],
    tasks: Sequence[str] = MULTIGRAPH_CYCLE_TASKS,
    base_seed: int = 0,
    split: str = "train",
    slope_thresh: float = 0.015,
    strong_slope_thresh: float = 0.03,
    family_index: int = 0,
) -> int:
    sampled_params = _sampled_param_grid(
        n_pool=n_pool,
        d_pool=d_pool,
        r_pool=r_pool,
        generators=generators,
        base_seed=base_seed,
    )

    candidates = []
    for graph_id, (n, d, r, generator) in enumerate(sampled_params):
        split_seed = derive_seed(
            base_seed,
            f"{split}_g{graph_id}_{n}_{d}_{r}_{generator}",
        )
        graph = make_multigraph_sim(
            n=n,
            d=d,
            r=r,
            generator=generator,
            seed=split_seed,
        ).generate_pytorch_graph()
        graph.edge_attr = None
        graph, _ = set_cycle_y_and_get_motifs(graph, tasks)

        profile = graph.y.detach().cpu().numpy().mean(axis=0)
        graph_family = _family_from_profile(
            profile,
            slope_thresh=slope_thresh,
            strong_slope_thresh=strong_slope_thresh,
        )
        if graph_family == family:
            candidates.append(graph_id)

    if not candidates:
        raise ValueError(f"No graph found for selected_family='{family}'")

    return candidates[int(family_index) % len(candidates)]


def _prefix_keys(values: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in values.items()}


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if len(values) else 0.0


def _cycle_motif_builders(tasks: Sequence[str]) -> dict:
    builders = {}
    for task_name in tasks:
        k = int(task_name.replace("cycle", ""))
        builders[task_name] = Cn_check_with_witness(k)
    return builders


def set_cycle_y_and_get_motifs(
    data: GraphData, tasks: Sequence[str] = MULTIGRAPH_CYCLE_TASKS
) -> tuple[GraphData, dict]:
    num_nodes = int(data.num_nodes)
    y = torch.zeros((num_nodes, len(tasks)), dtype=torch.float32)
    motifs_by_task = {}
    builders = _cycle_motif_builders(tasks)

    for col, task_name in enumerate(tasks):
        labels, motifs = builders[task_name](data)
        y[:, col] = labels[:, 0]
        motifs_by_task[task_name] = motifs

    data.y = y
    data.num_classes = y.shape[1]
    return data, motifs_by_task


def make_multigraph_sim(
    n: int,
    d: int,
    r: float,
    generator: str,
    seed: int,
) -> GraphSimulator:
    # GraphSimulator uses global numpy/torch RNG state; seed it per split.
    set_seed(seed)
    return GraphSimulator(
        num_nodes=n,
        avg_degree=d,
        num_edges=None,
        network_type="type1",
        readout="node",
        node_feats=False,
        bidirectional=False,
        delta=r,
        num_graphs=1,
        generator=generator,
    )


def graph_basic_stats(data: GraphData) -> dict:
    edge_index = data.edge_index.detach().cpu()
    n = int(data.num_nodes)
    e_raw = int(edge_index.size(1))

    src = edge_index[0].numpy().astype(np.int64)
    dst = edge_index[1].numpy().astype(np.int64)

    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    out_deg_raw = np.bincount(src, minlength=n).astype(np.int64)
    in_deg_raw = np.bincount(dst, minlength=n).astype(np.int64)

    unique_edges = {(int(u), int(v)) for u, v in zip(src.tolist(), dst.tolist())}
    e_unique = len(unique_edges)
    possible_edges = n * (n - 1)

    if e_unique > 0:
        unique_src = np.fromiter((u for u, _ in unique_edges), dtype=np.int64)
        unique_dst = np.fromiter((v for _, v in unique_edges), dtype=np.int64)
        out_deg_unique = np.bincount(unique_src, minlength=n).astype(np.int64)
        in_deg_unique = np.bincount(unique_dst, minlength=n).astype(np.int64)
    else:
        out_deg_unique = np.zeros(n, dtype=np.int64)
        in_deg_unique = np.zeros(n, dtype=np.int64)

    reciprocal_unique = sum((v, u) in unique_edges for (u, v) in unique_edges)
    reciprocity_unique = (reciprocal_unique / e_unique) if e_unique > 0 else 0.0
    duplicate_edge_fraction = (1.0 - (e_unique / e_raw)) if e_raw > 0 else 0.0

    return {
        "num_nodes": n,
        "num_edges_raw": e_raw,
        "num_edges_unique": e_unique,
        "density_raw": (e_raw / possible_edges) if possible_edges > 0 else 0.0,
        "density_unique": (e_unique / possible_edges) if possible_edges > 0 else 0.0,
        "duplicate_edge_fraction": duplicate_edge_fraction,
        "reciprocity_unique": reciprocity_unique,
        "mean_in_degree_raw": _safe_mean(in_deg_raw),
        "mean_out_degree_raw": _safe_mean(out_deg_raw),
        "std_in_degree_raw": _safe_std(in_deg_raw),
        "std_out_degree_raw": _safe_std(out_deg_raw),
        "max_in_degree_raw": int(in_deg_raw.max()) if len(in_deg_raw) else 0,
        "max_out_degree_raw": int(out_deg_raw.max()) if len(out_deg_raw) else 0,
        "mean_in_degree_unique": _safe_mean(in_deg_unique),
        "mean_out_degree_unique": _safe_mean(out_deg_unique),
        "std_in_degree_unique": _safe_std(in_deg_unique),
        "std_out_degree_unique": _safe_std(out_deg_unique),
        "max_in_degree_unique": int(in_deg_unique.max()) if len(in_deg_unique) else 0,
        "max_out_degree_unique": int(out_deg_unique.max())
        if len(out_deg_unique)
        else 0,
    }


def split_task_stats(
    data: GraphData,
    split_motifs: dict,
    tasks: Sequence[str] = MULTIGRAPH_CYCLE_TASKS,
) -> dict:
    y = data.y.detach().cpu().numpy()
    n = int(data.num_nodes)
    out = {}

    for col, task_name in enumerate(tasks):
        motif_count = int(len(split_motifs[task_name]))
        pos_nodes = int(y[:, col].sum())
        pos_rate = (pos_nodes / n) if n > 0 else 0.0

        out[f"{task_name}_motif_count"] = motif_count
        out[f"{task_name}_pos_nodes"] = pos_nodes
        out[f"{task_name}_pos_rate"] = pos_rate
        out[f"{task_name}_all_zero"] = int(pos_nodes == 0)

    return out


def write_motif_counts_csv(path: str, split_motifs: dict, tasks: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split"] + list(tasks))
        for split_name in MULTIGRAPH_SPLITS:
            motifs = split_motifs[split_name]
            writer.writerow([split_name] + [len(motifs[t]) for t in tasks])


def write_label_support_csv(
    path: str,
    split_task_summary: dict,
    tasks: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "task", "motif_count", "pos_nodes", "pos_rate"])
        for split_name in MULTIGRAPH_SPLITS:
            for task_name in tasks:
                writer.writerow(
                    [
                        split_name,
                        task_name,
                        split_task_summary[split_name][f"{task_name}_motif_count"],
                        split_task_summary[split_name][f"{task_name}_pos_nodes"],
                        split_task_summary[split_name][f"{task_name}_pos_rate"],
                    ]
                )


def add_cross_split_summary(
    row: dict,
    tasks: Sequence[str] = MULTIGRAPH_CYCLE_TASKS,
) -> dict:
    for task_name in tasks:
        pos_rates = [
            row[f"{split_name}_{task_name}_pos_rate"]
            for split_name in MULTIGRAPH_SPLITS
        ]
        pos_nodes = [
            row[f"{split_name}_{task_name}_pos_nodes"]
            for split_name in MULTIGRAPH_SPLITS
        ]
        motif_counts = [
            row[f"{split_name}_{task_name}_motif_count"]
            for split_name in MULTIGRAPH_SPLITS
        ]

        pos_nodes_mean = float(np.mean(pos_nodes))
        row[f"{task_name}_pos_nodes_mean"] = pos_nodes_mean
        row[f"{task_name}_pos_nodes_std"] = float(np.std(pos_nodes))
        row[f"{task_name}_pos_nodes_max_diff_from_mean"] = float(
            max(
                np.max(pos_nodes) - pos_nodes_mean,
                pos_nodes_mean - np.min(pos_nodes),
            )
        )

        pos_rate_mean = float(np.mean(pos_rates))
        row[f"{task_name}_pos_rate_mean"] = pos_rate_mean
        row[f"{task_name}_pos_rate_std"] = float(np.std(pos_rates))
        row[f"{task_name}_pos_rate_max_diff_from_mean"] = float(
            max(
                np.max(pos_rates) - pos_rate_mean,
                pos_rate_mean - np.min(pos_rates),
            )
        )

        row[f"{task_name}_all_splits_have_pos"] = int(min(pos_nodes) > 0)
        row[f"{task_name}_motif_count_mean"] = float(np.mean(motif_counts))
        row[f"{task_name}_motif_count_std"] = float(np.std(motif_counts))

    return row


def graph_outputs_exist(out_dir_pt: str) -> bool:
    required_files = [
        os.path.join(out_dir_pt, "motif_counts.csv"),
        os.path.join(out_dir_pt, "label_support.csv"),
        *[os.path.join(out_dir_pt, f"{split}.pt") for split in MULTIGRAPH_SPLITS],
    ]
    return all(
        os.path.isfile(path) and os.path.getsize(path) > 0 for path in required_files
    )


def load_split_task_summary_from_csv(path: str, tasks: Sequence[str]) -> dict:
    frame = pd.read_csv(path)

    out = {}
    for split_name in MULTIGRAPH_SPLITS:
        sub = frame[frame["split"] == split_name].set_index("task")

        split_dict = {}
        for task_name in tasks:
            motif_count = int(sub.loc[task_name, "motif_count"])
            pos_nodes = int(sub.loc[task_name, "pos_nodes"])
            pos_rate = float(sub.loc[task_name, "pos_rate"])

            split_dict[f"{task_name}_motif_count"] = motif_count
            split_dict[f"{task_name}_pos_nodes"] = pos_nodes
            split_dict[f"{task_name}_pos_rate"] = pos_rate
            split_dict[f"{task_name}_all_zero"] = int(pos_nodes == 0)

        out[split_name] = split_dict

    return out


def load_split_basic_from_pt(out_dir_pt: str) -> dict:
    out = {}
    for split_name in MULTIGRAPH_SPLITS:
        graph = torch.load(
            os.path.join(out_dir_pt, f"{split_name}.pt"),
            weights_only=False,
        )
        out[split_name] = graph_basic_stats(graph)
    return out


def build_registry_row(
    graph_id: int,
    n: int,
    d: int,
    r: float,
    generator: str,
    did: str,
    out_dir_pt: str,
    split_basic: dict,
    split_task_summary: dict,
    base_seed: int,
    tasks: Sequence[str],
) -> dict:
    split_seeds = {
        split_name: derive_seed(
            base_seed,
            f"{split_name}_g{graph_id}_{n}_{d}_{r}_{generator}",
        )
        for split_name in MULTIGRAPH_SPLITS
    }

    row = {
        "graph_id": graph_id,
        "dataset_id": did,
        "data_dir": out_dir_pt,
        "n": int(n),
        "d": int(d),
        "r": float(r),
        "type": generator,
        "seed_train": split_seeds["train"],
        "seed_val": split_seeds["val"],
        "seed_test": split_seeds["test"],
    }

    for split_name in MULTIGRAPH_SPLITS:
        row.update(_prefix_keys(split_basic[split_name], f"{split_name}_"))
        row.update(_prefix_keys(split_task_summary[split_name], f"{split_name}_"))

    row = add_cross_split_summary(row, tasks)
    return row


def generate_multigraph_dataset_registry(
    data_root: str,
    graph_param_csv: str,
    n_pool: Sequence[int],
    d_pool: Sequence[int],
    r_pool: Sequence[float],
    generators: Sequence[str],
    tasks: Sequence[str] = MULTIGRAPH_CYCLE_TASKS,
    base_seed: int = 0,
    run_balanced_selection: bool = True,
    selection_clients_per_family: int = 4,
    selection_pmin_low: float = 0.03,
    selection_split: str = "train",
    selection_slope_thresh: float = 0.015,
    selection_strong_slope_thresh: float = 0.03,
    selection_family_order: Sequence[str] = (
        "strong_decreasing",
        "mild_decreasing",
        "flat_balanced",
        "mild_increasing",
        "strong_increasing",
    ),
) -> pd.DataFrame:
    set_seed(base_seed)
    rng = np.random.default_rng(base_seed)

    all_combos = list(product(n_pool, d_pool, r_pool, generators))
    num_graphs = len(n_pool) * len(d_pool) * len(r_pool) * len(generators)

    if num_graphs > len(all_combos):
        raise ValueError(
            f"num_graphs={num_graphs} > total combos={len(all_combos)}. "
            "Reduce num_graphs or expand pools."
        )

    chosen_idx = rng.choice(len(all_combos), size=num_graphs, replace=False)
    sampled_params = [all_combos[i] for i in chosen_idx]

    rows = []

    for graph_id, (n, d, r, generator) in enumerate(sampled_params):
        did = dataset_id(n, d, r, generator)
        out_dir_pt = os.path.join(data_root, did)

        if graph_outputs_exist(out_dir_pt):
            split_basic = load_split_basic_from_pt(out_dir_pt)
            split_task_summary = load_split_task_summary_from_csv(
                os.path.join(out_dir_pt, "label_support.csv"),
                tasks,
            )
            row = build_registry_row(
                graph_id=graph_id,
                n=n,
                d=d,
                r=r,
                generator=generator,
                did=did,
                out_dir_pt=out_dir_pt,
                split_basic=split_basic,
                split_task_summary=split_task_summary,
                base_seed=base_seed,
                tasks=tasks,
            )
            rows.append(row)
            continue

        split_graphs = {}
        split_motifs = {}
        split_basic = {}
        split_task_summary = {}

        split_seeds = {
            split_name: derive_seed(
                base_seed,
                f"{split_name}_g{graph_id}_{n}_{d}_{r}_{generator}",
            )
            for split_name in MULTIGRAPH_SPLITS
        }

        for split_name in MULTIGRAPH_SPLITS:
            graph = make_multigraph_sim(
                n=n,
                d=d,
                r=r,
                generator=generator,
                seed=split_seeds[split_name],
            ).generate_pytorch_graph()
            graph.edge_attr = None

            graph, motifs = set_cycle_y_and_get_motifs(graph, tasks)
            split_graphs[split_name] = graph
            split_motifs[split_name] = motifs
            split_basic[split_name] = graph_basic_stats(graph)
            split_task_summary[split_name] = split_task_stats(graph, motifs, tasks)

        os.makedirs(out_dir_pt, exist_ok=True)

        for split_name in MULTIGRAPH_SPLITS:
            torch.save(
                split_graphs[split_name], os.path.join(out_dir_pt, f"{split_name}.pt")
            )

        write_motif_counts_csv(
            os.path.join(out_dir_pt, "motif_counts.csv"),
            split_motifs,
            tasks,
        )
        write_label_support_csv(
            os.path.join(out_dir_pt, "label_support.csv"),
            split_task_summary,
            tasks,
        )

        row = build_registry_row(
            graph_id=graph_id,
            n=n,
            d=d,
            r=r,
            generator=generator,
            did=did,
            out_dir_pt=out_dir_pt,
            split_basic=split_basic,
            split_task_summary=split_task_summary,
            base_seed=base_seed,
            tasks=tasks,
        )
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["graph_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(graph_param_csv), exist_ok=True)
    frame.to_csv(graph_param_csv, index=False)

    if run_balanced_selection:
        try:
            from robust_multignn.synthetic.cluster_selection import run_selection

            run_selection(
                graph_param_csv=graph_param_csv,
                out_dir=os.path.dirname(graph_param_csv),
                split=selection_split,
                clients_per_family=selection_clients_per_family,
                pmin_low=selection_pmin_low,
                slope_thresh=selection_slope_thresh,
                strong_slope_thresh=selection_strong_slope_thresh,
                tasks=tasks,
                family_order=selection_family_order,
            )
        except Exception as exc:
            print(f"WARNING: balanced subset selection skipped ({exc})")

    return frame
