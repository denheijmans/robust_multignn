import json
import os
from typing import List, Sequence

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def rate_col(split: str, task: str) -> str:
    return f"{split}_{task}_pos_rate"


def dataset_id_from_row(row: pd.Series) -> str:
    if "dataset_id" in row and pd.notna(row["dataset_id"]):
        return str(row["dataset_id"])
    if "data_dir" in row and pd.notna(row["data_dir"]):
        return os.path.basename(str(row["data_dir"]).rstrip("/"))
    graph_type = str(row["type"]) if "type" in row else "graph"
    return f"data_{int(row['n'])}_{int(row['d'])}_{row['r']}_{graph_type}"


def profile_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    m, _b = np.polyfit(x, y, deg=1)
    return float(m)


def rough_shape_family(
    values: np.ndarray,
    slope_thresh: float,
    strong_slope_thresh: float,
) -> str:
    values = np.asarray(values, dtype=np.float64)
    slope = profile_slope(values)

    if slope <= slope_thresh and slope >= -slope_thresh:
        return "flat_balanced"

    if slope > slope_thresh:
        if slope <= strong_slope_thresh:
            return "mild_increasing"
        if slope > strong_slope_thresh:
            return "strong_increasing"

    if slope < -slope_thresh:
        if slope > -strong_slope_thresh:
            return "mild_decreasing"
        if slope <= -strong_slope_thresh:
            return "strong_decreasing"

    return "error"


def build_profile_table(
    frame: pd.DataFrame,
    tasks: Sequence[str],
    split: str,
    slope_thresh: float,
    strong_slope_thresh: float,
) -> pd.DataFrame:
    out = frame.copy()
    out["dataset_id"] = out.apply(dataset_id_from_row, axis=1)

    p_cols: List[str] = []
    for task in tasks:
        pcol = f"p_{task}"
        out[pcol] = out[rate_col(split, task)].astype(np.float64)
        p_cols.append(pcol)

    p_matrix = out[p_cols].to_numpy(dtype=np.float64)

    out["p_sum"] = p_matrix.sum(axis=1)
    out["p_mean"] = p_matrix.mean(axis=1)
    out["p_min"] = p_matrix.min(axis=1)
    out["p_max"] = p_matrix.max(axis=1)
    out["p_spread"] = out["p_max"] - out["p_min"]
    out["slope"] = [profile_slope(p) for p in p_matrix]
    out["dominant_task"] = [tasks[int(np.argmax(p))] for p in p_matrix]
    out["shape_family"] = [
        rough_shape_family(p, slope_thresh, strong_slope_thresh) for p in p_matrix
    ]

    return out


def print_pool_summary(
    frame: pd.DataFrame,
    title: str,
    family_order: Sequence[str],
) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"N clients: {len(frame)}")

    counts = (
        frame["shape_family"]
        .value_counts()
        .reindex(list(family_order) + ["error"], fill_value=0)
    )
    pct = (counts / max(len(frame), 1) * 100.0).round(2)
    print("\nFAMILY COUNTS")
    print(pd.DataFrame({"count": counts, "percent": pct}).to_string())


def print_selected_clients_per_family(
    selected_frame: pd.DataFrame,
    tasks: Sequence[str],
    family_order: Sequence[str],
) -> None:
    show_cols = [
        "graph_id",
        "dataset_id",
        "d",
        "r",
        "shape_family",
        "slope",
        "p_mean",
        "p_min",
        "p_max",
        "p_spread",
        "dist_to_family_center",
        "dominant_task",
    ] + [f"p_{task}" for task in tasks]

    print("\n" + "=" * 80)
    print("SELECTED CLIENTS PER FAMILY")
    print("=" * 80)

    for family in family_order:
        sub = selected_frame[selected_frame["shape_family"] == family].sort_values(
            "dist_to_family_center"
        )
        print(f"\nFAMILY: {family}  (n={len(sub)})")
        print("-" * 80)
        if len(sub) == 0:
            print("No selected clients")
            continue
        print(sub[show_cols].to_string(index=False))


def filter_candidate_pool(profile_frame: pd.DataFrame, pmin_low: float) -> pd.DataFrame:
    out = profile_frame.copy()

    print("\n" + "=" * 80)
    print("FILTERING CANDIDATE POOL")
    print("=" * 80)
    print(f"start: {len(out)} clients")

    out = out[out["p_min"] >= pmin_low].copy()
    print(f"after p_min >= {pmin_low:.2f}: {len(out)} clients")

    return out


def select_representatives_per_family(
    candidate_frame: pd.DataFrame,
    per_family: int,
    family_order: Sequence[str],
) -> pd.DataFrame:
    picks = []

    for family in family_order:
        sub = candidate_frame[candidate_frame["shape_family"] == family].copy()
        if len(sub) == 0:
            print(f"WARNING: family {family} has 0 candidates")
            continue

        center_slope = sub["slope"].median()
        center_spread = sub["p_spread"].median()
        center_pmean = sub["p_mean"].median()

        slope_scale = sub["slope"].std(ddof=0) + 1e-12
        spread_scale = sub["p_spread"].std(ddof=0) + 1e-12
        pmean_scale = sub["p_mean"].std(ddof=0) + 1e-12

        sub["dist_to_family_center"] = np.sqrt(
            ((sub["slope"] - center_slope) / slope_scale) ** 2
            + ((sub["p_spread"] - center_spread) / spread_scale) ** 2
            + ((sub["p_mean"] - center_pmean) / pmean_scale) ** 2
        )

        chosen = sub.sort_values("dist_to_family_center").head(per_family)
        picks.append(chosen)

        if len(chosen) < per_family:
            print(
                f"WARNING: family {family} has only {len(chosen)} selected clients "
                f"(< requested {per_family})"
            )

    if len(picks) == 0:
        return candidate_frame.iloc[[]].copy()

    return pd.concat(picks, ignore_index=False).reset_index(drop=True)


def build_combined_manifest_from_selected(
    selected_frame: pd.DataFrame,
    family_order: Sequence[str],
) -> pd.DataFrame:
    if len(selected_frame) == 0:
        return pd.DataFrame()

    sub = selected_frame.copy().sort_values(
        ["shape_family", "dist_to_family_center", "graph_id"]
    )

    graph_ids = [int(x) for x in sub["graph_id"].tolist()]
    dataset_ids = [str(x) for x in sub["dataset_id"].tolist()]
    subset_clients = "|".join(str(x) for x in graph_ids)

    subset_id = f"combined_five_family_benchmark_{subset_clients.replace('|', '_')}"

    family_to_graph_ids = {}
    family_to_dataset_ids = {}
    graph_to_family = {}
    dataset_to_family = {}
    membership_records = []

    for family in family_order:
        fam_sub = sub[sub["shape_family"] == family].copy()
        fam_graph_ids = [int(x) for x in fam_sub["graph_id"].tolist()]
        fam_dataset_ids = [str(x) for x in fam_sub["dataset_id"].tolist()]

        family_to_graph_ids[family] = fam_graph_ids
        family_to_dataset_ids[family] = fam_dataset_ids

        for _, row in fam_sub.iterrows():
            gid = int(row["graph_id"])
            did = str(row["dataset_id"])
            graph_to_family[str(gid)] = family
            dataset_to_family[did] = family
            membership_records.append(
                {
                    "graph_id": gid,
                    "dataset_id": did,
                    "family": family,
                }
            )

    family_counts = (
        sub["shape_family"]
        .value_counts()
        .reindex(list(family_order), fill_value=0)
        .to_dict()
    )

    row = {
        "family": "combined_five_family_benchmark",
        "subset_id": subset_id,
        "subset_size": int(len(graph_ids)),
        "subset_clients": subset_clients,
        "graph_ids_json": json.dumps(graph_ids),
        "dataset_ids_json": json.dumps(dataset_ids),
        "family_order_json": json.dumps(list(family_order)),
        "family_to_graph_ids_json": json.dumps(family_to_graph_ids),
        "family_to_dataset_ids_json": json.dumps(family_to_dataset_ids),
        "graph_to_family_json": json.dumps(graph_to_family),
        "dataset_to_family_json": json.dumps(dataset_to_family),
        "membership_json": json.dumps(membership_records),
        "family_counts_json": json.dumps(family_counts),
        "mean_slope": float(sub["slope"].mean()),
        "std_slope": float(sub["slope"].std(ddof=0)),
        "mean_p_mean": float(sub["p_mean"].mean()),
        "std_p_mean": float(sub["p_mean"].std(ddof=0)),
        "mean_p_min": float(sub["p_min"].mean()),
        "mean_p_spread": float(sub["p_spread"].mean()),
        "max_dist_to_family_center": float(sub["dist_to_family_center"].max()),
    }

    return pd.DataFrame([row])


def run_selection(
    graph_param_csv: str,
    out_dir: str,
    split: str,
    clients_per_family: int,
    pmin_low: float,
    slope_thresh: float,
    strong_slope_thresh: float,
    tasks: Sequence[str],
    family_order: Sequence[str],
) -> pd.DataFrame:
    ensure_dir(out_dir)

    graph_frame = pd.read_csv(graph_param_csv)
    profile_frame = build_profile_table(
        graph_frame,
        tasks=tasks,
        split=split,
        slope_thresh=slope_thresh,
        strong_slope_thresh=strong_slope_thresh,
    )
    print_pool_summary(
        profile_frame,
        title="FULL PROFILE POOL",
        family_order=family_order,
    )

    candidate_frame = filter_candidate_pool(profile_frame, pmin_low=pmin_low)
    print_pool_summary(
        candidate_frame,
        title="FILTERED CANDIDATE POOL",
        family_order=family_order,
    )

    selected_frame = select_representatives_per_family(
        candidate_frame,
        per_family=clients_per_family,
        family_order=family_order,
    ).copy()
    selected_frame["family"] = selected_frame["shape_family"]
    print_selected_clients_per_family(
        selected_frame,
        tasks=tasks,
        family_order=family_order,
    )

    manifest_frame = build_combined_manifest_from_selected(
        selected_frame,
        family_order=family_order,
    )
    manifest_csv = os.path.join(out_dir, "selected_subset.csv")
    manifest_frame.to_csv(manifest_csv, index=False)
    print(f"Balanced subset manifest written -> {manifest_csv}")

    return manifest_frame


def main() -> None:
    raise RuntimeError(
        "run_selection is config-driven and must be called with explicit arguments"
    )


if __name__ == "__main__":
    main()
