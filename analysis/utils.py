import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from wandb.apis.public import Run

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="fetch runs from W&B even when a cached export already exists",
    )
    return parser.parse_args()


def _query_key(query: dict[str, Any]) -> str:
    payload = json.dumps(query, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir: Path, query: dict[str, Any]) -> Path:
    return cache_dir / f"runs_{_query_key(query)}.csv"


def _run_to_record(run: Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "name": run.name,
        "created_at": run.created_at,
        "tags": list(run.tags or []),
        "summary": run.summary._json_dict,
        "config": {
            key: value for key, value in run.config.items() if not key.startswith("_")
        },
    }


def load_cached_runs(cache_file: Path) -> pd.DataFrame | None:
    if not cache_file.exists():
        return None

    try:
        frame = pd.read_csv(cache_file)
    except pd.errors.EmptyDataError:
        # Cache file exists but is empty or corrupt (e.g. interrupted write);
        # treat as a cache miss so the caller re-fetches from W&B.
        cache_file.unlink(missing_ok=True)
        return None

    if "summary" in frame.columns:
        frame["summary"] = frame["summary"].map(json.loads)
    if "config" in frame.columns:
        frame["config"] = frame["config"].map(json.loads)
    if "tags" in frame.columns:
        frame["tags"] = frame["tags"].map(json.loads)
    return frame


def fetch_runs(
    path: str,
    filters: dict[str, Any],
    *,
    order: str = "+created_at",
    per_page: int = 200,
    include_sweeps: bool = False,
) -> pd.DataFrame:
    api = wandb.Api()

    runs = api.runs(
        path=path,
        filters=filters,
        order=order,
        per_page=per_page,
        include_sweeps=include_sweeps,
        lazy=False,
    )
    return pd.DataFrame.from_records(_run_to_record(run) for run in runs)


def save_cached_runs(cache_file: Path, frame: pd.DataFrame) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = cache_file.with_suffix(".tmp")

    serializable = frame.copy()
    if "summary" in serializable.columns:
        serializable["summary"] = serializable["summary"].map(json.dumps)
    if "config" in serializable.columns:
        serializable["config"] = serializable["config"].map(json.dumps)
    if "tags" in serializable.columns:
        serializable["tags"] = serializable["tags"].map(json.dumps)

    serializable.to_csv(temp_file, index=False)
    temp_file.replace(cache_file)


def load_or_fetch_runs(
    path: str,
    filters: dict[str, Any],
    *,
    cache_dir: Path | None = None,
    refresh_cache: bool | None = None,
    order: str = "+created_at",
    per_page: int = 200,
    include_sweeps: bool = False,
) -> pd.DataFrame:
    if cache_dir is None:
        cache_dir = Path(__file__).with_name(".cache")

    if refresh_cache is None:
        refresh_cache = parse_args().refresh_cache

    query = {
        "path": path,
        "filters": filters,
        "order": order,
        "per_page": per_page,
        "include_sweeps": include_sweeps,
        # Bump when cached run schema changes (e.g. adding tags).
        "schema": 2,
    }
    cache_file = cache_path(cache_dir, query)

    if not refresh_cache:
        cached = load_cached_runs(cache_file)
        if cached is not None:
            return cached

    frame = fetch_runs(
        path,
        filters,
        order=order,
        per_page=per_page,
        include_sweeps=include_sweeps,
    )
    save_cached_runs(cache_file, frame)
    return frame
