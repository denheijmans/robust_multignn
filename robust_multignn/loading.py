import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

import wandb
from experiments.common import wandb_disabled
from robust_multignn.aml import AML_DATASET_NAMES, AMLDataLoader
from robust_multignn.data import GraphData
from robust_multignn.models import Model, create_model
from robust_multignn.synthetic.generator import (
    BALANCE_TYPES,
    define_subtasks_thresholds_and_witness_builders,
    generate_named_multigraph_splits,
    generate_synthetic_graph,
    split_seed,
)
from robust_multignn.utils import log_table


@dataclass
class GraphDataset:
    train: GraphData
    val: GraphData
    test: GraphData
    class_names: List[str]


@wandb_disabled()
def load_subgraph_pattern_detection_dataset(
    seed: int,
) -> GraphDataset:
    splits = {}
    for split in ["train", "val", "test"]:
        splits[split] = generate_synthetic_graph(split_seed(seed, split))

    _, _, _, task_names, _ = define_subtasks_thresholds_and_witness_builders()
    return GraphDataset(**splits, class_names=task_names)


@wandb_disabled()
def load_aml_dataset(name: str, data_dir: str) -> GraphDataset:
    """Load an AML transaction dataset and return a :class:`GraphDataset` with
    cumulative temporal splits and per-split evaluation index masks.

    Parameters
    ----------
    name : str
        Dataset variant, one of :data:`robust_multignn.aml.AML_DATASET_NAMES`
        (e.g. ``"Small_HI"``).
    data_dir : str
        Root directory containing a sub-folder named ``name`` with the raw CSV.

    Returns
    -------
    GraphDataset
        Dataset whose ``train_inds``, ``val_inds``, and ``test_inds`` fields
        hold the edge indices within each split that belong to that evaluation
        period.
    """
    tr_data, val_data, te_data = AMLDataLoader(
        name, data_dir
    ).get_data()
    return GraphDataset(
        train=tr_data,
        val=val_data,
        test=te_data,
        class_names=["is_laundering"]
    )


@wandb_disabled()
def load_cycle_dataset(
    seed: int,
    data_balance: str | None = None,
) -> GraphDataset:
    splits, class_names, _ = generate_named_multigraph_splits(
        seed=seed,
        selected_family=data_balance,
    )
    return GraphDataset(**splits, class_names=class_names)


def load_dataset(
    name: str,
    data_dir: str,
    seed: int,
) -> GraphDataset:
    """Load a dataset by name, either from cache or by generating it. The dataset is
    stored in a specified directory and is associated with a seed for reproducibility.
    If the dataset already exists in the cache, it is loaded directly; otherwise, it is
    generated and saved for future use.

    Parameters
    ----------
    name : str
        Name of the dataset to load (e.g., "subgraph_pattern_detection").
    data_dir : str
        Directory where the dataset should be stored or loaded from.
    seed : int
        Seed for reproducibility of graph generation and training.

    Returns
    -------
    GraphDataset
        A GraphDataset object containing the training, validation, and test splits
        of the dataset.

    Raises
    ------
    NotImplementedError
        If the specified dataset name is not supported.
    """
    # Create subdirectory for dataset name
    data_path = Path(data_dir)
    dataset_path = data_path / name
    dataset_path.mkdir(parents=True, exist_ok=True)

    if name in AML_DATASET_NAMES:
        # AML datasets are deterministic given the CSV (no seed dependence),
        # and AMLDataLoader already caches via data.pt. A seed-keyed cache on
        # top of that would just be a second, independently-stale copy of the
        # same data, so load straight from AMLDataLoader instead.
        dataset = load_aml_dataset(name, data_dir)
    else:
        # Use cached dataset if available, otherwise generate and cache it
        cache_file: Path = dataset_path / f"seed_{seed}.pt"
        if cache_file.exists():
            dataset: GraphDataset = torch.load(cache_file, weights_only=False)
        else:
            if name in BALANCE_TYPES:
                dataset = load_cycle_dataset(seed, name)
            elif name == "subgraph_pattern_detection":
                dataset = load_subgraph_pattern_detection_dataset(seed)
            else:
                raise NotImplementedError(f"Dataset with name '{name}' is not supported")
            torch.save(dataset, cache_file)

    # Log positive ratios for each task in the test set
    if name not in AML_DATASET_NAMES:
        positive_ratios = dataset.test.y.float().mean(dim=0)
        log_table("Positive Ratios", dataset.class_names, positive_ratios)

    return dataset


def save_model_artifact(
    state_dict: Dict[str, Any],
    artifact_name: str,
    metadata: Dict[str, Any],
):
    """Save model state dict as a wandb artifact.

    Parameters
    ----------
    state_dict : Dict[str, Any]
        Model state dictionary to save.
    artifact_name : str
        Name for the wandb artifact.
    metadata : Dict[str, Any]
        Metadata needed to reconstruct the model (e.g., model name and params).
    """
    # Create a new wandb artifact for the model
    artifact = wandb.Artifact(name=artifact_name, type="model")
    artifact.metadata = metadata

    # Save state dict to temporary file and add to artifact
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = f"{temp_dir}/{artifact_name}.pt"
        torch.save(state_dict, model_path)
        artifact.add_file(model_path)

    # Log artifact
    wandb.log_artifact(artifact)


RESUME_FILENAME = "resume.pt"


def resume_state_path(resume_dir: str | Path, artifact_name: str) -> Path:
    """Where the resume file of the run named ``artifact_name`` lives.

    One subfolder per run, keyed on the artifact name, so runs sharing a
    ``resume_dir`` never overwrite each other's state.
    """
    return Path(resume_dir) / artifact_name / RESUME_FILENAME


def save_resume_state(payload: Dict[str, Any], path: str | Path) -> None:
    """Write full training state to a local file so a crashed run can continue.

    Deliberately a local file rather than a W&B artifact: resume state is only
    ever needed by the next attempt of the same run, and uploading it every
    epoch stores a full copy per epoch on the server and in the local W&B cache.

    The file is overwritten each epoch, since only the latest state is needed to
    continue. It is written to a temporary file first and then renamed over the
    old one; the rename is atomic on POSIX, so a crash mid-write leaves the
    previous epoch's file intact instead of a truncated one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_resume_state(path: str | Path) -> Dict[str, Any]:
    """Load a payload written by :func:`save_resume_state`.

    Returns the raw dict rather than a reconstructed model: the caller already
    has a model to load the weights into, and needs the optimizer state and
    counters as well. Tensors are mapped to CPU so a run can resume on a
    different GPU (or none); ``load_state_dict`` moves them back onto the
    model's device.

    Raises ``FileNotFoundError`` if there is no resume file yet.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = {"model", "optimizer", "epoch"} - set(payload)
    if missing:
        raise ValueError(f"resume file {str(path)!r} is missing {sorted(missing)}")
    return payload


def load_model_artifact(
    artifact_name: str,
    version: str = "latest",
    **create_model_kwargs: Any,
) -> Tuple[Model, Dict[str, Any]]:
    """Load and reconstruct a model from a wandb artifact.

    Parameters
    ----------
    artifact_name : str
        Name of the wandb artifact to load.
    version : str, optional
        Version of the artifact to load, by default "latest".
    **create_model_kwargs : Any
        Extra keyword arguments forwarded to :func:`create_model`. Used for
        models that need construction-time arguments not captured in the
        saved metadata (e.g. ``train_inds``/``val_inds``/``test_inds`` for
        :class:`~robust_multignn.models.line_graph_gnn.LineGraphGNNModel`,
        whose buffers are then overwritten by ``state_dict`` load).

    Returns
    -------
    Model
        Reconstructed model with loaded weights.
    Dict[str, Any]
        Metadata stored with the artifact.
    """
    # Load artifact from wandb
    artifact = wandb.use_artifact(f"{artifact_name}:{version}", type="model")
    source_run = artifact.logged_by()
    print(
        f"[load_model_artifact] resolved artifact : {artifact.name}\n"
        f"[load_model_artifact] source W&B run    : {source_run.url if source_run else 'unknown'}"
    )

    # Load state dict from artifact
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = artifact.download(root=temp_dir)
        model_path = Path(artifact_dir) / f"{artifact_name}.pt"
        state_dict = torch.load(model_path, weights_only=False)

    # Reconstruct model
    metadata = artifact.metadata
    model = create_model(
        metadata["model_type"],
        metadata["model_params"],
        **create_model_kwargs,
    )
    model.load_state_dict(state_dict)
    return model, metadata


_MISSING = object()


def _dict_mismatches(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
) -> Dict[str, Tuple[Any, Any]]:
    """Return keys whose values differ between two flat dicts."""
    mismatches: Dict[str, Tuple[Any, Any]] = {}
    for key in set(expected) | set(actual):
        exp_val = expected.get(key, _MISSING)
        act_val = actual.get(key, _MISSING)
        if exp_val != act_val:
            mismatches[key] = (exp_val, act_val)
    return mismatches


def _format_mismatch_message(label: str, mismatches: Dict[str, Tuple[Any, Any]]) -> str:
    parts = [
        f"{key}: expected {exp!r}, got {act!r}"
        for key, (exp, act) in sorted(mismatches.items())
    ]
    return f"{label} mismatch: " + "; ".join(parts)


def validate_baseline_artifact_metadata(
    artifact_metadata: Dict[str, Any],
    *,
    init_artifact: str | None,
    model_type: str,
    dataset: str,
    data_seed: int,
    seed: int,
    hyperparams: Dict[str, Any],
    loader_params: Dict[str, Any],
    train_params: Dict[str, Any],
) -> List[str]:
    """Check that a baseline W&B artifact matches the adv-train configuration.

    Raises ``ValueError`` on critical mismatches (model/data identity).
    Returns non-fatal warning messages for training-hyperparameter drift.
    """
    warnings: List[str] = []

    if init_artifact and "-baseline-" not in init_artifact:
        warnings.append(
            f"init_artifact {init_artifact!r} does not contain '-baseline-'; "
            "ensure this is a baseline training checkpoint"
        )

    if artifact_metadata.get("model_type") != model_type:
        raise ValueError(
            f"model_type mismatch: artifact has "
            f"{artifact_metadata.get('model_type')!r}, config has {model_type!r}"
        )

    if artifact_metadata.get("dataset") != dataset:
        raise ValueError(
            f"dataset mismatch: artifact has "
            f"{artifact_metadata.get('dataset')!r}, config has {dataset!r}"
        )

    if artifact_metadata.get("data_seed") != data_seed:
        raise ValueError(
            f"data_seed mismatch: artifact has "
            f"{artifact_metadata.get('data_seed')!r}, config has {data_seed!r}"
        )

    artifact_model_params = artifact_metadata.get("model_params", {})
    model_param_mismatches = _dict_mismatches(hyperparams, artifact_model_params)
    if model_param_mismatches:
        raise ValueError(
            _format_mismatch_message("model_params", model_param_mismatches)
        )

    if artifact_metadata.get("seed") != seed:
        warnings.append(
            f"seed mismatch: artifact has {artifact_metadata.get('seed')!r}, "
            f"config has {seed!r}"
        )

    artifact_loader = artifact_metadata.get("loader_params", {})
    loader_mismatches = _dict_mismatches(loader_params, artifact_loader)
    if loader_mismatches:
        warnings.append(_format_mismatch_message("loader_params", loader_mismatches))

    artifact_train = artifact_metadata.get("train_params", {})
    train_mismatches = _dict_mismatches(train_params, artifact_train)
    if train_mismatches:
        warnings.append(_format_mismatch_message("train_params", train_mismatches))

    return warnings


def create_config_name(
    arch: str,
    train_type: str,
    dataset: str,
    data_seed: int,
    model_seed: int,
    adv_train_params: Dict[str, Any] = None,
) -> str:
    """Creates a unique name for a given training configuration based on the provided
    parameters.

    The name format is as follows:
    - For adversarial training:
        "{arch}-adv_{attack_type}_{attack_ratio}-{dataset}_{data_seed}-{model_seed}"
    - For other training types:
        "{arch}-{train_type}-{dataset}_{data_seed}-{model_seed}"

    Parameters
    ----------
    arch : str
        The architecture of the model (e.g., "gin").
    train_type : str
        The type of training (e.g., "baseline", "adv").
    dataset : str
        The name of the dataset.
    data_seed : int
        The random seed for data generation and splitting.
    model_seed : int
        The random seed for model initialization and training.
    adv_train_params : Dict[str, Any], optional
        The parameters for adversarial training, by default None.

    Returns
    -------
    str
        A unique name for the training configuration.
    """
    # If adversarial training, include attack type and ratio in the name
    if train_type == "adv":
        assert adv_train_params is not None
        attack_type = adv_train_params["attack_type"]
        attack_ratio = adv_train_params["attack_ratio"]
        return (
            f"{arch}-{train_type}_{attack_type}_{attack_ratio}-"
            f"{dataset}_{data_seed}-{model_seed}"
        )

    # For non-adversarial training, just include training type
    return f"{arch}-{train_type}-{dataset}_{data_seed}-{model_seed}"
