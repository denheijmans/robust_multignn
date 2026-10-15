"""Tests for baseline artifact metadata validation (no W&B calls)."""

import pytest

from robust_multignn.loading import validate_baseline_artifact_metadata


def _baseline_metadata(**overrides):
    base = {
        "model_type": "multigin_scatter",
        "dataset": "Small_HI",
        "data_seed": 0,
        "seed": 0,
        "model_params": {
            "hidden_dim": 32,
            "num_layers": 2,
            "edge_dim": 4,
            "dropout": 0.1,
            "ego_dim": 1,
            "eps": 0.0,
            "train_eps": True,
            "port_vocab_size": 2048,
            "port_emb_dim": 8,
            "reverse_mp": True,
            "edge_updates": True,
            "in_dim": 10,
            "out_dim": 2,
        },
        "loader_params": {
            "batch_size": 8192,
            "num_neighbors": [[50, 50]],
            "shuffle": True,
        },
        "train_params": {
            "max_epochs": 80,
            "lr": 3e-3,
            "weight_decay": 1e-4,
            "min_epochs": 20,
            "patience": 10,
        },
    }
    base.update(overrides)
    return base


def _adv_kwargs(**overrides):
    base = {
        "init_artifact": "multigin_scatter-baseline-Small_HI_0-0",
        "model_type": "multigin_scatter",
        "dataset": "Small_HI",
        "data_seed": 0,
        "seed": 0,
        "hyperparams": _baseline_metadata()["model_params"],
        "loader_params": _baseline_metadata()["loader_params"],
        "train_params": {
            "max_epochs": 40,
            "lr": 3e-3,
            "weight_decay": 1e-4,
            "min_epochs": 20,
            "patience": 10,
        },
    }
    base.update(overrides)
    return base


def test_matching_metadata_returns_no_warnings():
    metadata = _baseline_metadata()
    train_params = metadata["train_params"]
    warnings = validate_baseline_artifact_metadata(
        metadata, **_adv_kwargs(train_params=train_params)
    )
    assert warnings == []


def test_model_type_mismatch_raises():
    metadata = _baseline_metadata(model_type="line_graph_gnn")
    with pytest.raises(ValueError, match="model_type mismatch"):
        validate_baseline_artifact_metadata(metadata, **_adv_kwargs())


def test_dataset_mismatch_raises():
    metadata = _baseline_metadata(dataset="Small_LI")
    with pytest.raises(ValueError, match="dataset mismatch"):
        validate_baseline_artifact_metadata(metadata, **_adv_kwargs())


def test_data_seed_mismatch_raises():
    metadata = _baseline_metadata(data_seed=1)
    with pytest.raises(ValueError, match="data_seed mismatch"):
        validate_baseline_artifact_metadata(metadata, **_adv_kwargs())


def test_model_params_mismatch_raises():
    metadata = _baseline_metadata()
    metadata["model_params"] = dict(metadata["model_params"], hidden_dim=64)
    with pytest.raises(ValueError, match="model_params mismatch"):
        validate_baseline_artifact_metadata(metadata, **_adv_kwargs())


def test_seed_mismatch_warns():
    metadata = _baseline_metadata(seed=1)
    warnings = validate_baseline_artifact_metadata(metadata, **_adv_kwargs())
    assert any("seed mismatch" in w for w in warnings)


def test_loader_params_mismatch_warns():
    metadata = _baseline_metadata()
    metadata["loader_params"] = dict(metadata["loader_params"], batch_size=4096)
    warnings = validate_baseline_artifact_metadata(metadata, **_adv_kwargs())
    assert any("loader_params mismatch" in w for w in warnings)


def test_train_params_mismatch_warns():
    metadata = _baseline_metadata()
    metadata["train_params"] = dict(metadata["train_params"], max_epochs=60)
    warnings = validate_baseline_artifact_metadata(metadata, **_adv_kwargs())
    assert any("train_params mismatch" in w for w in warnings)


def test_non_baseline_artifact_name_warns():
    metadata = _baseline_metadata()
    warnings = validate_baseline_artifact_metadata(
        metadata,
        **_adv_kwargs(init_artifact="multigin_scatter-adv_prbcd_aml_0.05-Small_HI_0-0"),
    )
    assert any("does not contain '-baseline-'" in w for w in warnings)
