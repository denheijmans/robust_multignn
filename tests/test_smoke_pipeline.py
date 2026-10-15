"""End-to-end CPU smoke tests (no W&B account or downloaded data needed).

Runs training, the uniform and PR-BCD attacks (with the cycle regularizer),
and adversarial training on a shrunken synthetic graph and on a tiny random
CSV in the IBM AML format. Model artifacts are kept in memory instead of W&B,
and W&B runs in disabled mode. This only checks that the code runs; it does
not reproduce paper numbers.

    pytest tests/test_smoke_pipeline.py
"""

import copy
import datetime as dt
import random
from pathlib import Path

import pytest
import wandb

import robust_multignn.synthetic.generator as generator
from experiments import (
    experiment_adv_train,
    experiment_adv_train_multigin,
    experiment_attack,
    experiment_train,
)
from robust_multignn.models import create_model

NUM_NODES = 256
DEVICE = "cpu"

MODEL_PARAMS = {
    "hidden_dim": 16,
    "num_layers": 2,
    "ego_dim": 4,
    "eps": 0.0,
    "train_eps": True,
    "dropout": 0.1,
    "port_vocab_size": 32,
    "port_emb_dim": 4,
    "reverse_mp": True,
    "edge_updates": True,
}
LOADER_PARAMS = {"batch_size": 64, "num_neighbors": [10, 10], "shuffle": True}
TRAIN_PARAMS = {"max_epochs": 1, "lr": 1e-3, "weight_decay": 1e-4}
COMMON = {"dataset": "subgraph_pattern_detection", "data_seed": 0, "device": DEVICE}


@pytest.fixture
def offline_pipeline(tmp_path, monkeypatch):
    """Shrink the generated graph and replace W&B artifacts by a dict."""
    simulator = generator.GraphSimulator
    monkeypatch.setattr(
        generator,
        "GraphSimulator",
        lambda **kwargs: simulator(**{**kwargs, "num_nodes": NUM_NODES}),
    )

    store = {}

    def save(state_dict, artifact_name, metadata):
        store[artifact_name] = (copy.deepcopy(state_dict), metadata)

    def load(artifact_name, version="latest", **create_model_kwargs):
        state_dict, metadata = store[artifact_name]
        model = create_model(
            metadata["model_type"], metadata["model_params"], **create_model_kwargs
        )
        model.load_state_dict(state_dict)
        return model, metadata

    monkeypatch.setattr(experiment_train, "save_model_artifact", save)
    monkeypatch.setattr(experiment_adv_train_multigin, "save_model_artifact", save)
    monkeypatch.setattr(experiment_adv_train, "save_model_artifact", save)
    monkeypatch.setattr(experiment_adv_train, "load_model_artifact", load)
    monkeypatch.setattr(experiment_attack, "load_model_artifact", load)

    wandb.init(mode="disabled")
    yield {"data_dir": str(tmp_path), **COMMON}
    wandb.finish()


def test_train_attack_adv_train(offline_pipeline):
    cfg = offline_pipeline
    attack_cfg = {
        **cfg,
        "model_seed": 0,
        "seed": 0,
        "model_type": "multigin",
        "loader_params": LOADER_PARAMS,
    }
    prbcd_params = {
        "max_epochs": 2,
        "max_resamples": 1,
        "lr": 0.1,
        "attack_ratio": 0.05,
        "sample_size": 2,
        "regularizer_terms": {"cycle_loss": 0.05},
    }

    experiment_train.run(
        **cfg,
        seed=0,
        model_type="multigin",
        model_params=MODEL_PARAMS,
        loader_params=LOADER_PARAMS,
        train_params=TRAIN_PARAMS,
    )

    experiment_attack.run(
        **attack_cfg,
        train_type="baseline",
        attack_type="uniform",
        attack_params={"attack_ratio": 0.05, "sample_size": 2},
    )
    experiment_attack.run(
        **attack_cfg,
        train_type="baseline",
        attack_type="prbcd",
        attack_params=prbcd_params,
    )

    experiment_adv_train_multigin.run(
        **cfg,
        seed=0,
        model_type="multigin",
        model_params=MODEL_PARAMS,
        loader_params=LOADER_PARAMS,
        pretrain_params=TRAIN_PARAMS,
        adv_train_params=TRAIN_PARAMS,
        attack_type="prbcd",
        attack_params=prbcd_params,
    )
    experiment_attack.run(
        **attack_cfg,
        train_type="adv",
        adv_train_params={"attack_type": "prbcd", "attack_ratio": 0.05},
        attack_type="prbcd",
        attack_params=prbcd_params,
    )


def _write_fake_aml_csv(path, num_accounts=400, num_transactions=8000, days=20):
    """Random transactions in the raw IBM AML CSV format."""
    rng = random.Random(0)
    accounts = [
        (f"{rng.randint(1, 50):03d}", f"{rng.getrandbits(36):09X}")
        for _ in range(num_accounts)
    ]
    currencies = ["US Dollar", "Euro", "Yuan"]
    formats = ["ACH", "Cheque", "Wire", "Credit Card"]
    start = dt.datetime(2022, 9, 1)
    path.parent.mkdir(parents=True)
    with open(path, "w") as f:
        f.write(
            "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
            "Receiving Currency,Amount Paid,Payment Currency,Payment Format,"
            "Is Laundering\n"
        )
        for _ in range(num_transactions):
            (from_bank, from_acc), (to_bank, to_acc) = rng.sample(accounts, 2)
            minutes = rng.randint(0, days * 24 * 60)
            ts = (start + dt.timedelta(minutes=minutes)).strftime("%Y/%m/%d %H:%M")
            currency = rng.choice(currencies)
            amount = f"{rng.uniform(1, 1e4):.2f}"
            f.write(
                f"{ts},{from_bank},{from_acc},{to_bank},{to_acc},{amount},"
                f"{currency},{amount},{currency},{rng.choice(formats)},"
                f"{int(rng.random() < 0.05)}\n"
            )


def test_aml_train_attack_adv_train(offline_pipeline):
    cfg = {**offline_pipeline, "dataset": "Small_HI"}
    _write_fake_aml_csv(Path(cfg["data_dir"]) / "Small_HI" / "HI-Small_Trans.csv")
    model_params = {
        "hidden_dim": 8,
        "num_layers": 2,
        "edge_dim": 4,
        "dropout": 0.1,
        "ego_dim": 1,
        "eps": 0.0,
        "train_eps": True,
        "port_vocab_size": 64,
        "port_emb_dim": 4,
        "reverse_mp": True,
        "edge_updates": True,
    }
    loader_params = {"batch_size": 512, "num_neighbors": [10, 10], "shuffle": True}
    train_params = {**TRAIN_PARAMS, "pos_weight": 7.0}
    prbcd_params = {
        "attack_ratio": 0.05,
        "sample_size": 2,
        "lr": 0.1,
        "max_epochs": 2,
        "max_resamples": 1,
        "block_size": [64, 512],
        "regularizer_terms": {"cycle_loss": 0.05},
    }
    attack_cfg = {
        **cfg,
        "model_seed": 0,
        "seed": 0,
        "model_type": "multigin_scatter",
        "loader_params": loader_params,
    }

    experiment_train.run(
        **cfg,
        seed=0,
        model_type="multigin_scatter",
        model_params=model_params,
        loader_params=loader_params,
        train_params=train_params,
    )

    for threat_model in ("unrestricted", "criminal"):
        experiment_attack.run(
            **attack_cfg,
            train_type="baseline",
            attack_type="uniform_aml",
            attack_params={"attack_ratio": 0.05, "sample_size": 2},
            threat_model=threat_model,
        )
        experiment_attack.run(
            **attack_cfg,
            train_type="baseline",
            attack_type="prbcd_aml",
            attack_params=prbcd_params,
            threat_model=threat_model,
        )

    experiment_adv_train.run(
        **cfg,
        seed=0,
        model_type="multigin_scatter",
        train_type="adv",
        init_artifact="multigin_scatter-baseline-Small_HI_0-0",
        model_params=model_params,
        loader_params=loader_params,
        train_params=train_params,
        attack_type="prbcd_aml",
        attack_params=prbcd_params,
    )
