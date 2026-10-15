"""Tests for the `on_new_best` checkpoint callback in `robust_multignn.train.train_aml`.

These tests stub out `train_edge_epoch`/`val_edge_epoch`/`wandb.log` so the
loop's control flow (which epochs count as "new best", how the callback is
invoked, and crash-safety when the callback fails) can be exercised without a
real graph, data loader, or W&B session.
"""

import copy

import torch

from robust_multignn.train import train_aml


class _TinyModel(torch.nn.Module):
    """Minimal model with a single scalar parameter we can snapshot per epoch."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def _stub_epoch_fns(monkeypatch, val_pr_aucs, model):
    """Patch train_edge_epoch/val_edge_epoch so epoch `i` reports
    `val_pr_aucs[i]` and tags the model's weight with `i`, without touching
    real loaders/criterion/optimizer."""
    calls = {"n": 0}

    def fake_train_edge_epoch(model, loader, criterion, optimizer, split):
        return 0.0, None, None

    def fake_val_edge_epoch(model, loader, split):
        idx = calls["n"]
        calls["n"] += 1
        with torch.no_grad():
            model.weight.fill_(float(idx))
        return 0.0, val_pr_aucs[idx], 0.0, 0.0

    monkeypatch.setattr("robust_multignn.train.train_edge_epoch", fake_train_edge_epoch)
    monkeypatch.setattr("robust_multignn.train.val_edge_epoch", fake_val_edge_epoch)
    monkeypatch.setattr("robust_multignn.train.wandb.log", lambda *a, **k: None)


def _run(monkeypatch, val_pr_aucs, on_new_best=None):
    model = _TinyModel()
    _stub_epoch_fns(monkeypatch, val_pr_aucs, model)
    best_state = train_aml(
        model=model,
        train_loader=None,
        val_loader=None,
        max_epochs=len(val_pr_aucs),
        lr=1e-3,
        weight_decay=0.0,
        class_weights=1.0,
        class_names=["is_laundering"],
        on_new_best=on_new_best,
    )
    return best_state


def test_on_new_best_called_once_per_improvement(monkeypatch):
    # Improves at epochs 0 (0.5 > -inf), 1 (0.6 > 0.5), 3 (0.8 > 0.6).
    # Epoch 2 (0.6, ties previous best) and epoch 4 (0.7, below best) do not.
    val_pr_aucs = [0.5, 0.6, 0.6, 0.8, 0.7]
    checkpointed = []

    def on_new_best(state_dict):
        checkpointed.append(copy.deepcopy(state_dict))

    best_state = _run(monkeypatch, val_pr_aucs, on_new_best=on_new_best)

    assert [w["weight"].item() for w in checkpointed] == [0.0, 1.0, 3.0]
    assert best_state["weight"].item() == 3.0


def test_on_new_best_not_called_when_no_improvement_after_first_epoch(monkeypatch):
    val_pr_aucs = [0.9, 0.5, 0.4]
    checkpointed = []

    best_state = _run(
        monkeypatch, val_pr_aucs, on_new_best=lambda sd: checkpointed.append(sd)
    )

    # Only epoch 0 ever improves over -inf.
    assert len(checkpointed) == 1
    assert best_state["weight"].item() == 0.0


def test_none_callback_is_a_safe_no_op(monkeypatch):
    # Should behave exactly like the pre-existing training loop when no
    # callback is supplied.
    best_state = _run(monkeypatch, [0.5, 0.7], on_new_best=None)
    assert best_state["weight"].item() == 1.0


def test_failing_callback_is_caught_and_training_continues(monkeypatch, caplog):
    val_pr_aucs = [0.5, 0.6, 0.7]
    call_count = {"n": 0}

    def failing_on_new_best(state_dict):
        call_count["n"] += 1
        raise RuntimeError("simulated W&B outage")

    with caplog.at_level("ERROR"):
        best_state = _run(monkeypatch, val_pr_aucs, on_new_best=failing_on_new_best)

    # All three epochs improve, so the callback is invoked each time...
    assert call_count["n"] == 3
    # ...but the exception never propagates: training finishes normally and
    # still returns the correct best (final, best-performing) state.
    assert best_state["weight"].item() == 2.0
    assert any(
        "Failed to persist new-best checkpoint" in record.message
        for record in caplog.records
    )
