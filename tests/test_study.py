import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import experiments
from data import make_loaders, make_split
from models import CNN, MLP
from report import collect_runs, summarize
from training import evaluate, set_seed, train_epoch


def test_split_is_stratified_disjoint_and_repeatable():
    labels = np.repeat(np.arange(10), 6000)
    train, validation = make_split(labels)
    assert len(train) == 50_000
    assert len(validation) == 10_000
    assert not np.intersect1d(train, validation).size
    np.testing.assert_array_equal(np.sort(np.r_[train, validation]), np.arange(60_000))
    np.testing.assert_array_equal(np.bincount(labels[train]), np.full(10, 5000))
    np.testing.assert_array_equal(np.bincount(labels[validation]), np.full(10, 1000))
    repeated = make_split(labels)
    np.testing.assert_array_equal(train, repeated[0])
    np.testing.assert_array_equal(validation, repeated[1])


def test_shuffle_does_not_depend_on_model_initialization():
    dataset = TensorDataset(torch.arange(40).float().view(-1, 1), torch.arange(40))
    orders = []
    for model_class in [MLP, CNN]:
        set_seed(0)
        model_class()
        loader, _ = make_loaders(dataset, np.arange(30), np.arange(30, 40), 0, 8)
        orders.append([torch.cat([labels for _, labels in loader]) for _ in range(2)])
    for first, second in zip(*orders):
        assert torch.equal(first, second)
    assert not torch.equal(orders[0][0], orders[0][1])


@pytest.mark.parametrize("model_class", [MLP, CNN])
def test_training_checkpoint_and_repeatability(model_class, tmp_path):
    generator = torch.Generator().manual_seed(8)
    images = torch.rand(9, 1, 28, 28, generator=generator)
    labels = torch.arange(9) % 10
    dataset = TensorDataset(images, labels)
    states = []
    for _ in range(2):
        set_seed(7)
        model = model_class()
        assert model(images).shape == (9, 10)
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_epoch(model, loader, optimizer)
        assert any(not torch.equal(initial[name], value) for name, value in model.state_dict().items())
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        states.append({name: value.clone() for name, value in model.state_dict().items()})
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)

    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = model_class()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        assert torch.equal(model(images), restored(images))
    before = {name: value.clone() for name, value in restored.state_dict().items()}
    evaluate(restored, loader)
    assert all(torch.equal(before[name], value) for name, value in restored.state_dict().items())


def test_losses_are_weighted_by_examples():
    logits = torch.tensor([[5.0, 0], [4.0, 0], [3.0, 0], [2.0, 0], [0.0, 7.0]])
    labels = torch.zeros(5, dtype=torch.long)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    expected = nn.CrossEntropyLoss()(logits, labels).item()
    metrics, targets, predictions = evaluate(model, loader)
    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert targets.tolist() == [0] * 5
    assert predictions.tolist() == [0, 0, 0, 0, 1]
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    assert train_epoch(model, loader, optimizer)["loss"] == pytest.approx(expected)


def test_validation_ties_keep_the_first_checkpoint(tmp_path, monkeypatch):
    dataset = TensorDataset(torch.zeros(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))
    config = dict(experiments.PROTOCOL, model="mlp", seed=0, epochs=3)

    def fake_epoch(model, loader, optimizer):
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return {"loss": 1.0, "accuracy": 0.5}

    losses = iter([2.0, 1.0, 1.0])
    monkeypatch.setattr(experiments, "train_epoch", fake_epoch)
    monkeypatch.setattr(experiments, "evaluate", lambda *args: ({"loss": next(losses), "accuracy": 0.5}, None, None))
    run_dir = tmp_path / "run"
    assert experiments.train_run(run_dir, config, dataset, [0, 1], [2, 3])
    assert json.loads((run_dir / "training.json").read_text())["best_epoch"] == 2


@pytest.mark.parametrize("failure", [False, True])
def test_test_data_is_only_loaded_after_all_training(tmp_path, monkeypatch, failure):
    events = []
    dataset = TensorDataset(torch.zeros(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))
    dataset.targets = torch.zeros(4, dtype=torch.long)

    def fake_load(data_dir, train):
        events.append("load_train" if train else "load_test")
        return dataset

    def fake_train(run_dir, config, *args):
        events.append("train")
        return not (failure and config["seed"] == 1 and config["model"] == "cnn")

    monkeypatch.setattr(experiments, "environment", lambda: {})
    monkeypatch.setattr(experiments, "load_dataset", fake_load)
    monkeypatch.setattr(experiments, "make_split", lambda _: (np.array([0, 1]), np.array([2, 3])))
    monkeypatch.setattr(experiments, "train_run", fake_train)
    monkeypatch.setattr(experiments, "evaluate_run", lambda *args: events.append("evaluate"))
    assert experiments.run_study(tmp_path / "study", tmp_path / "data") is not failure
    assert events.count("train") == 6
    if failure:
        assert "load_test" not in events
        assert "evaluate" not in events
    else:
        assert events == ["load_train"] + ["train"] * 6 + ["load_test"] + ["evaluate"] * 6


def test_failed_run_is_preserved(tmp_path, monkeypatch):
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.zeros(2, dtype=torch.long))

    def fail(*args):
        raise ValueError("Erreur simulée pour le test")

    monkeypatch.setattr(experiments, "train_epoch", fail)
    run_dir = tmp_path / "failed"
    config = dict(experiments.PROTOCOL, model="mlp", seed=0)
    assert not experiments.train_run(run_dir, config, dataset, [0], [1])
    state = json.loads((run_dir / "status.json").read_text())
    assert state["status"] == "failed"
    assert "Erreur simulée" in state["error"]
    assert (run_dir / "error.txt").exists()
    with pytest.raises(FileExistsError):
        experiments.train_run(run_dir, config, dataset, [0], [1])


def test_existing_study_is_not_overwritten(tmp_path):
    marker = tmp_path / "preserve.txt"
    marker.write_text("résultat antérieur")
    with pytest.raises(FileExistsError):
        experiments.run_study(tmp_path, tmp_path)
    assert marker.read_text() == "résultat antérieur"


def test_report_includes_failed_and_missing_runs(tmp_path):
    protocol = dict(experiments.PROTOCOL)
    protocol["smoke"] = False
    run_dir = tmp_path / "mlp-seed-0"
    run_dir.mkdir()
    experiments.save_json(run_dir / "status.json", {"status": "failed", "error": "Échec simulé"})
    rows = collect_runs(tmp_path, protocol)
    assert len(rows) == 6
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "Échec simulé"
    assert sum(row["status"] == "not_started" for row in rows) == 5
    assert all(row["n"] == 0 for row in summarize(rows, protocol))


def test_summary_uses_all_seeds_and_sample_standard_deviation():
    rows = [
        {"model": "mlp", "status": "complete", "split": "test", "accuracy": accuracy,
         "macro_f1": accuracy, "training_seconds": seconds}
        for accuracy, seconds in [(0.7, 10), (0.8, 20), (0.9, 30)]
    ]
    result = summarize(rows, dict(experiments.PROTOCOL, smoke=False))[0]
    assert result["n"] == 3
    assert result["accuracy_mean"] == pytest.approx(0.8)
    assert result["accuracy_std"] == pytest.approx(0.1)
    assert result["training_seconds_mean"] == pytest.approx(20)
    assert result["training_seconds_std"] == pytest.approx(10)
