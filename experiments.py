"""Exécuter le protocole fixé, puis évaluer tous les checkpoints sur le test."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

# Fixer la branche de calcul MKL avant les premières opérations numériques.
os.environ["MKL_CBWR"] = "COMPATIBLE"

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader

from data import SPLIT_SEED, load_dataset, make_loaders, make_split
from models import make_model
from training import evaluate, set_seed, train_epoch


PROTOCOL = {
    "models": ["mlp", "cnn"],
    "seeds": [0, 1, 2],
    "epochs": 15,
    "batch_size": 128,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "weight_decay": 0,
    "split_seed": SPLIT_SEED,
    "selection": "minimum validation loss; first epoch in a tie",
    "device": "cpu",
    "threads": 2,
    "num_workers": 0,
    "deterministic": True,
    "mkl_cbwr": "COMPATIBLE",
    "preprocessing": "float32 pixels divided by 255; no augmentation",
    "error_sample_seed": 2026,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, values):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def environment():
    cpu_name = platform.processor()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    sources = ["data.py", "models.py", "training.py", "experiments.py"]
    root = Path(__file__).resolve().parent
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_name,
        "mkl_cbwr": os.environ.get("MKL_CBWR"),
        "packages": {
            name: version(name)
            for name in ["torch", "torchvision", "numpy", "scikit-learn", "matplotlib", "pytest"]
        },
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sources
        },
        "torch_configuration": torch.__config__.show(),
    }


def train_run(run_dir, config, dataset, train_indices, validation_indices):
    run_dir.mkdir()
    save_json(run_dir / "config.json", config)
    state = {"status": "training", "started_at": utc_now()}
    save_json(run_dir / "status.json", state)

    try:
        set_seed(config["seed"])
        model = make_model(config["model"])
        train_loader, validation_loader = make_loaders(
            dataset, train_indices, validation_indices,
            config["seed"], config["batch_size"],
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"], weight_decay=0
        )
        best_loss = float("inf")
        best_epoch = None
        fields = [
            "epoch", "train_loss", "train_accuracy", "validation_loss",
            "validation_accuracy", "epoch_seconds",
        ]
        start = time.perf_counter()
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, config["epochs"] + 1):
                epoch_start = time.perf_counter()
                train_metrics = train_epoch(model, train_loader, optimizer)
                validation_metrics, _, _ = evaluate(model, validation_loader)
                epoch_seconds = time.perf_counter() - epoch_start
                writer.writerow({
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "epoch_seconds": epoch_seconds,
                })
                file.flush()
                if validation_metrics["loss"] < best_loss:
                    best_loss = validation_metrics["loss"]
                    best_epoch = epoch
                    torch.save(model.state_dict(), run_dir / "checkpoint.tmp")
                    (run_dir / "checkpoint.tmp").replace(run_dir / "checkpoint.pt")
                print(
                    f"{config['model']} seed={config['seed']} epoch={epoch:02d} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={validation_metrics['loss']:.4f} "
                    f"val_accuracy={validation_metrics['accuracy']:.4f} "
                    f"seconds={epoch_seconds:.1f}",
                    flush=True,
                )
        summary = {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "training_seconds": time.perf_counter() - start,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        save_json(run_dir / "training.json", summary)
        state.update(status="trained", finished_at=utc_now())
        save_json(run_dir / "status.json", state)
        return True
    except (Exception, KeyboardInterrupt) as error:
        state.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            finished_at=utc_now(), error=f"{type(error).__name__}: {error}",
        )
        save_json(run_dir / "status.json", state)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if isinstance(error, KeyboardInterrupt):
            raise
        print(f"Échec conservé dans {run_dir}: {error}", file=sys.stderr, flush=True)
        return False


def evaluate_run(run_dir, dataset, config, split):
    set_seed(config["seed"])
    model = make_model(config["model"])
    model.load_state_dict(
        torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    metrics, targets, predictions = evaluate(model, loader)
    metrics.update(
        macro_f1=float(f1_score(targets, predictions, labels=list(range(10)), average="macro", zero_division=0)),
        split=split,
        examples=len(targets),
    )
    matrix = confusion_matrix(targets, predictions, labels=list(range(10)))
    save_json(run_dir / "metrics.json", metrics)
    np.savetxt(run_dir / "confusion.csv", matrix, fmt="%d", delimiter=",")
    np.savetxt(
        run_dir / "predictions.csv",
        np.column_stack((np.arange(len(targets)), targets, predictions)),
        fmt="%d", delimiter=",", header="index,target,prediction", comments="",
    )
    if split == "test" and config["seed"] == 0:
        errors = np.flatnonzero(targets != predictions)
        chosen = np.random.default_rng(config["error_sample_seed"]).choice(
            errors, size=min(12, len(errors)), replace=False
        )
        np.savez_compressed(
            run_dir / "error_samples.npz",
            indices=chosen,
            images=dataset.data[chosen].numpy(),
            targets=targets[chosen],
            predictions=predictions[chosen],
        )
    state = json.loads((run_dir / "status.json").read_text())
    state.update(status="complete", evaluated_at=utc_now(), evaluation_split=split)
    save_json(run_dir / "status.json", state)
    print(f"{run_dir.name}: {split} accuracy={metrics['accuracy']:.4f}", flush=True)


def run_study(output_dir, data_dir, smoke=False):
    # Un chemin déjà présent n'est jamais réutilisé, même après un échec.
    output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(PROTOCOL)
    config["smoke"] = smoke
    if smoke:
        config.update(seeds=[0], epochs=2)
    save_json(output_dir / "protocol.json", config)
    state = {"status": "preparing", "started_at": utc_now()}
    save_json(output_dir / "status.json", state)
    try:
        save_json(output_dir / "environment.json", environment())
        dataset = load_dataset(data_dir, train=True)
        train_indices, validation_indices = make_split(dataset.targets)
        if smoke:
            train_indices = train_indices[:1024]
            validation_indices = validation_indices[:256]
        np.savez_compressed(
            output_dir / "split.npz", train=train_indices, validation=validation_indices
        )
        config.update(train_size=len(train_indices), validation_size=len(validation_indices))
        save_json(output_dir / "protocol.json", config)
        state["status"] = "training"
        save_json(output_dir / "status.json", state)

        runs = []
        successes = []
        for seed in config["seeds"]:
            for model_name in config["models"]:
                run_config = dict(config, model=model_name, seed=seed)
                run_dir = output_dir / f"{model_name}-seed-{seed}"
                successes.append(train_run(
                    run_dir, run_config, dataset, train_indices, validation_indices
                ))
                runs.append((run_dir, run_config))

        if not all(successes):
            state.update(status="incomplete", finished_at=utc_now(), error="Au moins un entraînement a échoué ; aucune évaluation finale.")
            save_json(output_dir / "status.json", state)
            return False

        state["status"] = "evaluating"
        save_json(output_dir / "status.json", state)
        if smoke:
            evaluation_data = torch.utils.data.Subset(dataset, validation_indices)
            split = "validation_smoke"
        else:
            # Le jeu de test n'est chargé qu'après les six entraînements réussis.
            evaluation_data = load_dataset(data_dir, train=False)
            split = "test"
        for run_dir, run_config in runs:
            try:
                evaluate_run(run_dir, evaluation_data, run_config, split)
            except (Exception, KeyboardInterrupt) as error:
                run_state = json.loads((run_dir / "status.json").read_text())
                run_state.update(
                    status="interrupted" if isinstance(error, KeyboardInterrupt) else "evaluation_failed",
                    error=f"{type(error).__name__}: {error}",
                )
                save_json(run_dir / "status.json", run_state)
                (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                raise
        state.update(status="complete", finished_at=utc_now())
        save_json(output_dir / "status.json", state)
        return True
    except (Exception, KeyboardInterrupt) as error:
        state.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            finished_at=utc_now(), error=f"{type(error).__name__}: {error}",
        )
        save_json(output_dir / "status.json", state)
        (output_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Essai technique court sans test officiel")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, help="Nouveau dossier exclusivement ; jamais écrasé")
    args = parser.parse_args()
    if args.output_dir is None:
        parent = Path("results/smoke" if args.smoke else "results")
        args.output_dir = parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    print(f"Résultats : {args.output_dir}", flush=True)
    try:
        complete = run_study(args.output_dir, args.data_dir, args.smoke)
    except KeyboardInterrupt:
        return 130
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
