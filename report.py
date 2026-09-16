"""Régénérer le rapport à partir des mesures, sans modèle ni téléchargement."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from data import CLASS_NAMES


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_runs(results_dir, protocol):
    rows = []
    for seed in protocol["seeds"]:
        for model in protocol["models"]:
            run_dir = results_dir / f"{model}-seed-{seed}"
            status_path = run_dir / "status.json"
            status = read_json(status_path) if status_path.exists() else {"status": "not_started"}
            row = {"model": model, "seed": seed, "status": status["status"], "error": status.get("error", "")}
            training_path = run_dir / "training.json"
            if training_path.exists():
                row.update(read_json(training_path))
            if row["status"] == "complete":
                row.update(read_json(run_dir / "metrics.json"))
            rows.append(row)
    return rows


def summarize(rows, protocol):
    summary = []
    expected_split = "validation_smoke" if protocol["smoke"] else "test"
    for model in protocol["models"]:
        complete = [
            row for row in rows
            if row["model"] == model and row["status"] == "complete"
            and row.get("split") == expected_split
        ]
        result = {"model": model, "n": len(complete), "expected": len(protocol["seeds"])}
        for metric in ["accuracy", "macro_f1", "training_seconds"]:
            values = [row[metric] for row in complete]
            result[f"{metric}_mean"] = float(np.mean(values)) if values else None
            result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(result)
    return summary


def save_table(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(results_dir, report_dir, protocol):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="row")
    colors = ["#176b9b", "#b95720", "#47853c"]
    for column, model in enumerate(protocol["models"]):
        for index, seed in enumerate(protocol["seeds"]):
            path = results_dir / f"{model}-seed-{seed}" / "history.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as file:
                history = list(csv.DictReader(file))
            if not history:
                continue
            epochs = [int(row["epoch"]) for row in history]
            for row_index, metric in enumerate(["loss", "accuracy"]):
                ax = axes[row_index, column]
                for split, linestyle in [("train", "--"), ("validation", "-")]:
                    values = [float(row[f"{split}_{metric}"]) for row in history]
                    ax.plot(epochs, values, linestyle, color=colors[index % len(colors)],
                            label=f"{split}, graine {seed}", linewidth=1.2)
        axes[0, column].set_title(model.upper())
        axes[0, column].set_ylabel("Entropie croisée")
        axes[1, column].set_ylabel("Exactitude")
        axes[1, column].set_xlabel("Époque")
        for ax in axes[:, column]:
            ax.grid(alpha=0.2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if axes[0, column].lines:
            axes[0, column].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(report_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


def plot_confusion(run_dir, report_dir):
    matrix = np.loadtxt(run_dir / "confusion.csv", delimiter=",", dtype=int)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(10):
        for column in range(10):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8,
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=55, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.set(xlabel="Classe prédite", ylabel="Classe réelle", title=run_dir.name)
    fig.colorbar(image, ax=ax, label="Nombre d'images")
    fig.tight_layout()
    filename = f"{run_dir.name}-confusion.png"
    fig.savefig(report_dir / filename, dpi=140)
    plt.close(fig)
    return filename


def plot_errors(run_dir, report_dir):
    path = run_dir / "error_samples.npz"
    if not path.exists():
        return None
    with np.load(path) as samples:
        fig, axes = plt.subplots(3, 4, figsize=(10, 8))
        for ax in axes.flat:
            ax.axis("off")
        for index, ax in enumerate(axes.flat):
            if index >= len(samples["indices"]):
                break
            ax.imshow(samples["images"][index], cmap="gray", vmin=0, vmax=255)
            target = CLASS_NAMES[int(samples["targets"][index])]
            prediction = CLASS_NAMES[int(samples["predictions"][index])]
            ax.set_title(f"#{samples['indices'][index]} : {target}\nPrédit : {prediction}", fontsize=9)
        fig.suptitle(f"{run_dir.name} — échantillon aléatoire des erreurs")
        if len(samples["indices"]) == 0:
            fig.text(0.5, 0.5, "Aucune erreur observée", ha="center")
    fig.tight_layout()
    filename = f"{run_dir.name}-errors.png"
    fig.savefig(report_dir / filename, dpi=140)
    plt.close(fig)
    return filename


def format_number(value, scale=1):
    return "—" if value is None else f"{value * scale:.2f}"


def generate_report(results_dir):
    protocol = read_json(results_dir / "protocol.json")
    state = read_json(results_dir / "status.json")
    rows = collect_runs(results_dir, protocol)
    summary = summarize(rows, protocol)
    report_dir = results_dir / "report"
    report_dir.mkdir(exist_ok=True)
    # Seuls les fichiers dérivés sont remplacés ; les mesures restent intactes.
    fields = ["model", "seed", "status", "split", "examples", "best_epoch", "parameters",
              "accuracy", "macro_f1", "loss", "training_seconds", "error"]
    save_table(report_dir / "runs.csv", rows, fields)
    save_table(report_dir / "summary.csv", summary, list(summary[0]))
    plot_curves(results_dir, report_dir, protocol)

    complete = state["status"] == "complete" and all(row["status"] == "complete" for row in rows)
    title = "Vérification technique — validation uniquement" if protocol["smoke"] else "Résultats Fashion-MNIST"
    lines = [f"# {title}", "", f"Statut enregistré : **{state['status']}**.", ""]
    if not complete:
        lines += ["**Comparaison incomplète : les exécutions absentes ou échouées restent visibles ci-dessous.**",
                  "Un statut actif peut signaler un processus encore en cours ou un arrêt brutal non intercepté.", ""]
    if protocol["smoke"]:
        lines += ["Cet essai sur sous-ensemble ne constitue pas un résultat scientifique sur le test.", ""]
    lines += ["## Toutes les exécutions", "",
              "| Modèle | Graine | Statut | Époque retenue | Paramètres | Exactitude (%) | Macro-F1 (%) | Durée (s) |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['status']} | {row.get('best_epoch', '—')} | "
            f"{row.get('parameters', '—')} | {format_number(row.get('accuracy'), 100)} | "
            f"{format_number(row.get('macro_f1'), 100)} | {format_number(row.get('training_seconds'))} |"
        )
    errors = [row for row in rows if row["error"]]
    if errors:
        lines += ["", "Erreurs enregistrées :", ""]
        lines += [f"- {row['model']}, graine {row['seed']} : {row['error']}" for row in errors]
    lines += ["", "## Moyennes et variabilité", "",
              "Moyenne ± écart-type d'échantillon (`ddof=1`) des exécutions terminées. "
              "Ces écarts décrivent les graines sur une partition fixe, pas un intervalle de confiance.", "",
              "| Modèle | Exécutions | Exactitude (%) | Macro-F1 (%) | Durée (s) |",
              "|---|---:|---:|---:|---:|"]
    for row in summary:
        cells = []
        for metric, scale in [("accuracy", 100), ("macro_f1", 100), ("training_seconds", 1)]:
            cells.append(f"{format_number(row[metric + '_mean'], scale)} ± {format_number(row[metric + '_std'], scale)}")
        lines.append(f"| {row['model']} | {row['n']}/{row['expected']} | " + " | ".join(cells) + " |")
    lines += ["", "La durée couvre les époques d'entraînement et de validation, l'écriture des historiques et "
              "des checkpoints. Elle exclut le téléchargement, l'initialisation et le test final.",
              "", "## Courbes", "", "![Courbes d'apprentissage](learning_curves.png)", "",
              "L'exactitude d'entraînement est mesurée au fil des mises à jour des poids ; "
              "celle de validation est mesurée en fin d'époque avec des poids fixes.", "",
              "## Matrices de confusion", "",
              "Chaque matrice correspond à une graine. Les prédictions répétées sur les mêmes images "
              "ne sont pas traitées comme des observations indépendantes.", ""]
    for row in rows:
        if row["status"] != "complete":
            continue
        run_dir = results_dir / f"{row['model']}-seed-{row['seed']}"
        filename = plot_confusion(run_dir, report_dir)
        lines += [f"![{run_dir.name}]({filename})", ""]
    if not protocol["smoke"]:
        lines += ["## Exemples d'erreurs", "",
                  "La graine d'entraînement 0 et la graine d'échantillonnage 2026 ont été fixées avant "
                  "les expériences. Jusqu'à 12 erreurs sont tirées sans remise par modèle. "
                  "Les numéros sont les indices du test officiel.", ""]
        for row in rows:
            if row["seed"] == 0 and row["status"] == "complete":
                run_dir = results_dir / f"{row['model']}-seed-0"
                filename = plot_errors(run_dir, report_dir)
                if filename:
                    lines += [f"![Erreurs {row['model']}]({filename})", ""]
        lines += ["Images : Fashion-MNIST, Zalando SE. Voir la notice de licence du dépôt.", ""]
    (report_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return report_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    print(generate_report(args.results_dir))


if __name__ == "__main__":
    main()
