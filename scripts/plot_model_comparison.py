"""
Build comparison plots from the current EfficientNet-B0 and CLIP result files.
"""
import csv
import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def read_efficientnet_metrics(path: Path) -> dict:
    metrics = {
        "model": "EfficientNet-B0 + text",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        normalized = key.lower().replace("-", "").replace(" ", "_")
        if normalized == "accuracy":
            metrics["accuracy"] = float(value)
        elif normalized == "macro_f1":
            metrics["macro_f1"] = float(value)
        elif normalized == "top3_accuracy":
            metrics["top3_accuracy"] = float(value)
        elif normalized == "random_baseline":
            metrics["random_baseline"] = float(value)
    return metrics


def read_clip_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "model": "CLIP linear probe",
        "accuracy": float(payload["accuracy"]),
        "macro_f1": float(payload["macro_f1"]),
        "top3_accuracy": float(payload["top3_accuracy"]),
        "random_baseline": float(payload["random_baseline"]),
    }


def read_clip_finetune_metrics(path: Path, model_label: str = "CLIP two-stage fine-tune") -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "model": model_label,
        "accuracy": float(payload["accuracy"]),
        "macro_f1": float(payload["macro_f1"]),
        "top3_accuracy": float(payload["top3_accuracy"]),
        "random_baseline": float(payload["random_baseline"]),
    }


def save_single_metric_bar(metrics: dict, output_dir: Path) -> None:
    labels = ["Accuracy", "Macro F1", "Top-3 Accuracy"]
    keys = ["accuracy", "macro_f1", "top3_accuracy"]
    values = [metrics[key] for key in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"], edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(metrics["model"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_bar.png", dpi=160)
    plt.close(fig)


def save_comparison_plot(rows: list[dict], output_path: Path) -> None:
    metric_keys = ["accuracy", "macro_f1", "top3_accuracy"]
    metric_labels = ["Accuracy", "Macro F1", "Top-3 Accuracy"]
    model_labels = [row["model"] for row in rows]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    width = min(0.26, 0.8 / max(1, len(rows)))
    x_positions = range(len(metric_keys))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for model_idx, row in enumerate(rows):
        center_offset = (model_idx - (len(rows) - 1) / 2) * width
        offsets = [x + center_offset for x in x_positions]
        values = [row[key] for key in metric_keys]
        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=model_labels[model_idx],
            color=colors[model_idx % len(colors)],
            edgecolor="white",
        )
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)

    ax.set_xticks(list(x_positions), metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Test Metric Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_evaluation_summary(metrics: dict, output_dir: Path) -> None:
    confusion_path = output_dir / "confusion_matrix.png"
    per_class_path = output_dir / "per_class_accuracy.png"
    metric_bar_path = output_dir / "metrics_bar.png"

    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.75, 1.25])

    ax_text = fig.add_subplot(grid[0, 0])
    ax_text.axis("off")
    ax_text.set_title(metrics["model"], fontsize=20, fontweight="bold", loc="left", pad=12)
    summary = (
        "Evaluation dataset: data/All/test.csv\n"
        "Training data: data/All/train_aug.csv when available\n\n"
        f"Accuracy        {metrics['accuracy']:.4f}\n"
        f"Macro F1        {metrics['macro_f1']:.4f}\n"
        f"Top-3 Accuracy  {metrics['top3_accuracy']:.4f}\n"
        f"Random Baseline {metrics['random_baseline']:.4f}"
    )
    ax_text.text(
        0,
        0.86,
        summary,
        va="top",
        ha="left",
        fontsize=14,
        family="monospace",
        linespacing=1.55,
    )

    ax_metric = fig.add_subplot(grid[0, 1])
    ax_metric.imshow(mpimg.imread(metric_bar_path))
    ax_metric.axis("off")

    ax_confusion = fig.add_subplot(grid[1, 0])
    ax_confusion.imshow(mpimg.imread(confusion_path))
    ax_confusion.axis("off")

    ax_per_class = fig.add_subplot(grid[1, 1])
    ax_per_class.imshow(mpimg.imread(per_class_path))
    ax_per_class.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_summary.png", dpi=160)
    plt.close(fig)


def save_summary_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = ["model", "accuracy", "macro_f1", "top3_accuracy", "random_baseline"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def copy_if_source_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    efficientnet_dir = results_dir / "efficientnet_b0"
    clip_dir = results_dir / "clip_linear_probe"
    clip_finetune_dir = results_dir / "clip_finetune"
    clip_finetune_mps_dir = results_dir / "clip_finetune_mps"
    efficientnet_dir.mkdir(parents=True, exist_ok=True)

    efficientnet_metrics = read_efficientnet_metrics(results_dir / "test_metrics.txt")
    clip_metrics = read_clip_metrics(clip_dir / "metrics.json")
    rows = [efficientnet_metrics, clip_metrics]
    clip_finetune_metrics_path = clip_finetune_dir / "metrics.json"
    if clip_finetune_metrics_path.exists():
        rows.append(read_clip_finetune_metrics(clip_finetune_metrics_path, "CLIP two-stage fine-tune (CPU 3e)"))
    clip_finetune_mps_metrics_path = clip_finetune_mps_dir / "metrics.json"
    if clip_finetune_mps_metrics_path.exists():
        rows.append(read_clip_finetune_metrics(clip_finetune_mps_metrics_path, "CLIP two-stage fine-tune (MPS best)"))

    copy_if_source_exists(results_dir / "test_metrics.txt", efficientnet_dir / "classification_report.txt")
    copy_if_source_exists(results_dir / "test_confusion_matrix.png", efficientnet_dir / "confusion_matrix.png")
    copy_if_source_exists(results_dir / "test_per_class_accuracy.png", efficientnet_dir / "per_class_accuracy.png")
    save_single_metric_bar(efficientnet_metrics, efficientnet_dir)
    save_single_metric_bar(clip_metrics, clip_dir)
    save_evaluation_summary(efficientnet_metrics, efficientnet_dir)
    save_evaluation_summary(clip_metrics, clip_dir)
    save_comparison_plot(rows, results_dir / "model_metric_comparison.png")
    save_summary_csv(rows, results_dir / "model_metric_summary.csv")

    print(f"Saved EfficientNet artifacts to: {efficientnet_dir}")
    print(f"Saved comparison plot to: {results_dir / 'model_metric_comparison.png'}")
    print(f"Saved summary CSV to: {results_dir / 'model_metric_summary.csv'}")


if __name__ == "__main__":
    main()
