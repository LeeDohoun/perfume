"""
4-class note_classification_4class 폴더의 누락된 시각화 파일 생성.
Usage: python scripts/fill_4class_results.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from pathlib import Path

BASE = Path("results/note_classification_4class")

# ── helpers ──────────────────────────────────────────────────────────────────

def save_metrics_bar(metrics: dict, out_dir: Path, title: str):
    keys   = ["accuracy", "macro_f1", "top3_accuracy"]
    labels = ["Accuracy", "Macro F1", "Top-3 Acc"]
    values = [metrics.get(k, 0) for k in keys]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(metrics.get("random_baseline", 0), color="gray", linestyle="--",
               linewidth=1, label=f"Random ({metrics.get('random_baseline',0):.2f})")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "metrics_bar.png", dpi=150)
    plt.close()
    print(f"  saved metrics_bar.png → {out_dir}")


def save_evaluation_summary(out_dir: Path, title: str):
    candidates = ["confusion_matrix.png", "per_class_accuracy.png",
                  "metrics_bar.png", "training_curves.png"]
    imgs = [(fn, out_dir / fn) for fn in candidates if (out_dir / fn).exists()]
    if not imgs:
        print(f"  [skip] no images found in {out_dir}")
        return

    n = len(imgs)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 6))
    axes = np.array(axes).flatten()

    for ax, (fn, fp) in zip(axes, imgs):
        ax.imshow(mpimg.imread(fp))
        ax.axis("off")
        ax.set_title(fn.replace(".png", "").replace("_", " ").title(), fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "evaluation_summary.png", dpi=130)
    plt.close()
    print(f"  saved evaluation_summary.png → {out_dir}")


# ── 1. EfficientNet/with_text ─────────────────────────────────────────────────
print("[EfficientNet/with_text]")
eff_dir = BASE / "efficientnet" / "with_text"

metrics_eff = {
    "model": "EfficientNet-B0 + text (4-class)",
    "accuracy": 0.5177,
    "macro_f1": 0.4593,
    "top3_accuracy": 0.9355,
    "random_baseline": 0.25,
    "num_classes": 4,
}
(eff_dir / "metrics.json").write_text(
    json.dumps(metrics_eff, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("  saved metrics.json")

save_metrics_bar(metrics_eff, eff_dir, "EfficientNet-B0 + text (4-class)")
save_evaluation_summary(eff_dir, "EfficientNet-B0 + text  |  4-class")

# ── 2. CLIP/linear_probe ──────────────────────────────────────────────────────
print("[CLIP/linear_probe]")
lp_dir = BASE / "clip" / "linear_probe"
save_evaluation_summary(lp_dir, "CLIP Linear Probe  |  4-class")

# ── 3. CLIP/with_text ─────────────────────────────────────────────────────────
print("[CLIP/with_text]")
wt_dir = BASE / "clip" / "with_text"

with open(wt_dir / "metrics.json", encoding="utf-8") as f:
    metrics_wt = json.load(f)

save_metrics_bar(metrics_wt, wt_dir, "CLIP + text (4-class)")
save_evaluation_summary(wt_dir, "CLIP + text  |  4-class")

print("\nDone.")
