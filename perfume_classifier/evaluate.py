"""
evaluate.py
-----------
테스트셋 평가 파이프라인.

평가 척도:
  - Accuracy
  - Macro F1-Score
  - Confusion Matrix (시각화 + 저장)
  - Top-3 Accuracy

사용법:
  python evaluate.py
  또는 main.py에서 run_evaluation() 호출
"""

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.cuda.amp import autocast

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import get_config
from dataset import make_dataloader
from model import build_model, PerfumeClassifier
from utils import load_checkpoint, get_device

cfg = get_config()


# ──────────────────────────────────────────────
# 추론 (all_preds, all_labels, all_probs)
# ──────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: PerfumeClassifier,
    loader,
    device: torch.device,
    use_amp: bool = True,
):
    """
    전체 데이터셋에 대해 추론을 수행합니다.

    Returns
    -------
    all_labels : np.ndarray  (N,)
    all_preds  : np.ndarray  (N,)
    all_probs  : np.ndarray  (N, num_classes)  — softmax 확률
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        if cfg.text.use_text:
            images, text_ids, labels = batch
            text_ids = text_ids.to(device)
        else:
            images, labels = batch
            text_ids = None
        images = images.to(device)

        with autocast(enabled=use_amp):
            logits = model(images, text_ids) if cfg.text.use_text else model(images)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs, axis=0),
    )


# ──────────────────────────────────────────────
# Top-k Accuracy
# ──────────────────────────────────────────────

def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """상위 k개 예측 안에 정답이 포함된 비율을 반환합니다."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]   # (N, k) — 내림차순 상위 k
    correct = np.array([labels[i] in top_k_preds[i] for i in range(len(labels))])
    return float(correct.mean())


# ──────────────────────────────────────────────
# Confusion Matrix 시각화
# ──────────────────────────────────────────────

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list,
    save_dir: str,
    filename: str = "confusion_matrix.png",
    normalize: bool = True,
):
    """Confusion Matrix를 저장합니다."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(labels, preds)

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Confusion Matrix (Normalized)"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Evaluate] Confusion Matrix 저장: {save_path}")


# ──────────────────────────────────────────────
# 클래스별 Accuracy 시각화
# ──────────────────────────────────────────────

def plot_per_class_accuracy(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list,
    save_dir: str,
    filename: str = "per_class_accuracy.png",
):
    """클래스별 정확도 막대 그래프를 저장합니다."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(class_names, per_class_acc, color="steelblue", edgecolor="white")
    ax.axhline(y=per_class_acc.mean(), color="red", linestyle="--", label=f"Mean: {per_class_acc.mean():.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.legend()

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Evaluate] Per-Class Accuracy 저장: {save_path}")


# ──────────────────────────────────────────────
# 전체 평가 진입점
# ──────────────────────────────────────────────

def run_evaluation(
    checkpoint_path: Optional[str] = None,
    split: str = "test",
):
    """
    Parameters
    ----------
    checkpoint_path : 사용할 .pth 파일 경로.
                      None이면 stage2_best.pth 자동 탐색.
    split           : 평가할 분할. 'test' | 'val'
    """
    device = get_device()
    tc = cfg.train

    # ── 모델 로드 ────────────────────────────────
    model = build_model().to(device)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.path.checkpoint_dir, "stage2_best.pth")
    load_checkpoint(model, checkpoint_path, device=str(device))

    # ── DataLoader ───────────────────────────────
    csv_map = {"test": cfg.path.test_csv, "val": cfg.path.val_csv}
    loader = make_dataloader(
        csv_path=csv_map[split],
        split=split,
        image_root=cfg.path.image_root,
    )

    # ── 추론 ─────────────────────────────────────
    print(f"\n[Evaluate] '{split}' 데이터셋 추론 중...")
    all_labels, all_preds, all_probs = run_inference(model, loader, device, use_amp=tc.use_amp)

    # ── 지표 계산 ────────────────────────────────
    class_names = cfg.cls.note_classes
    acc     = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    top3_acc = top_k_accuracy(all_probs, all_labels, k=3)
    random_baseline = 1.0 / len(class_names)

    print("\n" + "=" * 55)
    print(f"  Accuracy      : {acc:.4f}  (random={random_baseline:.4f})")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Top-3 Accuracy: {top3_acc:.4f}")
    print("=" * 55)
    print("\n[Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ── 시각화 저장 ──────────────────────────────
    os.makedirs(cfg.path.result_dir, exist_ok=True)

    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_dir=cfg.path.result_dir,
        filename=f"{split}_confusion_matrix.png",
        normalize=True,
    )
    plot_per_class_accuracy(
        all_labels, all_preds, class_names,
        save_dir=cfg.path.result_dir,
        filename=f"{split}_per_class_accuracy.png",
    )

    # ── 결과 요약 텍스트 저장 ─────────────────────
    summary_path = os.path.join(cfg.path.result_dir, f"{split}_metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Split         : {split}\n")
        f.write(f"Checkpoint    : {checkpoint_path}\n")
        f.write(f"Accuracy      : {acc:.4f}\n")
        f.write(f"Macro F1      : {macro_f1:.4f}\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.4f}\n")
        f.write(f"Random Baseline: {random_baseline:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"[Evaluate] 결과 저장: {summary_path}")

    return {
        "accuracy":     acc,
        "macro_f1":     macro_f1,
        "top3_accuracy": top3_acc,
    }


# ──────────────────────────────────────────────
# 직접 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation(split="test")
