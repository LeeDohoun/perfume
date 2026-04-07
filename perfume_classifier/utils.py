"""
utils.py
--------
학습 지원 유틸리티 모음.

- set_seed()         : 재현성 보장
- EarlyStopping      : val_loss 기반 조기 종료
- AverageMeter       : 배치 단위 손실·정확도 누적
- save_checkpoint()  : 체크포인트 저장
- load_checkpoint()  : 체크포인트 로드
- plot_history()     : 학습 곡선 저장
"""

import os
import random
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# 재현성
# ──────────────────────────────────────────────

def set_seed(seed: int = 42):
    """모든 난수 생성기 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] seed={seed} 고정 완료")


# ──────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────

class EarlyStopping:
    """
    Validation Loss가 개선되지 않으면 학습을 조기 종료합니다.

    Parameters
    ----------
    patience  : 개선 없이 허용할 epoch 수
    min_delta : 개선으로 인정할 최소 감소량
    mode      : 'min' (loss 감소) | 'max' (accuracy 증가)
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "min"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] 개선 없음 {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                print("[EarlyStopping] 조기 종료 트리거!")

        return self.should_stop

    def reset(self):
        self.counter   = 0
        self.best      = float("inf") if self.mode == "min" else float("-inf")
        self.should_stop = False


# ──────────────────────────────────────────────
# 배치 누적 미터
# ──────────────────────────────────────────────

class AverageMeter:
    """배치별 값(손실, 정확도 등)을 누적하여 평균을 계산합니다."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


# ──────────────────────────────────────────────
# 체크포인트
# ──────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    save_path: str,
    extra: Optional[Dict] = None,
):
    """모델 상태와 학습 메타 정보를 저장합니다."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "epoch":      epoch,
        "metric":     metric,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, save_path)
    print(f"[Checkpoint] 저장: {save_path}  (epoch={epoch}, metric={metric:.4f})")


def load_checkpoint(
    model: nn.Module,
    load_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    """저장된 체크포인트를 로드합니다."""
    ckpt = torch.load(load_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Checkpoint] 로드: {load_path}  (epoch={ckpt.get('epoch')}, metric={ckpt.get('metric'):.4f})")
    return ckpt


# ──────────────────────────────────────────────
# 학습 곡선 시각화
# ──────────────────────────────────────────────

def plot_history(
    history: Dict[str, list],
    save_dir: str,
    filename: str = "training_curves.png",
):
    """
    학습 곡선(loss, accuracy)을 저장합니다.

    history 딕셔너리 예시:
      {
        "train_loss": [...], "val_loss": [...],
        "train_acc":  [...], "val_acc":  [...],
      }
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss", marker="o", markersize=3)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"],   label="Val Loss",   marker="s", markersize=3)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="Train Acc", marker="o", markersize=3)
    if "val_acc" in history:
        axes[1].plot(history["val_acc"],   label="Val Acc",   marker="s", markersize=3)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] 학습 곡선 저장: {save_path}")


# ──────────────────────────────────────────────
# 장치 감지
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    """사용 가능한 최적 device를 반환합니다."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU 사용: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS 사용")
    else:
        device = torch.device("cpu")
        print("[Device] CPU 사용")
    return device
