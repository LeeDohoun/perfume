"""
train.py
--------
2단계 Fine-tuning 학습 파이프라인.

Stage 1 (stage1_epochs)
  - Backbone Freeze, Head만 학습
  - lr = stage1_lr

Stage 2 (stage2_epochs)
  - Backbone 마지막 3블록부터 Unfreeze
  - 차등 lr: head = stage2_lr, backbone = stage2_lr * 0.1
  - CosineAnnealingLR + Early Stopping

사용법:
  python train.py
  또는 main.py에서 run_training() 호출
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import get_config
from dataset import make_dataloader
from model import build_model, PerfumeClassifier
from utils import (
    set_seed, EarlyStopping, AverageMeter,
    save_checkpoint, load_checkpoint, plot_history, get_device,
)

cfg = get_config()


# ──────────────────────────────────────────────
# CutMix 헬퍼
# ──────────────────────────────────────────────

def _cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    배치 전체에 CutMix를 적용합니다.

    Returns
    -------
    mixed_images, labels_a, labels_b, lam
      loss = lam * CE(logits, labels_a) + (1-lam) * CE(logits, labels_b)
    """
    lam = float(np.random.beta(alpha, alpha))
    B, _, H, W = images.shape
    rand_idx = torch.randperm(B, device=images.device)

    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)

    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]
    # 실제 잘린 비율로 lam 보정
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return images, labels, labels[rand_idx], lam


# ──────────────────────────────────────────────
# 단일 에포크 학습
# ──────────────────────────────────────────────

def train_one_epoch(
    model: PerfumeClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> Tuple[float, float]:
    """
    Returns
    -------
    avg_loss, accuracy (0~1)
    """
    model.train()
    loss_meter = AverageMeter("loss")
    correct = total = 0
    tc = cfg.train

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # CutMix: 설정된 확률로 배치에 적용
        use_cutmix = tc.cutmix_prob > 0 and np.random.random() < tc.cutmix_prob
        if use_cutmix:
            images, labels_a, labels_b, lam = _cutmix_batch(images, labels, tc.cutmix_alpha)

        with autocast(enabled=use_amp):
            logits = model(images)
            if use_cutmix:
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), n=images.size(0))
        preds    = logits.argmax(dim=1)
        ref      = labels_a if use_cutmix else labels
        correct += (preds == ref).sum().item()
        total   += labels.size(0)

    return loss_meter.avg, correct / total


# ──────────────────────────────────────────────
# 단일 에포크 검증
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: PerfumeClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    """
    Returns
    -------
    avg_loss, accuracy (0~1)
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        loss_meter.update(loss.item(), n=images.size(0))
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return loss_meter.avg, correct / total


# ──────────────────────────────────────────────
# Stage 1 : Head Only
# ──────────────────────────────────────────────

def run_stage1(
    model: PerfumeClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, list]:
    """Backbone을 Freeze하고 분류 Head만 학습합니다."""
    tc = cfg.train
    print("\n" + "=" * 50)
    print(f"[Stage 1] Head 학습 시작 | {tc.stage1_epochs} epochs | lr={tc.stage1_lr}")
    print("=" * 50)

    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tc.stage1_lr,
        weight_decay=tc.weight_decay,
    )
    scaler = GradScaler(enabled=tc.use_amp)

    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_loss = float("inf")
    os.makedirs(cfg.path.checkpoint_dir, exist_ok=True)

    for epoch in range(1, tc.stage1_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, tc.use_amp
        )
        vl_loss, vl_acc = validate(model, val_loader, criterion, device, tc.use_amp)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"[S1 E{epoch:02d}/{tc.stage1_epochs}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f} acc={vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            save_checkpoint(
                model, optimizer, epoch, vl_loss,
                save_path=os.path.join(cfg.path.checkpoint_dir, "stage1_best.pth"),
                extra={"stage": 1},
            )

    return history


# ──────────────────────────────────────────────
# Stage 2 : Gradual Unfreeze + Full Fine-tuning
# ──────────────────────────────────────────────

def run_stage2(
    model: PerfumeClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, list]:
    """Backbone 마지막 블록부터 Unfreeze하여 전체 Fine-tuning을 수행합니다."""
    tc = cfg.train
    print("\n" + "=" * 50)
    print(f"[Stage 2] Full Fine-tuning 시작 | {tc.stage2_epochs} epochs | lr={tc.stage2_lr}")
    print("=" * 50)

    # 마지막 3개 블록 Unfreeze + 이후 에포크마다 점진적 확장
    model.unfreeze_last_n_blocks(3)

    criterion = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
    param_groups = model.get_param_groups(tc.stage2_lr)
    optimizer = optim.AdamW(param_groups, weight_decay=tc.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc.t_max, eta_min=tc.eta_min
    )
    scaler = GradScaler(enabled=tc.use_amp)
    early_stop = EarlyStopping(patience=tc.patience, min_delta=tc.min_delta, mode="min")

    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_loss = float("inf")
    # 점진적 Unfreeze 스케줄: epoch 6, 11 에 블록 추가 해제
    unfreeze_schedule = {6: 5, 11: 9}   # epoch → 누적 블록 수

    for epoch in range(1, tc.stage2_epochs + 1):
        # 점진적 Unfreeze
        if epoch in unfreeze_schedule:
            model.unfreeze_last_n_blocks(unfreeze_schedule[epoch])

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, tc.use_amp
        )
        vl_loss, vl_acc = validate(model, val_loader, criterion, device, tc.use_amp)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[S2 E{epoch:02d}/{tc.stage2_epochs}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"lr={current_lr:.6f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            save_checkpoint(
                model, optimizer, epoch, vl_loss,
                save_path=os.path.join(cfg.path.checkpoint_dir, "stage2_best.pth"),
                extra={"stage": 2},
            )

        if early_stop(vl_loss):
            print(f"[Stage 2] Early Stopping at epoch {epoch}")
            break

    return history


# ──────────────────────────────────────────────
# 전체 학습 진입점
# ──────────────────────────────────────────────

def run_training():
    """Stage 1 → Stage 2 순서로 전체 학습을 실행합니다."""
    tc = cfg.train
    set_seed(tc.seed)
    device = get_device()

    # DataLoader
    train_loader = make_dataloader(
        csv_path=cfg.path.train_csv,
        split="train",
        image_root=cfg.path.image_root,
    )
    val_loader = make_dataloader(
        csv_path=cfg.path.val_csv,
        split="val",
        image_root=cfg.path.image_root,
    )

    # 모델
    model = build_model().to(device)

    # Stage 1
    h1 = run_stage1(model, train_loader, val_loader, device)

    # Stage 2: Stage 1 최적 가중치 로드
    best_s1 = os.path.join(cfg.path.checkpoint_dir, "stage1_best.pth")
    if os.path.exists(best_s1):
        load_checkpoint(model, best_s1, device=str(device))

    h2 = run_stage2(model, train_loader, val_loader, device)

    # 학습 곡선 저장
    full_history = {
        "train_loss": h1["train_loss"] + h2["train_loss"],
        "val_loss":   h1["val_loss"]   + h2["val_loss"],
        "train_acc":  h1["train_acc"]  + h2["train_acc"],
        "val_acc":    h1["val_acc"]    + h2["val_acc"],
    }
    plot_history(full_history, save_dir=cfg.path.result_dir, filename="training_curves.png")

    print("\n[Training] 학습 완료!")
    print(f"  최적 체크포인트: {cfg.path.checkpoint_dir}/stage2_best.pth")
    return model


# ──────────────────────────────────────────────
# 직접 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    run_training()
