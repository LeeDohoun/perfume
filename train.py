"""
향수 이미지 분류 학습 스크립트

사용법:
  python train.py --task note      # Note 계열 분류 (6클래스)
  python train.py --task brand     # Brand 분류 (35클래스)

학습 전략 (model_recommendation.md 기반):
  - EfficientNet-B0 Transfer Learning
  - 2단계 Fine-tuning: Head만 → 전체 Unfreeze
  - WeightedRandomSampler (클래스 불균형 보정)
  - Data Augmentation (RandomCrop, Flip, Rotation, ColorJitter)
  - CosineAnnealingLR + Early Stopping
"""
import os
import argparse
import time
import copy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, ".torch_cache"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

TARGET_ACCURACY = {
    "note": (0.35, 0.50),
    "brand": (0.40, 0.65),
}


# ──────────────────────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────────────────────
class PerfumeDataset(Dataset):
    """CSV(image_path, label) 기반 이미지 데이터셋"""

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # 라벨 → 숫자 매핑
        labels_sorted = sorted(self.df["label"].unique())
        self.label2idx = {lbl: i for i, lbl in enumerate(labels_sorted)}
        self.idx2label = {i: lbl for lbl, i in self.label2idx.items()}
        self.num_classes = len(labels_sorted)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.normpath(str(row["image_path"]).replace("\\", os.sep))
        img = Image.open(image_path).convert("RGB")
        label = self.label2idx[row["label"]]

        if self.transform:
            img = self.transform(img)

        return img, label


# ──────────────────────────────────────────────────────────────
# 2. Transforms (데이터 증강)
# ──────────────────────────────────────────────────────────────
def get_transforms():
    """Train: 증강 적용 / Val·Test: 고정 변환"""

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


# ──────────────────────────────────────────────────────────────
# 3. 모델 생성
# ──────────────────────────────────────────────────────────────
def build_model(num_classes, freeze_backbone=True):
    """EfficientNet-B0 + 커스텀 분류 Head"""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Backbone freeze
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # 분류 Head 교체
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


def unfreeze_backbone(model, unfreeze_from=-3):
    """
    Backbone 마지막 N개 블록 Unfreeze
    EfficientNet-B0: features[0]~features[8] (총 9개 블록)
    unfreeze_from=-3 → features[6], [7], [8] Unfreeze
    """
    # 전체 Unfreeze
    for param in model.features[unfreeze_from:].parameters():
        param.requires_grad = True

    # Unfreeze된 파라미터 수 출력
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Unfreeze 완료: {trainable:,} / {total:,} 파라미터 학습 가능 "
          f"({trainable / total * 100:.1f}%)")


# ──────────────────────────────────────────────────────────────
# 4. WeightedRandomSampler (클래스 불균형 보정)
# ──────────────────────────────────────────────────────────────
def make_weighted_sampler(dataset):
    """소수 클래스 오버샘플링을 위한 가중 샘플러"""
    labels = [dataset.label2idx[row["label"]] for _, row in dataset.df.iterrows()]
    class_counts = np.bincount(labels, minlength=dataset.num_classes)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))


# ──────────────────────────────────────────────────────────────
# 5. 학습 루프
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ──────────────────────────────────────────────────────────────
# 6. 학습 실행
# ──────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, device, args, stage_name=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"\n{'=' * 60}")
    print(f"  {stage_name}")
    print(f"  Epochs: {args.epochs} | LR: {args.lr} | Patience: {args.patience}")
    print(f"{'=' * 60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch [{epoch:3d}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
            f"LR: {lr_now:.2e}  Time: {elapsed:.1f}s"
        )

        # Best 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"    * Best Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= args.patience:
            print(f"  Early Stopping at epoch {epoch} (patience={args.patience})")
            break

    model.load_state_dict(best_model_wts)
    return model, best_val_acc


# ──────────────────────────────────────────────────────────────
# 7. 테스트 평가
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def test_evaluate(model, loader, device, idx2label):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    # 라벨 이름 변환
    target_names = [idx2label[i] for i in range(len(idx2label))]

    print("\n" + "=" * 60)
    print("  Test 결과")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    test_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    return all_preds, all_labels, test_acc


def print_target_status(task, accuracy):
    """목표 정확도 범위와 현재 결과 비교"""
    low, high = TARGET_ACCURACY[task]
    print("\n" + "-" * 60)
    print(f"  Target Accuracy ({task}, EfficientNet-B0): {low * 100:.0f}% ~ {high * 100:.0f}%")
    if accuracy < low:
        print(f"  Result: {accuracy * 100:.2f}%  -> 목표 미달")
    elif accuracy <= high:
        print(f"  Result: {accuracy * 100:.2f}%  -> 목표 범위 도달")
    else:
        print(f"  Result: {accuracy * 100:.2f}%  -> 목표 상한 초과")
    print("-" * 60)


# ──────────────────────────────────────────────────────────────
# 8. Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="향수 이미지 분류 학습")
    parser.add_argument("--task", type=str, required=True,
                        choices=["note", "brand"],
                        help="분류 태스크: note(향 계열) / brand(브랜드)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None,
                        help="각 단계별 epoch 수 (기본: 1단계 10, 2단계 30)")
    parser.add_argument("--lr", type=float, default=None,
                        help="학습률 (기본: 1단계 1e-3, 2단계 1e-4)")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early Stopping patience")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (Windows: 0 권장)")
    args = parser.parse_args()

    # ── 경로 설정 ──
    data_dir = os.path.join("data", args.task)
    save_dir = os.path.join("checkpoints", args.task)
    os.makedirs(save_dir, exist_ok=True)

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")
    if device.type == "cpu":
        print("  [!] CPU mode: GPU not available. Training may be slow.")

    # ── 데이터 로드 ──
    train_tf, val_tf = get_transforms()

    train_ds = PerfumeDataset(os.path.join(data_dir, "train.csv"), train_tf)
    val_ds = PerfumeDataset(os.path.join(data_dir, "val.csv"), val_tf)
    test_ds = PerfumeDataset(os.path.join(data_dir, "test.csv"), val_tf)

    num_classes = train_ds.num_classes
    print(f"\nTask: {args.task.upper()}")
    print(f"Classes: {num_classes}  ({', '.join(train_ds.label2idx.keys())})")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    target_low, target_high = TARGET_ACCURACY[args.task]
    print(f"Target Accuracy (EfficientNet-B0): {target_low * 100:.0f}% ~ {target_high * 100:.0f}%")

    # WeightedRandomSampler
    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    # ── 모델 생성 ──
    model = build_model(num_classes, freeze_backbone=True)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: EfficientNet-B0")
    print(f"  파라미터: {total:,} (학습: {trainable:,})")

    # ════════════════════════════════════════════
    # 1단계: Head만 학습 (Backbone Freeze)
    # ════════════════════════════════════════════
    args_s1 = copy.copy(args)
    args_s1.epochs = args.epochs or 10
    args_s1.lr = args.lr or 1e-3

    model, best_acc_s1 = train(
        model, train_loader, val_loader, device, args_s1,
        stage_name="Stage 1: Head Only (Backbone Frozen)"
    )

    # 1단계 체크포인트 저장
    torch.save(model.state_dict(), os.path.join(save_dir, "stage1_best.pth"))
    print(f"\n  Stage 1 Best Val Acc: {best_acc_s1:.4f}")

    # ════════════════════════════════════════════
    # 2단계: Backbone 일부 Unfreeze + Fine-tuning
    # ════════════════════════════════════════════
    unfreeze_backbone(model, unfreeze_from=-3)

    args_s2 = copy.copy(args)
    args_s2.epochs = args.epochs or 30
    args_s2.lr = args.lr or 1e-4  # 작은 LR

    model, best_acc_s2 = train(
        model, train_loader, val_loader, device, args_s2,
        stage_name="Stage 2: Fine-tuning (Last 3 Blocks Unfrozen)"
    )

    # 최종 체크포인트 저장
    final_path = os.path.join(save_dir, "best_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "label2idx": train_ds.label2idx,
        "idx2label": train_ds.idx2label,
        "task": args.task,
        "val_acc": best_acc_s2,
    }, final_path)
    print(f"\n  Stage 2 Best Val Acc: {best_acc_s2:.4f}")
    print(f"  모델 저장: {final_path}")

    # ════════════════════════════════════════════
    # 테스트 평가
    # ════════════════════════════════════════════
    _, _, test_acc = test_evaluate(model, test_loader, device, train_ds.idx2label)
    print_target_status(args.task, test_acc)

    print("\n" + "=" * 60)
    print("  학습 완료!")
    print(f"  Task: {args.task.upper()}")
    print(f"  Stage 1 Val Acc: {best_acc_s1:.4f}")
    print(f"  Stage 2 Val Acc: {best_acc_s2:.4f}")
    print(f"  모델 저장: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
