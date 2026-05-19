"""
Two-stage CLIP image classifier for perfume Note classification.

This is the fair CLIP counterpart to the EfficientNet-B0 two-stage run:

Stage 1:
  - Freeze CLIP visual encoder.
  - Extract image embeddings.
  - Train only a small classifier head.

Stage 2:
  - Load the best Stage 1 head.
  - Unfreeze the last CLIP visual transformer block.
  - Fine-tune with a small visual learning rate.

The script deliberately does not use the `notes` column.
"""
import argparse
import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler


NOTE_CLASS_ORDER = [
    "Floral",
    "Woody",
    "Amber_Oriental",
    "Citrus",
    "Sweet",
    "Spicy",
    "Fresh",
]


class PerfumeImageDataset(Dataset):
    def __init__(self, csv_path: Path, label_to_idx: dict[str, int], transform, max_rows: int | None = None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df[self.df["label"].isin(label_to_idx)].reset_index(drop=True)
        if max_rows is not None and max_rows > 0:
            self.df = self.df.head(max_rows).reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.resolve_path(row["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"[Dataset] image load failed: {image_path} -> {exc}", flush=True)
            image = Image.new("RGB", (224, 224))

        label = self.label_to_idx[str(row["label"])]
        return self.transform(image), label

    @staticmethod
    def resolve_path(image_path: str) -> Path:
        normalized = str(image_path).replace("\\", os.sep)
        path = Path(normalized)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


class ClipHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ClipImageClassifier(nn.Module):
    def __init__(self, clip_model: nn.Module, head: ClipHead):
        super().__init__()
        self.clip_model = clip_model
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(images)
        features = nn.functional.normalize(features.float(), dim=-1)
        return self.head(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preferred: str) -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_labels(train_csv: Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    train_df = pd.read_csv(train_csv)
    unique_labels = set(train_df["label"].astype(str).unique())
    labels = [label for label in NOTE_CLASS_ORDER if label in unique_labels]
    labels.extend(label for label in sorted(unique_labels) if label not in labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return labels, label_to_idx, idx_to_label


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    sample_weights = np.array([1.0 / counts[label] for label in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def extract_features(clip_model, loader, device, split_name: str, log_every: int):
    clip_model.eval()
    features_all = []
    labels_all = []

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        features = clip_model.encode_image(images)
        features = nn.functional.normalize(features.float(), dim=-1)
        features_all.append(features.cpu())
        labels_all.append(labels)

        if log_every and (step == 1 or step % log_every == 0 or step == len(loader)):
            print(f"[Features:{split_name}] batch {step}/{len(loader)}", flush=True)

    return torch.cat(features_all, dim=0), torch.cat(labels_all, dim=0)


def load_or_extract_features(clip_model, dataset, loader, device, args, split_name: str):
    cache_dir = PROJECT_ROOT / args.feature_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_key = f"{args.model}_{args.pretrained}".replace("/", "_")
    cache_path = cache_dir / f"{model_key}_{args.train_split}_{split_name}.pt"

    if args.cache_features and cache_path.exists() and not args.rebuild_feature_cache:
        print(f"[Features:{split_name}] cache load: {cache_path}", flush=True)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload["features"], payload["labels"]

    features, labels = extract_features(clip_model, loader, device, split_name, args.log_every)
    if args.cache_features:
        torch.save({"features": features, "labels": labels}, cache_path)
        print(f"[Features:{split_name}] cache save: {cache_path}", flush=True)
    return features, labels


def train_feature_epoch(head, loader, criterion, optimizer, device, train: bool):
    head.train(train)
    loss_sum = 0.0
    y_true = []
    y_pred = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = head(features)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    return (
        loss_sum / len(loader.dataset),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


def train_image_epoch(model, loader, criterion, optimizer, device, train: bool, log_every: int):
    model.train(train)
    loss_sum = 0.0
    y_true = []
    y_pred = []
    mode = "train" if train else "val"

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

        if log_every and (step == 1 or step % log_every == 0 or step == len(loader)):
            running_loss = loss_sum / max(1, len(y_true))
            running_acc = accuracy_score(y_true, y_pred)
            running_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            print(
                f"[Stage2:{mode}] batch {step}/{len(loader)} "
                f"loss={running_loss:.4f} acc={running_acc:.4f} f1={running_f1:.4f}",
                flush=True,
            )

    return (
        loss_sum / len(loader.dataset),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


def freeze_clip(clip_model) -> None:
    for param in clip_model.parameters():
        param.requires_grad = False


def unfreeze_last_visual_blocks(clip_model, n_blocks: int) -> None:
    freeze_clip(clip_model)
    visual = clip_model.visual
    blocks = list(visual.transformer.resblocks)
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    for name in ["ln_post", "proj"]:
        module_or_param = getattr(visual, name, None)
        if module_or_param is None:
            continue
        if isinstance(module_or_param, nn.Parameter):
            module_or_param.requires_grad = True
        else:
            for param in module_or_param.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in clip_model.parameters())
    print(f"[Stage2] CLIP visual last {n_blocks} block(s) unfrozen | trainable={trainable:,}/{total:,}", flush=True)


def save_checkpoint(path: Path, model_state: dict, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["model_state"] = model_state
    torch.save(payload, path)
    print(f"[Checkpoint] saved: {path}", flush=True)


def run_stage1(clip_model, train_ds, val_ds, device, args, num_classes):
    print("\n" + "=" * 60, flush=True)
    print("[Stage 1] Frozen CLIP feature head training", flush=True)
    print("=" * 60, flush=True)

    freeze_clip(clip_model)
    train_loader = DataLoader(train_ds, batch_size=args.extract_batch_size, shuffle=False, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.extract_batch_size, shuffle=False, num_workers=args.workers)
    train_features, train_labels = load_or_extract_features(clip_model, train_ds, train_loader, device, args, args.train_split)
    val_features, val_labels = load_or_extract_features(clip_model, val_ds, val_loader, device, args, "val")

    train_sampler = make_weighted_sampler(train_labels.numpy())
    train_feat_ds = TensorDataset(train_features, train_labels)
    val_feat_ds = TensorDataset(val_features, val_labels)
    train_feat_loader = DataLoader(train_feat_ds, batch_size=args.head_batch_size, sampler=train_sampler)
    val_feat_loader = DataLoader(val_feat_ds, batch_size=args.head_batch_size, shuffle=False)

    head = ClipHead(train_features.shape[1], num_classes, args.hidden_dim, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_labels.numpy(), num_classes, device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.stage1_lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_path = PROJECT_ROOT / args.checkpoint_dir / "clip_stage1_best.pt"

    for epoch in range(1, args.stage1_epochs + 1):
        tr_loss, tr_acc, _ = train_feature_epoch(head, train_feat_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc, vl_f1 = train_feature_epoch(head, val_feat_loader, criterion, optimizer, device, train=False)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)
        print(
            f"[S1 E{epoch:02d}/{args.stage1_epochs}] "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val_loss={vl_loss:.4f} acc={vl_acc:.4f} f1={vl_f1:.4f}",
            flush=True,
        )
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            save_checkpoint(best_path, head.state_dict(), {"stage": 1, "epoch": epoch, "macro_f1": vl_f1})

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["model_state"])
    return head, history, best_f1


def run_stage2(clip_model, head, train_ds, val_ds, device, args, num_classes):
    print("\n" + "=" * 60, flush=True)
    print("[Stage 2] CLIP last visual block fine-tuning", flush=True)
    print("=" * 60, flush=True)

    unfreeze_last_visual_blocks(clip_model, args.unfreeze_last_blocks)
    model = ClipImageClassifier(clip_model, head).to(device)

    train_sampler = make_weighted_sampler(train_ds.df["label"].map(train_ds.label_to_idx).to_numpy())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.stage2_batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
    )
    val_loader = DataLoader(val_ds, batch_size=args.stage2_batch_size, shuffle=False, num_workers=args.workers)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_ds.df["label"].map(train_ds.label_to_idx).to_numpy(), num_classes, device),
        label_smoothing=args.label_smoothing,
    )
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    visual_params = [p for p in model.clip_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.stage2_head_lr},
            {"params": visual_params, "lr": args.stage2_visual_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage2_epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_path = PROJECT_ROOT / args.checkpoint_dir / "clip_stage2_best.pt"

    for epoch in range(1, args.stage2_epochs + 1):
        tr_loss, tr_acc, _ = train_image_epoch(
            model, train_loader, criterion, optimizer, device, train=True, log_every=args.log_every
        )
        vl_loss, vl_acc, vl_f1 = train_image_epoch(
            model, val_loader, criterion, optimizer, device, train=False, log_every=args.log_every
        )
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)
        print(
            f"[S2 E{epoch:02d}/{args.stage2_epochs}] "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val_loss={vl_loss:.4f} acc={vl_acc:.4f} f1={vl_f1:.4f}",
            flush=True,
        )
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            save_checkpoint(
                best_path,
                model.state_dict(),
                {"stage": 2, "epoch": epoch, "macro_f1": vl_f1},
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model, history, best_f1


@torch.no_grad()
def predict(model, loader, device, log_every: int):
    model.eval()
    y_true = []
    y_pred = []
    scores = []
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        logits = model(images).float()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        scores.append(logits.cpu().numpy())
        if log_every and (step == 1 or step % log_every == 0 or step == len(loader)):
            print(f"[Test] batch {step}/{len(loader)}", flush=True)
    return np.array(y_true), np.array(y_pred), np.concatenate(scores, axis=0)


def top_k_accuracy(scores: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    top_k = np.argsort(scores, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(labels, top_k)]))


def softmax_np(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def plot_confusion(y_true, y_pred, class_names, output_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for row in range(cm_norm.shape[0]):
        for col in range(cm_norm.shape[1]):
            value = cm_norm[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_title("CLIP Fine-tuned Normalized Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def plot_per_class_accuracy(y_true, y_pred, class_names, output_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    values = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, values, color="#4c78a8", edgecolor="white")
    ax.axhline(values.mean(), color="#d62728", linestyle="--", label=f"Mean: {values.mean():.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("CLIP Fine-tuned Per-Class Accuracy")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_accuracy.png", dpi=160)
    plt.close(fig)


def plot_metrics(metrics: dict, output_dir: Path) -> None:
    labels = ["Accuracy", "Macro F1", "Top-3 Accuracy"]
    values = [metrics["accuracy"], metrics["macro_f1"], metrics["top3_accuracy"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"], edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("CLIP Fine-tuned Test Metrics")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_bar.png", dpi=160)
    plt.close(fig)


def plot_training_curves(history, output_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[2].plot(epochs, history["val_f1"], label="val macro f1")
    axes[2].set_title("Macro F1")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def save_evaluation_summary(metrics, output_dir: Path) -> None:
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.75, 1.25])
    ax_text = fig.add_subplot(grid[0, 0])
    ax_text.axis("off")
    ax_text.set_title("CLIP Fine-tuned", fontsize=20, fontweight="bold", loc="left", pad=12)
    text = (
        "Evaluation dataset: data/All/test.csv\n"
        "Training data: data/All/train_aug.csv\n"
        "Input: image only, no notes\n\n"
        f"Accuracy        {metrics['accuracy']:.4f}\n"
        f"Macro F1        {metrics['macro_f1']:.4f}\n"
        f"Top-3 Accuracy  {metrics['top3_accuracy']:.4f}\n"
        f"Random Baseline {metrics['random_baseline']:.4f}"
    )
    ax_text.text(0, 0.86, text, va="top", ha="left", fontsize=14, family="monospace", linespacing=1.55)

    for ax, filename in [
        (fig.add_subplot(grid[0, 1]), "metrics_bar.png"),
        (fig.add_subplot(grid[1, 0]), "confusion_matrix.png"),
        (fig.add_subplot(grid[1, 1]), "per_class_accuracy.png"),
    ]:
        ax.imshow(mpimg.imread(output_dir / filename))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_summary.png", dpi=160)
    plt.close(fig)


def save_outputs(args, model, test_ds, test_preprocess, device, idx_to_label, class_names):
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    y_true, y_pred, scores = predict(model, test_loader, device, args.log_every)
    metrics = {
        "model": f"CLIP {args.model} ({args.pretrained}) two-stage fine-tune",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "top3_accuracy": top_k_accuracy(scores, y_true, 3),
        "random_baseline": 1.0 / len(class_names),
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": getattr(args, "best_stage2_epoch", args.stage2_epochs),
        "best_val_macro_f1": getattr(args, "best_val_macro_f1", None),
        "unfreeze_last_blocks": args.unfreeze_last_blocks,
        "notes_used": False,
    }
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    plot_confusion(y_true, y_pred, class_names, output_dir)
    plot_per_class_accuracy(y_true, y_pred, class_names, output_dir)
    plot_metrics(metrics, output_dir)
    save_evaluation_summary(metrics, output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        (
            f"Model        : {metrics['model']}\n"
            f"Data         : data/All\n"
            f"Train split  : {args.train_split}\n"
            f"Notes used   : False\n"
            f"Accuracy     : {metrics['accuracy']:.4f}\n"
            f"Macro F1     : {metrics['macro_f1']:.4f}\n"
            f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}\n"
            f"Random Baseline: {metrics['random_baseline']:.4f}\n\n"
            f"{report}"
        ),
        encoding="utf-8",
    )

    probs = softmax_np(scores)
    rows = []
    eval_df = test_ds.df.reset_index(drop=True)
    for idx, row in eval_df.iterrows():
        top3 = np.argsort(probs[idx])[-3:][::-1]
        rows.append({
            "image_path": row.get("image_path", ""),
            "name": row.get("name", ""),
            "brand": row.get("brand", ""),
            "actual": idx_to_label[int(y_true[idx])],
            "predicted": idx_to_label[int(y_pred[idx])],
            "confidence": float(probs[idx, int(y_pred[idx])]),
            "top3_predictions": " | ".join(idx_to_label[int(i)] for i in top3),
            "top3_correct": bool(int(y_true[idx]) in top3),
        })
    pd.DataFrame(rows).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    print(report, flush=True)
    print(f"[Final] metrics={metrics}", flush=True)
    print(f"[Final] outputs saved: {output_dir}", flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-stage CLIP image classifier.")
    parser.add_argument("--data-dir", default="data/All")
    parser.add_argument("--train-split", default="train_aug")
    parser.add_argument("--model", default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=3)
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-head-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-visual-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", default="results/clip_finetune")
    parser.add_argument("--checkpoint-dir", default="checkpoints/clip_finetune")
    parser.add_argument("--feature-cache-dir", default="results/clip_finetune/features")
    parser.add_argument("--eval-checkpoint", default="",
                        help="Train 없이 저장된 CLIP fine-tune checkpoint만 평가합니다.")
    parser.add_argument("--cache-features", action="store_true", default=True)
    parser.add_argument("--no-cache-features", dest="cache_features", action="store_false")
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    return parser.parse_args()


def load_eval_checkpoint(clip_model, checkpoint_path: Path, args, device, num_classes):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state"]

    if "head.net.1.weight" in state:
        hidden_dim, feature_dim = state["head.net.1.weight"].shape
        head = ClipHead(feature_dim, num_classes, int(hidden_dim), args.dropout)
        model = ClipImageClassifier(clip_model, head).to(device)
        model.load_state_dict(state)
    else:
        hidden_dim, feature_dim = state["net.1.weight"].shape
        head = ClipHead(feature_dim, num_classes, int(hidden_dim), args.dropout).to(device)
        head.load_state_dict(state)
        model = ClipImageClassifier(clip_model, head).to(device)

    args.best_stage2_epoch = checkpoint.get("epoch", args.stage2_epochs)
    args.best_val_macro_f1 = checkpoint.get("macro_f1")
    print(
        f"[Eval] checkpoint={checkpoint_path} "
        f"stage={checkpoint.get('stage')} epoch={checkpoint.get('epoch')} "
        f"val_macro_f1={checkpoint.get('macro_f1')}",
        flush=True,
    )
    return model


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cpu":
        print("[Warning] CPU run. Stage 2 fine-tuning can be slow.", flush=True)

    import open_clip

    data_dir = PROJECT_ROOT / args.data_dir
    train_csv = data_dir / f"{args.train_split}.csv"
    if not train_csv.exists():
        print(f"[Data] {train_csv} not found; fallback to train.csv", flush=True)
        train_csv = data_dir / "train.csv"
        args.train_split = "train"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    class_names, label_to_idx, idx_to_label = read_labels(data_dir / "train.csv")
    print(f"Classes: {class_names}", flush=True)
    print(f"Train CSV: {train_csv}", flush=True)

    cache_dir = PROJECT_ROOT / ".torch_cache" / "open_clip"
    clip_model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
        cache_dir=str(cache_dir),
    )

    max_train = args.max_train_samples or None
    max_eval = args.max_eval_samples or None
    train_ds_stage1 = PerfumeImageDataset(train_csv, label_to_idx, test_preprocess, max_rows=max_train)
    train_ds_stage2 = PerfumeImageDataset(train_csv, label_to_idx, train_preprocess, max_rows=max_train)
    val_ds = PerfumeImageDataset(val_csv, label_to_idx, test_preprocess, max_rows=max_eval)
    test_ds = PerfumeImageDataset(test_csv, label_to_idx, test_preprocess, max_rows=max_eval)
    print(f"Rows train/val/test: {len(train_ds_stage2)}/{len(val_ds)}/{len(test_ds)}", flush=True)

    if args.eval_checkpoint:
        checkpoint_path = PROJECT_ROOT / args.eval_checkpoint
        model = load_eval_checkpoint(clip_model, checkpoint_path, args, device, len(class_names))
        save_outputs(args, model, test_ds, test_preprocess, device, idx_to_label, class_names)
        return

    head, h1, _ = run_stage1(clip_model, train_ds_stage1, val_ds, device, args, len(class_names))
    model, h2, _ = run_stage2(clip_model, head, train_ds_stage2, val_ds, device, args, len(class_names))

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    full_history = {
        "train_loss": h1["train_loss"] + h2["train_loss"],
        "val_loss": h1["val_loss"] + h2["val_loss"],
        "train_acc": h1["train_acc"] + h2["train_acc"],
        "val_acc": h1["val_acc"] + h2["val_acc"],
        "val_f1": h1["val_f1"] + h2["val_f1"],
    }
    plot_training_curves(full_history, output_dir)
    save_outputs(args, model, test_ds, test_preprocess, device, idx_to_label, class_names)


if __name__ == "__main__":
    main()
