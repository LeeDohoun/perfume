import argparse
import json
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: transformers. Install dependencies with "
        "`pip install -r data_classifier/requirements.txt`."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_DIR


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_csv(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    for base in (DATA_ROOT, PROJECT_DIR, SCRIPT_DIR, Path.cwd()):
        resolved = base / candidate
        if resolved.exists():
            return resolved

    return DATA_ROOT / candidate


def resolve_image_path(image_path: str, csv_dir: Path) -> Path:
    normalized_path = str(image_path).replace("\\", "/")
    raw_path = Path(normalized_path)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    candidates = [
        DATA_ROOT / raw_path,
        PROJECT_DIR / raw_path,
        csv_dir / raw_path,
        Path.cwd() / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return DATA_ROOT / raw_path


def maybe_copy_data_to_local(args) -> None:
    global DATA_ROOT

    should_auto_copy = args.auto_copy_data_to_local and str(PROJECT_DIR).startswith("/content/drive/")
    if not args.copy_data_to_local and not should_auto_copy:
        return

    local_root = Path(args.local_data_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    for folder_name in ("data", "perfume_images"):
        src = PROJECT_DIR / folder_name
        dst = local_root / folder_name
        if not src.exists():
            continue
        print(f"Copying {src} -> {dst}", flush=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    DATA_ROOT = local_root
    print(f"Using local data root: {DATA_ROOT}", flush=True)


def build_text(row: pd.Series) -> str:
    name = str(row.get("name", "") or "").strip()
    brand = str(row.get("brand", "") or "").strip()
    notes = str(row.get("notes", "") or "").strip()

    parts = []
    if name:
        parts.append(f"perfume name: {name}")
    if brand:
        parts.append(f"brand: {brand}")
    if notes:
        parts.append(f"fragrance notes: {notes}")

    if not parts:
        return "a perfume product"
    return ". ".join(parts)


class PerfumeClipDataset(Dataset):
    def __init__(self, csv_path: Path, label_to_idx=None):
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.dropna(subset=["image_path", "label"]).reset_index(drop=True)

        if label_to_idx is None:
            labels = sorted(self.df["label"].astype(str).unique().tolist())
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = dict(label_to_idx)

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.df["label"] = self.df["label"].astype(str)
        self.df = self.df[self.df["label"].isin(self.label_to_idx)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row["image_path"], self.csv_dir)
        image = Image.open(image_path).convert("RGB")
        text = build_text(row)
        label = self.label_to_idx[row["label"]]
        name = str(row.get("name", image_path.name))
        return image, text, label, name


def collate_fn(batch):
    images, texts, labels, names = zip(*batch)
    return list(images), list(texts), torch.tensor(labels, dtype=torch.long), list(names)


class ClipFusionClassifier(nn.Module):
    def __init__(self, clip_model_name: str, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        projection_dim = self.clip.config.projection_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_dim * 2),
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def freeze_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    def unfreeze_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = True

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        features = torch.cat([image_features, text_features], dim=-1)
        return self.classifier(features)


class FeatureClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.classifier(features)


class FeatureDataset(Dataset):
    def __init__(self, features, labels, names):
        self.features = features.float()
        self.labels = labels.long()
        self.names = list(names)

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.names[idx]


def build_datasets(args):
    train_csv = resolve_csv(args.train_csv)
    val_csv = resolve_csv(args.val_csv)
    test_csv = resolve_csv(args.test_csv)

    train_ds = PerfumeClipDataset(train_csv)
    val_ds = PerfumeClipDataset(val_csv, label_to_idx=train_ds.label_to_idx)
    test_ds = PerfumeClipDataset(test_csv, label_to_idx=train_ds.label_to_idx)
    return train_ds, val_ds, test_ds


def build_loaders(args):
    train_ds, val_ds, test_ds = build_datasets(args)
    label_counts = train_ds.df["label"].value_counts()
    sample_weights = train_ds.df["label"].map(lambda label: 1.0 / label_counts[label]).to_numpy(copy=True)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_extraction_loaders(train_ds, val_ds, test_ds, args):
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def safe_cache_name(clip_model: str, split_name: str) -> str:
    model_name = clip_model.replace("/", "__").replace("\\", "__")
    return f"{model_name}_{split_name}_features.pt"


def extract_or_load_features(split_name, dataset, loader, clip_model, processor, args, device):
    cache_root = Path(args.embedding_cache_dir)
    if not cache_root.is_absolute():
        cache_root = PROJECT_DIR / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / safe_cache_name(args.clip_model, split_name)

    if cache_path.exists() and not args.rebuild_embedding_cache:
        print(f"Loading cached {split_name} embeddings: {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"Extracting {split_name} CLIP embeddings -> {cache_path}", flush=True)
    clip_model.eval()
    features = []
    labels_all = []
    names_all = []
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (images, texts, labels, names) in enumerate(loader, start=1):
            encoded, labels = prepare_batch(processor, images, texts, labels, device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = clip_model(
                    pixel_values=encoded["pixel_values"],
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_dict=True,
                )
                image_features = nn.functional.normalize(outputs.image_embeds, dim=-1)
                text_features = nn.functional.normalize(outputs.text_embeds, dim=-1)
                batch_features = torch.cat([image_features, text_features], dim=-1)

            features.append(batch_features.detach().cpu())
            labels_all.append(labels.detach().cpu())
            names_all.extend(names)

            if args.log_every > 0 and (
                batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == total_batches
            ):
                print(f"  cache {split_name} batch {batch_idx}/{total_batches}", flush=True)

    payload = {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels_all, dim=0),
        "names": names_all,
        "label_to_idx": dataset.label_to_idx,
        "idx_to_label": dataset.idx_to_label,
        "clip_model": args.clip_model,
    }
    torch.save(payload, cache_path)
    return payload


def build_feature_loaders(train_payload, val_payload, test_payload, args):
    train_ds = FeatureDataset(train_payload["features"], train_payload["labels"], train_payload["names"])
    val_ds = FeatureDataset(val_payload["features"], val_payload["labels"], val_payload["names"])
    test_ds = FeatureDataset(test_payload["features"], test_payload["labels"], test_payload["names"])

    labels_np = train_payload["labels"].numpy()
    counts = np.bincount(labels_np)
    sample_weights = np.array([1.0 / counts[label] for label in labels_np], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def prepare_batch(processor, images, texts, labels, device):
    encoded = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
    labels = labels.to(device, non_blocking=True)
    return encoded, labels


def top_k_accuracy_from_logits(logits, labels, k=3):
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).any(dim=1)
    return correct.detach().cpu().numpy().tolist()


def top_k_accuracy_from_probs(probs, y_true, k=3):
    if len(y_true) == 0:
        return 0.0
    k = min(k, probs.shape[1])
    topk = np.argsort(probs, axis=1)[:, -k:]
    correct = [int(actual) in row for actual, row in zip(y_true, topk)]
    return float(np.mean(correct))


def run_epoch(model, processor, loader, criterion, optimizer, scaler, device, train=True, log_every=50):
    model.train(train)
    total_loss = 0.0
    y_true = []
    y_pred = []
    top3_correct = []

    mode = "train" if train else "val"
    total_batches = len(loader)

    for batch_idx, (images, texts, labels, _) in enumerate(loader, start=1):
        encoded, labels = prepare_batch(processor, images, texts, labels, device)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(
                    pixel_values=encoded["pixel_values"],
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        top3_correct.extend(top_k_accuracy_from_logits(logits, labels, k=3))

        if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
            running_loss = total_loss / max(1, len(y_true))
            running_acc = accuracy_score(y_true, y_pred)
            running_top3_acc = float(np.mean(top3_correct))
            print(
                f"  {mode} batch {batch_idx}/{total_batches} "
                f"loss={running_loss:.4f} acc={running_acc:.4f} top3_acc={running_top3_acc:.4f}",
                flush=True,
            )

    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), float(np.mean(top3_correct))


def run_feature_epoch(model, loader, criterion, optimizer, scaler, device, train=True, log_every=50):
    model.train(train)
    total_loss = 0.0
    y_true = []
    y_pred = []
    top3_correct = []

    mode = "train" if train else "val"
    total_batches = len(loader)

    for batch_idx, (features, labels, _) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(features)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        top3_correct.extend(top_k_accuracy_from_logits(logits, labels, k=3))

        if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
            running_loss = total_loss / max(1, len(y_true))
            running_acc = accuracy_score(y_true, y_pred)
            running_top3_acc = float(np.mean(top3_correct))
            print(
                f"  {mode} batch {batch_idx}/{total_batches} "
                f"loss={running_loss:.4f} acc={running_acc:.4f} top3_acc={running_top3_acc:.4f}",
                flush=True,
            )

    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), float(np.mean(top3_correct))


def predict_all(model, processor, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    probs = []
    names = []

    with torch.no_grad():
        for images, texts, labels, batch_names in loader:
            encoded, labels = prepare_batch(processor, images, texts, labels, device)
            logits = model(
                pixel_values=encoded["pixel_values"],
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            batch_probs = torch.softmax(logits, dim=1)
            preds = batch_probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            probs.extend(batch_probs.cpu().numpy().tolist())
            names.extend(batch_names)

    return np.array(y_true), np.array(y_pred), np.array(probs), names


def predict_feature_all(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    probs = []
    names = []

    with torch.no_grad():
        for features, labels, batch_names in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            batch_probs = torch.softmax(logits, dim=1)
            preds = batch_probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            probs.extend(batch_probs.cpu().numpy().tolist())
            names.extend(batch_names)

    return np.array(y_true), np.array(y_pred), np.array(probs), names


def save_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    axes[2].plot(epochs, history["train_top3_acc"], label="train")
    axes[2].plot(epochs, history["val_top3_acc"], label="val")
    axes[2].set_title("Top-3 Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close()


def save_confusion_matrix(y_true, y_pred, idx_to_label, output_dir):
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    for row in range(cm_norm.shape[0]):
        for col in range(cm_norm.shape[1]):
            value = cm_norm[row, col]
            color = "white" if value > 0.5 else "black"
            plt.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def class_weights_from_dataset(dataset, device):
    counts = dataset.df["label"].value_counts()
    total = counts.sum()
    weights = []
    for idx in range(len(dataset.idx_to_label)):
        label = dataset.idx_to_label[idx]
        weights.append(total / (len(counts) * counts[label]))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def class_weights_from_labels(labels, num_classes, device):
    labels_np = labels.numpy()
    counts = np.bincount(labels_np, minlength=num_classes)
    total = counts.sum()
    weights = [total / (num_classes * max(1, count)) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def serializable_args(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def train_with_embedding_cache(args, output_dir, device):
    train_ds, val_ds, test_ds = build_datasets(args)
    print(f"Classes: {train_ds.label_to_idx}", flush=True)
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}", flush=True)
    train_extract_loader, val_extract_loader, test_extract_loader = build_extraction_loaders(
        train_ds, val_ds, test_ds, args
    )

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    for param in clip_model.parameters():
        param.requires_grad = False

    train_payload = extract_or_load_features(
        "train", train_ds, train_extract_loader, clip_model, processor, args, device
    )
    val_payload = extract_or_load_features("val", val_ds, val_extract_loader, clip_model, processor, args, device)
    test_payload = extract_or_load_features("test", test_ds, test_extract_loader, clip_model, processor, args, device)

    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_feat_ds, val_feat_ds, test_feat_ds, train_loader, val_loader, test_loader = build_feature_loaders(
        train_payload, val_payload, test_payload, args
    )

    num_classes = len(train_payload["label_to_idx"])
    feature_dim = int(train_payload["features"].shape[1])
    model = FeatureClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    weights = (
        class_weights_from_labels(train_payload["labels"], num_classes, device)
        if args.weighted_loss
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = 0.0
    best_epoch = 0
    patience_count = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_top3_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top3_acc": [],
    }

    print("Training classifier from cached CLIP embeddings.", flush=True)
    print(f"Feature dim: {feature_dim}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_top3_acc = run_feature_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=True,
            log_every=args.log_every,
        )
        val_loss, val_acc, val_top3_acc = run_feature_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=False,
            log_every=args.log_every,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_top3_acc"].append(train_top3_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top3_acc"].append(val_top3_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_top3_acc={train_top3_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top3_acc={val_top3_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_idx": train_payload["label_to_idx"],
                    "idx_to_label": train_payload["idx_to_label"],
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                    "feature_dim": feature_dim,
                    "args": serializable_args(args),
                },
                output_dir / "best_clip_classifier.pt",
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    checkpoint = torch.load(output_dir / "best_clip_classifier.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    y_true, y_pred, probs, names = predict_feature_all(model, test_loader, device)
    test_acc = accuracy_score(y_true, y_pred)
    test_top3_acc = top_k_accuracy_from_probs(probs, y_true, k=3)
    idx_to_label = train_payload["idx_to_label"]
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

    save_training_curves(history, output_dir)
    save_confusion_matrix(y_true, y_pred, idx_to_label, output_dir)

    pred_rows = []
    for actual, pred, prob, name in zip(y_true, y_pred, probs, names):
        top3_indices = np.argsort(prob)[-3:][::-1]
        pred_rows.append(
            {
                "name": name,
                "actual": idx_to_label[int(actual)],
                "predicted": idx_to_label[int(pred)],
                "confidence": float(np.max(prob)),
                "top3_predictions": " | ".join(idx_to_label[int(idx)] for idx in top3_indices),
                "top3_correct": bool(int(actual) in top3_indices),
            }
        )

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    report_text = (
        f"Test accuracy: {test_acc:.4f}\n"
        f"Test top-3 accuracy: {test_top3_acc:.4f}\n\n"
        f"{report}"
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "label_map.json").write_text(
        json.dumps(train_payload["label_to_idx"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "test_top3_acc": test_top3_acc,
                "output_dir": str(output_dir),
                "used_embedding_cache": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTest accuracy:", f"{test_acc:.4f}", flush=True)
    print("Test top-3 accuracy:", f"{test_top3_acc:.4f}", flush=True)
    print(report)
    print(f"Saved outputs to: {output_dir}", flush=True)


def train(args):
    seed_everything(args.seed)
    maybe_copy_data_to_local(args)
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_DIR / output_root
    output_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cache_embeddings and args.freeze_clip:
        print(f"Device: {device}")
        print(f"CLIP model: {args.clip_model}")
        print(f"Output: {output_dir}")
        train_with_embedding_cache(args, output_dir, device)
        return

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(args)

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = ClipFusionClassifier(
        clip_model_name=args.clip_model,
        num_classes=len(train_ds.label_to_idx),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    if args.freeze_clip:
        model.freeze_clip()
    model.to(device)

    weights = class_weights_from_dataset(train_ds, device) if args.weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = 0.0
    best_epoch = 0
    patience_count = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_top3_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top3_acc": [],
    }

    print(f"Device: {device}")
    print(f"CLIP model: {args.clip_model}")
    print(f"Classes: {train_ds.label_to_idx}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Output: {output_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_top3_acc = run_epoch(
            model,
            processor,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=True,
            log_every=args.log_every,
        )
        val_loss, val_acc, val_top3_acc = run_epoch(
            model,
            processor,
            val_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=False,
            log_every=args.log_every,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_top3_acc"].append(train_top3_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top3_acc"].append(val_top3_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_top3_acc={train_top3_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top3_acc={val_top3_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_idx": train_ds.label_to_idx,
                    "idx_to_label": train_ds.idx_to_label,
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                    "args": serializable_args(args),
                },
                output_dir / "best_clip_classifier.pt",
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    checkpoint = torch.load(output_dir / "best_clip_classifier.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    y_true, y_pred, probs, names = predict_all(model, processor, test_loader, device)
    test_acc = accuracy_score(y_true, y_pred)
    test_top3_acc = top_k_accuracy_from_probs(probs, y_true, k=3)
    target_names = [train_ds.idx_to_label[i] for i in range(len(train_ds.idx_to_label))]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

    save_training_curves(history, output_dir)
    save_confusion_matrix(y_true, y_pred, train_ds.idx_to_label, output_dir)

    pred_rows = []
    for actual, pred, prob, name in zip(y_true, y_pred, probs, names):
        top3_indices = np.argsort(prob)[-3:][::-1]
        pred_rows.append(
            {
                "name": name,
                "actual": train_ds.idx_to_label[int(actual)],
                "predicted": train_ds.idx_to_label[int(pred)],
                "confidence": float(np.max(prob)),
                "top3_predictions": " | ".join(train_ds.idx_to_label[int(idx)] for idx in top3_indices),
                "top3_correct": bool(int(actual) in top3_indices),
            }
        )

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    report_text = (
        f"Test accuracy: {test_acc:.4f}\n"
        f"Test top-3 accuracy: {test_top3_acc:.4f}\n\n"
        f"{report}"
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "label_map.json").write_text(
        json.dumps(train_ds.label_to_idx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "test_top3_acc": test_top3_acc,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTest accuracy:", f"{test_acc:.4f}")
    print("Test top-3 accuracy:", f"{test_top3_acc:.4f}")
    print(report)
    print(f"Saved outputs to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CLIP-based perfume family classifier.")
    parser.add_argument("--train-csv", default="data/All/train.csv")
    parser.add_argument("--val-csv", default="data/All/val.csv")
    parser.add_argument("--test-csv", default="data/All/test.csv")
    parser.add_argument("--output-dir", default="results/clip_classifier")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--embedding-cache-dir", default="results/clip_embedding_cache")
    parser.add_argument("--cache-embeddings", action="store_true", default=True)
    parser.add_argument("--no-cache-embeddings", dest="cache_embeddings", action="store_false")
    parser.add_argument("--rebuild-embedding-cache", action="store_true")
    parser.add_argument("--local-data-dir", default="/content/perfume_local")
    parser.add_argument("--copy-data-to-local", action="store_true")
    parser.add_argument("--auto-copy-data-to-local", action="store_true", default=True)
    parser.add_argument("--no-auto-copy-data-to-local", dest="auto_copy_data_to_local", action="store_false")
    parser.add_argument("--freeze-clip", action="store_true", default=True)
    parser.add_argument("--train-clip", dest="freeze_clip", action="store_false")
    parser.add_argument("--weighted-loss", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
