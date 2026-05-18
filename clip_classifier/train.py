"""
CLIP 이미지 단독 향수 계열 분류기

텍스트 모드:
  --text-mode none        : 이미지 임베딩만 사용 (누수 없음)
  --text-mode name_brand  : "name by brand" 텍스트 병합 (누수 없음)
  --text-mode notes       : notes 텍스트 사용 (비교용, 누수 있음)

기본 실행 (이미지 단독):
  python clip_classifier/train.py

name/brand 텍스트 포함:
  python clip_classifier/train.py --text-mode name_brand

Colab (ViT-L/14):
  python clip_classifier/train.py \
    --clip-model openai/clip-vit-large-patch14 \
    --text-mode none \
    --epochs 20 --patience 7 --batch-size 16 \
    --rebuild-embedding-cache
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
    raise SystemExit("pip install transformers 를 먼저 실행하세요.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_DIR


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    normalized = str(image_path).replace("\\", "/")
    raw = Path(normalized)
    if raw.is_absolute() and raw.exists():
        return raw
    for base in (DATA_ROOT, PROJECT_DIR, csv_dir, Path.cwd()):
        candidate = base / raw
        if candidate.exists():
            return candidate
    return DATA_ROOT / raw


def maybe_copy_data_to_local(args) -> None:
    global DATA_ROOT
    should_auto = args.auto_copy_data_to_local and str(PROJECT_DIR).startswith("/content/drive/")
    if not args.copy_data_to_local and not should_auto:
        return
    local_root = Path(args.local_data_dir)
    local_root.mkdir(parents=True, exist_ok=True)
    for folder_name in ("data", "perfume_images"):
        src = PROJECT_DIR / folder_name
        dst = local_root / folder_name
        if src.exists():
            print(f"Copying {src} -> {dst}", flush=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
    DATA_ROOT = local_root
    print(f"Local data root: {DATA_ROOT}", flush=True)


# ──────────────────────────────────────────────
# 텍스트 모드별 입력 생성
# ──────────────────────────────────────────────

def build_text(row: pd.Series, text_mode: str) -> str:
    if text_mode == "none":
        return "a perfume bottle"
    if text_mode == "name_brand":
        name = str(row.get("name", "") or "").strip()
        brand = str(row.get("brand", "") or "").strip()
        if name and brand:
            return f"{name} by {brand}"
        return name or brand or "a perfume"
    if text_mode == "notes":
        notes = str(row.get("notes", "") or "").strip()
        return f"fragrance notes: {notes}" if notes else "fragrance notes unavailable"
    raise ValueError(f"알 수 없는 text_mode: {text_mode}")


# ──────────────────────────────────────────────
# 데이터셋
# ──────────────────────────────────────────────

class PerfumeDataset(Dataset):
    def __init__(self, csv_path: Path, text_mode: str, label_to_idx=None):
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.text_mode = text_mode
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
        text = build_text(row, self.text_mode)
        label = self.label_to_idx[row["label"]]
        name = str(row.get("name", image_path.name))
        return image, text, label, name


def collate_fn(batch):
    images, texts, labels, names = zip(*batch)
    return list(images), list(texts), torch.tensor(labels, dtype=torch.long), list(names)


# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────

class ImageOnlyClassifier(nn.Module):
    """CLIP vision encoder + MLP head. 텍스트 인코더 사용 안 함."""

    def __init__(self, clip_model_name: str, num_classes: int, hidden_dim: int,
                 dropout: float, num_layers: int = 2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        feature_dim = self.clip.config.projection_dim

        if num_layers >= 3:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def freeze_clip(self):
        for p in self.clip.parameters():
            p.requires_grad = False

    def unfreeze_last_n_vision_layers(self, n: int) -> None:
        for p in self.clip.parameters():
            p.requires_grad = False
        if n <= 0:
            return
        for layer in self.clip.vision_model.encoder.layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.clip.vision_model.post_layernorm.parameters():
            p.requires_grad = True
        for p in self.clip.visual_projection.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.clip.parameters())
        print(f"Vision unfreeze last {n} layers: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)", flush=True)

    def forward(self, pixel_values):
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        image_features = nn.functional.normalize(image_features, dim=-1)
        return self.classifier(image_features)


class FusionClassifier(nn.Module):
    """CLIP vision + text encoder concat. text_mode=name_brand 용."""

    def __init__(self, clip_model_name: str, num_classes: int, hidden_dim: int,
                 dropout: float, num_layers: int = 2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        feature_dim = self.clip.config.projection_dim * 2

        if num_layers >= 3:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def freeze_clip(self):
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        img = nn.functional.normalize(outputs.image_embeds, dim=-1)
        txt = nn.functional.normalize(outputs.text_embeds, dim=-1)
        return self.classifier(torch.cat([img, txt], dim=-1))


class FeatureClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int,
                 dropout: float, num_layers: int = 2):
        super().__init__()
        if num_layers >= 3:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
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


# ──────────────────────────────────────────────
# 데이터 로더
# ──────────────────────────────────────────────

def build_datasets(args):
    train_ds = PerfumeDataset(resolve_csv(args.train_csv), args.text_mode)
    val_ds   = PerfumeDataset(resolve_csv(args.val_csv),   args.text_mode, train_ds.label_to_idx)
    test_ds  = PerfumeDataset(resolve_csv(args.test_csv),  args.text_mode, train_ds.label_to_idx)
    return train_ds, val_ds, test_ds


def make_weighted_sampler(labels_series):
    counts = labels_series.value_counts()
    weights = labels_series.map(lambda l: 1.0 / counts[l]).to_numpy(copy=True)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_loaders(args):
    train_ds, val_ds, test_ds = build_datasets(args)
    sampler = make_weighted_sampler(train_ds.df["label"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=False,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# 임베딩 캐시
# ──────────────────────────────────────────────

def safe_cache_name(clip_model: str, text_mode: str, split_name: str) -> str:
    model_name = clip_model.replace("/", "__").replace("\\", "__")
    return f"{model_name}_{text_mode}_{split_name}_features.pt"


def extract_or_load_features(split_name, dataset, clip_model, processor, args, device):
    cache_root = Path(args.embedding_cache_dir)
    if not cache_root.is_absolute():
        cache_root = PROJECT_DIR / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / safe_cache_name(args.clip_model, args.text_mode, split_name)

    if cache_path.exists() and not args.rebuild_embedding_cache:
        print(f"[cache] {split_name} 로드: {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"[extract] {split_name} 임베딩 추출 → {cache_path}", flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)
    clip_model.eval()
    features, labels_all, names_all = [], [], []
    total = len(loader)
    use_text = (args.text_mode != "none")

    with torch.no_grad():
        for i, (images, texts, labels, names) in enumerate(loader, 1):
            encoded = processor(
                images=images,
                text=texts if use_text else None,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            vision_out = clip_model.vision_model(pixel_values=encoded["pixel_values"])
            img_feat = nn.functional.normalize(
                clip_model.visual_projection(vision_out.pooler_output), dim=-1
            )
            if use_text:
                text_out = clip_model.text_model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                txt_feat = nn.functional.normalize(
                    clip_model.text_projection(text_out.pooler_output), dim=-1
                )
                batch_feat = torch.cat([img_feat, txt_feat], dim=-1)
            else:
                batch_feat = img_feat

            features.append(batch_feat.cpu())
            labels_all.append(labels)
            names_all.extend(names)

            if args.log_every > 0 and (i == 1 or i % args.log_every == 0 or i == total):
                print(f"  {split_name} {i}/{total}", flush=True)

    payload = {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels_all, dim=0),
        "names": names_all,
        "label_to_idx": dataset.label_to_idx,
        "idx_to_label": dataset.idx_to_label,
        "clip_model": args.clip_model,
        "text_mode": args.text_mode,
    }
    torch.save(payload, cache_path)
    return payload


def build_feature_loaders(train_p, val_p, test_p, args):
    train_ds = FeatureDataset(train_p["features"], train_p["labels"], train_p["names"])
    val_ds   = FeatureDataset(val_p["features"],   val_p["labels"],   val_p["names"])
    test_ds  = FeatureDataset(test_p["features"],  test_p["labels"],  test_p["names"])

    labels_np = train_p["labels"].numpy()
    counts = np.bincount(labels_np)
    sample_weights = np.array([1.0 / counts[l] for l in labels_np], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,   num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,   num_workers=0)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────

def top_k_accuracy(logits_or_probs, labels, k=3):
    if isinstance(logits_or_probs, torch.Tensor):
        k = min(k, logits_or_probs.size(1))
        topk = logits_or_probs.topk(k, dim=1).indices
        correct = topk.eq(labels.view(-1, 1)).any(dim=1)
        return correct.detach().cpu().numpy().tolist()
    # numpy probs
    k = min(k, logits_or_probs.shape[1])
    topk = np.argsort(logits_or_probs, axis=1)[:, -k:]
    return [int(a) in row for a, row in zip(labels, topk)]


def run_feature_epoch(model, loader, criterion, optimizer, scaler, device, train=True, log_every=50):
    model.train(train)
    total_loss, y_true, y_pred, top3 = 0.0, [], [], []
    mode = "train" if train else "val"

    for i, (features, labels, _) in enumerate(loader, 1):
        features = features.to(device)
        labels   = labels.to(device)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=False):
                logits = model(features)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        top3.extend(top_k_accuracy(logits, labels, k=3))

        if log_every > 0 and (i == 1 or i % log_every == 0 or i == len(loader)):
            print(f"  {mode} {i}/{len(loader)} "
                  f"loss={total_loss/len(y_true):.4f} "
                  f"acc={accuracy_score(y_true, y_pred):.4f} "
                  f"top3={float(np.mean(top3)):.4f}", flush=True)

    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), float(np.mean(top3))


def predict_feature_all(model, loader, device):
    model.eval()
    y_true, y_pred, probs_all, names = [], [], [], []
    with torch.no_grad():
        for features, labels, batch_names in loader:
            features = features.to(device)
            logits = model(features)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probs_all.extend(p.tolist())
            names.extend(batch_names)
    return np.array(y_true), np.array(y_pred), np.array(probs_all), names


# ──────────────────────────────────────────────
# 결과 저장
# ──────────────────────────────────────────────

def save_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key, title in zip(axes,
                               [("train_loss", "val_loss"), ("train_acc", "val_acc"), ("train_top3", "val_top3")],
                               ["Loss", "Accuracy", "Top-3 Accuracy"]):
        ax.plot(epochs, history[key[0]], label="train")
        ax.plot(epochs, history[key[1]], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if "acc" in key[0]:
            ax.set_ylim(0, 1)
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close()


def save_confusion_matrix(y_true, y_pred, idx_to_label, output_dir):
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    for r in range(cm_norm.shape[0]):
        for c in range(cm_norm.shape[1]):
            v = cm_norm[r, c]
            plt.text(c, r, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.5 else "black", fontsize=9)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def save_results(output_dir, history, y_true, y_pred, probs, idx_to_label,
                 best_epoch, best_val_acc, args):
    test_acc = accuracy_score(y_true, y_pred)
    test_top3 = float(np.mean(top_k_accuracy(probs, y_true, k=3)))
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

    save_training_curves(history, output_dir)
    save_confusion_matrix(y_true, y_pred, idx_to_label, output_dir)

    pred_rows = []
    for actual, pred, prob, name in zip(y_true, y_pred, probs, []):
        top3_idx = np.argsort(prob)[-3:][::-1]
        pred_rows.append({
            "actual": idx_to_label[int(actual)],
            "predicted": idx_to_label[int(pred)],
            "confidence": float(np.max(prob)),
            "top3": " | ".join(idx_to_label[int(i)] for i in top3_idx),
            "top3_correct": bool(int(actual) in top3_idx),
        })

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    report_text = (
        f"text_mode : {args.text_mode}\n"
        f"clip_model: {args.clip_model}\n\n"
        f"Test accuracy      : {test_acc:.4f}\n"
        f"Test top-3 accuracy: {test_top3:.4f}\n\n"
        f"{report}"
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps({
            "text_mode": args.text_mode,
            "clip_model": args.clip_model,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "test_top3_acc": test_top3,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'='*50}")
    print(f"text_mode : {args.text_mode}")
    print(f"clip_model: {args.clip_model}")
    print(f"Test accuracy      : {test_acc:.4f}")
    print(f"Test top-3 accuracy: {test_top3:.4f}")
    print(f"{'='*50}")
    print(report)
    print(f"저장 위치: {output_dir}", flush=True)

    return test_acc, test_top3


# ──────────────────────────────────────────────
# 메인 학습
# ──────────────────────────────────────────────

def train(args):
    seed_everything(args.seed)
    maybe_copy_data_to_local(args)

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_DIR / output_root
    run_name = f"{args.text_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device    : {device}", flush=True)
    print(f"text_mode : {args.text_mode}", flush=True)
    print(f"CLIP model: {args.clip_model}", flush=True)
    print(f"Output    : {output_dir}", flush=True)

    # ── 임베딩 추출 (캐시 방식)
    train_ds, val_ds, test_ds = build_datasets(args)
    print(f"Classes: {train_ds.label_to_idx}", flush=True)
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}", flush=True)

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    for p in clip_model.parameters():
        p.requires_grad = False

    train_p = extract_or_load_features("train", train_ds, clip_model, processor, args, device)
    val_p   = extract_or_load_features("val",   val_ds,   clip_model, processor, args, device)
    test_p  = extract_or_load_features("test",  test_ds,  clip_model, processor, args, device)

    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 분류기 학습
    _, _, _, train_loader, val_loader, test_loader = build_feature_loaders(train_p, val_p, test_p, args)

    num_classes = len(train_p["label_to_idx"])
    feature_dim = int(train_p["features"].shape[1])
    model = FeatureClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_head_layers,
    ).to(device)
    print(f"Feature dim: {feature_dim} | Head: {args.num_head_layers} layers, hidden={args.hidden_dim}", flush=True)

    labels_tensor = train_p["labels"]
    counts = np.bincount(labels_tensor.numpy(), minlength=num_classes)
    total = counts.sum()
    class_weights = torch.tensor(
        [total / (num_classes * max(1, c)) for c in counts], dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    best_val_acc, best_epoch, patience_count = 0.0, 0, 0
    history = {k: [] for k in ("train_loss", "train_acc", "train_top3", "val_loss", "val_acc", "val_top3")}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_top3 = run_feature_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True, log_every=args.log_every)
        vl_loss, vl_acc, vl_top3 = run_feature_epoch(
            model, val_loader, criterion, optimizer, scaler, device, train=False, log_every=0)
        scheduler.step()

        for key, val in zip(
            ("train_loss", "train_acc", "train_top3", "val_loss", "val_acc", "val_top3"),
            (tr_loss, tr_acc, tr_top3, vl_loss, vl_acc, vl_top3),
        ):
            history[key].append(val)

        print(f"Epoch {epoch:02d}/{args.epochs} "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} top3={tr_top3:.4f} | "
              f"val_loss={vl_loss:.4f} acc={vl_acc:.4f} top3={vl_top3:.4f}", flush=True)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = epoch
            patience_count = 0
            torch.save({
                "model_state": model.state_dict(),
                "label_to_idx": train_p["label_to_idx"],
                "idx_to_label": train_p["idx_to_label"],
                "val_acc": best_val_acc,
                "epoch": epoch,
                "feature_dim": feature_dim,
                "text_mode": args.text_mode,
            }, output_dir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    # ── 테스트 평가
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    y_true, y_pred, probs, names = predict_feature_all(model, test_loader, device)
    save_results(output_dir, history, y_true, y_pred, probs,
                 train_p["idx_to_label"], best_epoch, best_val_acc, args)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP 이미지 기반 향수 계열 분류기")
    parser.add_argument("--train-csv", default="data/All/train.csv")
    parser.add_argument("--val-csv",   default="data/All/val.csv")
    parser.add_argument("--test-csv",  default="data/All/test.csv")
    parser.add_argument("--output-dir", default="results/clip_experiments")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                        help="HuggingFace CLIP 모델 이름. 예: openai/clip-vit-large-patch14")
    parser.add_argument("--text-mode", default="none",
                        choices=["none", "name_brand", "notes"],
                        help="none: 이미지만 | name_brand: 이름+브랜드 | notes: 누수있음(비교용)")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--num-workers",   type=int,   default=2)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--hidden-dim",    type=int,   default=512)
    parser.add_argument("--dropout",       type=float, default=0.25)
    parser.add_argument("--patience",      type=int,   default=7)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--log-every",     type=int,   default=100)
    parser.add_argument("--num-head-layers", type=int, default=2, choices=[2, 3])
    parser.add_argument("--embedding-cache-dir", default="results/clip_embedding_cache")
    parser.add_argument("--rebuild-embedding-cache", action="store_true")
    parser.add_argument("--local-data-dir", default="/content/perfume_local")
    parser.add_argument("--copy-data-to-local",      action="store_true")
    parser.add_argument("--auto-copy-data-to-local",    action="store_true", default=True)
    parser.add_argument("--no-auto-copy-data-to-local", dest="auto_copy_data_to_local",
                        action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
