"""
CLIP image+text two-stage fine-tune for perfume Note classification.

train_clip_finetune.py 와 동일한 2단계 구조이지만,
CLIP 텍스트 인코더로 brand+name 메타데이터를 인코딩해
이미지 특징(512-dim)과 concat(→ 1024-dim)하여 분류합니다.

Stage 1:
  - CLIP visual+text encoder 전체 Freeze
  - (image_feat || text_feat) 캐시 → ClipHead만 학습

Stage 2:
  - CLIP visual encoder 마지막 1블록 Unfreeze
  - text encoder는 계속 Freeze (메타데이터 특성상 fine-tune 불필요)
  - head + visual block 학습

notes 컬럼은 사용하지 않습니다.
"""
import argparse
import json
import os
import random
from pathlib import Path

import torch.nn.functional as F

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
    "Floral", "Woody", "Amber_Oriental", "Citrus", "Sweet", "Spicy", "Fresh",
]


# ──────────────────────────────────────────────
# Dataset (이미지 + 텍스트 토큰 반환)
# ──────────────────────────────────────────────

class PerfumeImageTextDataset(Dataset):
    def __init__(self, csv_path, label_to_idx, transform, tokenizer, max_rows=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["label"].isin(label_to_idx)].reset_index(drop=True)
        if max_rows:
            self.df = self.df.head(max_rows).reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self._resolve(row["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"[Dataset] load failed: {image_path} -> {exc}")
            image = Image.new("RGB", (224, 224))

        brand = str(row.get("brand", "") or "")
        name  = str(row.get("name",  "") or "")
        text  = f"{brand} {name}".strip() or "perfume"
        text_tokens = self.tokenizer([text])[0]  # (77,)

        label = self.label_to_idx[str(row["label"])]
        return self.transform(image), text_tokens, label

    @staticmethod
    def _resolve(image_path):
        p = Path(str(image_path).replace("\\", os.sep))
        return p if p.is_absolute() else PROJECT_ROOT / p


# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────

class ClipHead(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ClipTextImageClassifier(nn.Module):
    """이미지 + 텍스트 특징을 concat하여 분류."""

    def __init__(self, clip_model, head):
        super().__init__()
        self.clip_model = clip_model
        self.head = head

    def forward(self, images, text_tokens):
        img_feat = F.normalize(self.clip_model.encode_image(images).float(), dim=-1)
        # 텍스트 인코더는 항상 Freeze
        with torch.no_grad():
            txt_feat = F.normalize(self.clip_model.encode_text(text_tokens).float(), dim=-1)
        return self.head(torch.cat([img_feat, txt_feat], dim=-1))


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preferred):
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_labels(train_csv):
    df = pd.read_csv(train_csv)
    unique = set(df["label"].astype(str).unique())
    labels = [l for l in NOTE_CLASS_ORDER if l in unique]
    labels += [l for l in sorted(unique) if l not in labels]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    return labels, label_to_idx, idx_to_label


def make_weighted_sampler(labels_arr):
    counts = np.bincount(labels_arr)
    weights = np.array([1.0 / counts[l] for l in labels_arr], dtype=np.float64)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                  len(weights), replacement=True)


def class_weights_tensor(labels_arr, num_classes, device):
    counts = np.bincount(labels_arr, minlength=num_classes)
    total = counts.sum()
    w = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def freeze_clip(clip_model):
    for p in clip_model.parameters():
        p.requires_grad = False


def unfreeze_last_visual_blocks(clip_model, n_blocks):
    freeze_clip(clip_model)
    blocks = list(clip_model.visual.transformer.resblocks)
    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    for name in ["ln_post", "proj"]:
        m = getattr(clip_model.visual, name, None)
        if m is None:
            continue
        if isinstance(m, nn.Parameter):
            m.requires_grad = True
        else:
            for p in m.parameters():
                p.requires_grad = True
    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in clip_model.parameters())
    print(f"[Stage2] visual last {n_blocks} block(s) unfrozen | {trainable:,}/{total:,}")


def save_checkpoint(path, state, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, model_state=state)
    torch.save(payload, path)
    print(f"[Checkpoint] saved: {path}")


def top_k_accuracy(scores, labels, k=3):
    top_k = np.argsort(scores, axis=1)[:, -k:]
    return float(np.mean([labels[i] in top_k[i] for i in range(len(labels))]))


def softmax_np(scores):
    e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


# ──────────────────────────────────────────────
# Stage 1: Frozen CLIP → 캐시 특징으로 Head만 학습
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_features(clip_model, loader, device, split_name, log_every):
    """이미지(512) + 텍스트(512) = 1024-dim concat 특징 추출."""
    clip_model.eval()
    feats, lbls = [], []
    for step, (images, text_tokens, labels) in enumerate(loader, 1):
        images      = images.to(device)
        text_tokens = text_tokens.to(device)
        img_feat = F.normalize(clip_model.encode_image(images).float(), dim=-1)
        txt_feat = F.normalize(clip_model.encode_text(text_tokens).float(), dim=-1)
        feats.append(torch.cat([img_feat, txt_feat], dim=-1).cpu())
        lbls.append(labels)
        if log_every and (step == 1 or step % log_every == 0 or step == len(loader)):
            print(f"[Features:{split_name}] batch {step}/{len(loader)}", flush=True)
    return torch.cat(feats), torch.cat(lbls)


def load_or_extract(clip_model, loader, device, args, split_name):
    cache_dir  = PROJECT_ROOT / args.feature_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_key  = f"{args.model}_{args.pretrained}_text".replace("/", "_")
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


def train_head_epoch(head, loader, criterion, optimizer, device, train):
    head.train(train)
    loss_sum, y_true, y_pred = 0.0, [], []
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = head(features)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())
    n = len(loader.dataset)
    return (loss_sum / n,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0))


def run_stage1(clip_model, train_ds, val_ds, device, args, num_classes):
    print("\n" + "=" * 60)
    print("[Stage 1] Frozen CLIP (image+text) head training")
    print("=" * 60)

    freeze_clip(clip_model)
    raw_loader = lambda ds: DataLoader(ds, batch_size=args.extract_batch_size,
                                        shuffle=False, num_workers=args.workers)
    tr_feat, tr_lbl = load_or_extract(clip_model, raw_loader(train_ds), device, args, args.train_split)
    vl_feat, vl_lbl = load_or_extract(clip_model, raw_loader(val_ds),   device, args, "val")

    feature_dim = tr_feat.shape[1]  # 1024
    print(f"[Stage1] feature_dim={feature_dim} (image 512 + text 512)")

    tr_sampler    = make_weighted_sampler(tr_lbl.numpy())
    tr_feat_ds    = TensorDataset(tr_feat, tr_lbl)
    vl_feat_ds    = TensorDataset(vl_feat, vl_lbl)
    tr_feat_loader = DataLoader(tr_feat_ds, batch_size=args.head_batch_size, sampler=tr_sampler)
    vl_feat_loader = DataLoader(vl_feat_ds, batch_size=args.head_batch_size, shuffle=False)

    head      = ClipHead(feature_dim, num_classes, args.hidden_dim, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor(tr_lbl.numpy(), num_classes, device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.stage1_lr,
                                   weight_decay=args.weight_decay)

    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1   = -1.0
    best_path = PROJECT_ROOT / args.checkpoint_dir / "clip_text_stage1_best.pt"

    for epoch in range(1, args.stage1_epochs + 1):
        tr_l, tr_a, _    = train_head_epoch(head, tr_feat_loader, criterion, optimizer, device, train=True)
        vl_l, vl_a, vl_f = train_head_epoch(head, vl_feat_loader, criterion, optimizer, device, train=False)
        history["train_loss"].append(tr_l); history["val_loss"].append(vl_l)
        history["train_acc"].append(tr_a);  history["val_acc"].append(vl_a)
        history["val_f1"].append(vl_f)
        print(f"[S1 E{epoch:02d}/{args.stage1_epochs}] "
              f"train_loss={tr_l:.4f} acc={tr_a:.4f} | "
              f"val_loss={vl_l:.4f} acc={vl_a:.4f} f1={vl_f:.4f}", flush=True)
        if vl_f > best_f1:
            best_f1 = vl_f
            save_checkpoint(best_path, head.state_dict(),
                            {"stage": 1, "epoch": epoch, "macro_f1": vl_f})

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["model_state"])
    return head, history, best_f1, feature_dim


# ──────────────────────────────────────────────
# Stage 2: 마지막 Visual 블록 Unfreeze + Head 학습
# ──────────────────────────────────────────────

def train_image_text_epoch(model, loader, criterion, optimizer, device, train, log_every):
    model.train(train)
    loss_sum, y_true, y_pred = 0.0, [], []
    mode = "train" if train else "val"
    for step, (images, text_tokens, labels) in enumerate(loader, 1):
        images, text_tokens, labels = images.to(device), text_tokens.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images, text_tokens)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())
        if log_every and (step == 1 or step % log_every == 0 or step == len(loader)):
            print(f"[Stage2:{mode}] batch {step}/{len(loader)} "
                  f"loss={loss_sum/max(len(y_true),1):.4f}", flush=True)
    n = len(loader.dataset)
    return (loss_sum / n,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0))


def run_stage2(clip_model, head, train_ds, val_ds, device, args, num_classes, feature_dim):
    print("\n" + "=" * 60)
    print("[Stage 2] CLIP visual last block + head fine-tuning (text encoder frozen)")
    print("=" * 60)

    unfreeze_last_visual_blocks(clip_model, args.unfreeze_last_blocks)
    model = ClipTextImageClassifier(clip_model, head).to(device)

    lbl_arr   = train_ds.df["label"].map(train_ds.label_to_idx).to_numpy()
    tr_sampler = make_weighted_sampler(lbl_arr)
    tr_loader  = DataLoader(train_ds, batch_size=args.stage2_batch_size,
                             sampler=tr_sampler, num_workers=args.workers)
    vl_loader  = DataLoader(val_ds,   batch_size=args.stage2_batch_size,
                             shuffle=False, num_workers=args.workers)

    criterion  = nn.CrossEntropyLoss(
        weight=class_weights_tensor(lbl_arr, num_classes, device),
        label_smoothing=args.label_smoothing,
    )
    head_params   = [p for p in model.head.parameters() if p.requires_grad]
    visual_params = [p for p in model.clip_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": head_params,   "lr": args.stage2_head_lr},
         {"params": visual_params, "lr": args.stage2_visual_lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage2_epochs)

    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1   = -1.0
    best_path = PROJECT_ROOT / args.checkpoint_dir / "clip_text_stage2_best.pt"

    for epoch in range(1, args.stage2_epochs + 1):
        tr_l, tr_a, _    = train_image_text_epoch(model, tr_loader, criterion, optimizer,
                                                    device, True,  args.log_every)
        vl_l, vl_a, vl_f = train_image_text_epoch(model, vl_loader, criterion, optimizer,
                                                    device, False, args.log_every)
        scheduler.step()
        history["train_loss"].append(tr_l); history["val_loss"].append(vl_l)
        history["train_acc"].append(tr_a);  history["val_acc"].append(vl_a)
        history["val_f1"].append(vl_f)
        print(f"[S2 E{epoch:02d}/{args.stage2_epochs}] "
              f"train_loss={tr_l:.4f} acc={tr_a:.4f} | "
              f"val_loss={vl_l:.4f} acc={vl_a:.4f} f1={vl_f:.4f}", flush=True)
        if vl_f > best_f1:
            best_f1 = vl_f
            save_checkpoint(best_path, model.state_dict(),
                            {"stage": 2, "epoch": epoch, "macro_f1": vl_f})

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, history, best_f1


# ──────────────────────────────────────────────
# 평가 및 결과 저장
# ──────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, device, log_every):
    model.eval()
    y_true, y_pred, scores = [], [], []
    for step, (images, text_tokens, labels) in enumerate(loader, 1):
        images, text_tokens = images.to(device), text_tokens.to(device)
        logits = model(images, text_tokens).float()
        y_true.extend(labels.numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())
        scores.append(logits.cpu().numpy())
        if log_every and (step == 1 or step % log_every == 0 or step == len(loader)):
            print(f"[Test] batch {step}/{len(loader)}", flush=True)
    return np.array(y_true), np.array(y_pred), np.concatenate(scores)


def save_outputs(args, model, test_ds, device, idx_to_label, class_names, h1, h2):
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size,
                              shuffle=False, num_workers=args.workers)
    y_true, y_pred, raw_scores = predict(model, test_loader, device, args.log_every)
    probs = softmax_np(raw_scores)

    metrics = {
        "model":          f"CLIP {args.model} ({args.pretrained}) + text two-stage fine-tune",
        "accuracy":       float(accuracy_score(y_true, y_pred)),
        "macro_f1":       float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "top3_accuracy":  top_k_accuracy(raw_scores, y_true, 3),
        "random_baseline": 1.0 / len(class_names),
        "notes_used":     False,
        "brand_name_used": True,
    }
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    # 학습 곡선
    full_h = {k: h1[k] + h2[k] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1"]}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(full_h["train_loss"], label="train"); axes[0].plot(full_h["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(full_h["train_acc"], label="train"); axes[1].plot(full_h["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_ylim(0, 1); axes[1].legend()
    axes[2].plot(full_h["val_f1"], label="val macro f1")
    axes[2].set_title("Macro F1"); axes[2].set_ylim(0, 1); axes[2].legend()
    plt.tight_layout(); plt.savefig(output_dir / "training_curves.png", dpi=160); plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_n = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for r in range(cm_n.shape[0]):
        for c in range(cm_n.shape[1]):
            v = cm_n[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_title("CLIP+Text Fine-tuned Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close()

    # Per-class accuracy
    per_cls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, per_cls, color="#4c78a8", edgecolor="white")
    ax.axhline(per_cls.mean(), color="#d62728", linestyle="--", label=f"Mean: {per_cls.mean():.3f}")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.set_title("CLIP+Text Per-Class Accuracy"); ax.legend()
    for bar, v in zip(bars, per_cls):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(output_dir / "per_class_accuracy.png", dpi=160); plt.close()

    # 저장
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        f"Model        : {metrics['model']}\n"
        f"brand+name   : True\n"
        f"notes_used   : False\n"
        f"Accuracy     : {metrics['accuracy']:.4f}\n"
        f"Macro F1     : {metrics['macro_f1']:.4f}\n"
        f"Top-3 Acc    : {metrics['top3_accuracy']:.4f}\n"
        f"Random Base  : {metrics['random_baseline']:.4f}\n\n{report}",
        encoding="utf-8",
    )

    rows = []
    for i, row in test_ds.df.reset_index(drop=True).iterrows():
        top3 = np.argsort(probs[i])[-3:][::-1]
        rows.append({
            "image_path": row.get("image_path", ""), "name": row.get("name", ""),
            "brand": row.get("brand", ""),
            "actual": idx_to_label[int(y_true[i])], "predicted": idx_to_label[int(y_pred[i])],
            "confidence": float(probs[i, int(y_pred[i])]),
            "top3_predictions": " | ".join(idx_to_label[int(j)] for j in top3),
            "top3_correct": bool(int(y_true[i]) in top3),
        })
    pd.DataFrame(rows).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    print(report, flush=True)
    print(f"[Final] {metrics}", flush=True)
    print(f"[Final] outputs saved: {output_dir}", flush=True)
    return metrics


# ──────────────────────────────────────────────
# 인자 파싱
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CLIP image+text two-stage fine-tune")
    p.add_argument("--data-dir",               default="data/All")
    p.add_argument("--train-split",            default="train_aug")
    p.add_argument("--model",                  default="ViT-B-32-quickgelu")
    p.add_argument("--pretrained",             default="openai")
    p.add_argument("--stage1-epochs",   type=int,   default=5)
    p.add_argument("--stage2-epochs",   type=int,   default=3)
    p.add_argument("--extract-batch-size", type=int, default=64)
    p.add_argument("--head-batch-size",  type=int,  default=256)
    p.add_argument("--stage2-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size",  type=int,  default=64)
    p.add_argument("--hidden-dim",       type=int,  default=512)
    p.add_argument("--dropout",          type=float, default=0.25)
    p.add_argument("--stage1-lr",        type=float, default=1e-3)
    p.add_argument("--stage2-head-lr",   type=float, default=1e-4)
    p.add_argument("--stage2-visual-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    p.add_argument("--label-smoothing",  type=float, default=0.05)
    p.add_argument("--unfreeze-last-blocks", type=int, default=1)
    p.add_argument("--workers",          type=int,  default=0)
    p.add_argument("--seed",             type=int,  default=42)
    p.add_argument("--device",           default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--log-every",        type=int,  default=50)
    p.add_argument("--output-dir",       default="results/clip_text")
    p.add_argument("--checkpoint-dir",   default="checkpoints/clip_text")
    p.add_argument("--feature-cache-dir", default="results/clip_text/features")
    p.add_argument("--cache-features",   action="store_true", default=True)
    p.add_argument("--no-cache-features", dest="cache_features", action="store_false")
    p.add_argument("--rebuild-feature-cache", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples",  type=int, default=0)
    return p.parse_args()


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}", flush=True)

    import open_clip

    data_dir  = PROJECT_ROOT / args.data_dir
    train_csv = data_dir / f"{args.train_split}.csv"
    if not train_csv.exists():
        print(f"[Data] {train_csv} not found; fallback to train.csv")
        train_csv = data_dir / "train.csv"
        args.train_split = "train"
    val_csv  = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    class_names, label_to_idx, idx_to_label = read_labels(data_dir / "train.csv")
    print(f"Classes: {class_names}")

    cache_dir = PROJECT_ROOT / ".torch_cache" / "open_clip"
    clip_model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device, cache_dir=str(cache_dir),
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    max_tr = args.max_train_samples or None
    max_ev = args.max_eval_samples  or None

    train_ds_s1 = PerfumeImageTextDataset(train_csv, label_to_idx, test_preprocess,  tokenizer, max_tr)
    train_ds_s2 = PerfumeImageTextDataset(train_csv, label_to_idx, train_preprocess, tokenizer, max_tr)
    val_ds      = PerfumeImageTextDataset(val_csv,   label_to_idx, test_preprocess,  tokenizer, max_ev)
    test_ds     = PerfumeImageTextDataset(test_csv,  label_to_idx, test_preprocess,  tokenizer, max_ev)
    print(f"Rows train/val/test: {len(train_ds_s2)}/{len(val_ds)}/{len(test_ds)}")

    head, h1, _, feature_dim = run_stage1(
        clip_model, train_ds_s1, val_ds, device, args, len(class_names))
    model, h2, _ = run_stage2(
        clip_model, head, train_ds_s2, val_ds, device, args, len(class_names), feature_dim)

    save_outputs(args, model, test_ds, device, idx_to_label, class_names, h1, h2)


if __name__ == "__main__":
    main()
