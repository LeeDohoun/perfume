"""
CLIP note classification.

Usage:
  python scripts/eval_clip_note.py --mode zero_shot --split test
  python scripts/eval_clip_note.py --mode linear_probe --split test

Target:
  Note classification with CLIP: 45% ~ 60% test accuracy
"""
import os
import argparse
import json
import warnings
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, ".torch_cache"))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".matplotlib_cache"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_ACCURACY = (0.45, 0.60)

NOTE_CLASS_ORDER = [
    "Floral",
    "Woody",
    "Amber_Oriental",
    "Citrus",
    "Sweet",
    "Spicy",
    "Fresh",
]

NOTE_PROMPTS = {
    "Amber_Oriental": [
        "a photo of an amber oriental perfume bottle",
        "a warm amber musk incense perfume bottle",
        "a resinous balsamic perfume product",
    ],
    "Citrus": [
        "a photo of a citrus perfume bottle",
        "a fresh lemon bergamot orange perfume bottle",
        "a bright zesty perfume product",
    ],
    "Floral": [
        "a photo of a floral perfume bottle",
        "a rose jasmine iris perfume bottle",
        "a soft flower fragrance perfume product",
    ],
    "Fresh": [
        "a photo of a fresh clean perfume bottle",
        "a green aquatic aromatic perfume bottle",
        "a crisp airy fresh fragrance product",
    ],
    "Spicy": [
        "a photo of a spicy perfume bottle",
        "a pepper cardamom cinnamon perfume bottle",
        "a warm spice fragrance perfume product",
    ],
    "Sweet": [
        "a photo of a sweet perfume bottle",
        "a vanilla caramel honey perfume bottle",
        "a gourmand sweet fragrance perfume product",
    ],
    "Woody": [
        "a photo of a woody perfume bottle",
        "a sandalwood cedar oud perfume bottle",
        "an earthy wood fragrance perfume product",
    ],
}


class ClipNoteDataset(Dataset):
    """CSV(image_path, label) based dataset for CLIP evaluation."""

    def __init__(self, csv_path, label2idx, preprocess):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.label2idx = label2idx
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.resolve_path(row["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"[CLIP Dataset] image load failed: {image_path} -> {exc}")
            image = Image.new("RGB", (224, 224))
        label = self.label2idx[row["label"]]
        return self.preprocess(image), label

    def resolve_path(self, image_path):
        image_path = os.path.normpath(str(image_path).replace("\\", os.sep))
        if os.path.isabs(image_path):
            return image_path
        return os.path.join(PROJECT_ROOT, image_path)


def get_device(preferred):
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def build_text_features(model, tokenizer, labels, device):
    """Average multiple prompts per class into one normalized text feature."""
    text_features = []

    for label in labels:
        prompts = NOTE_PROMPTS[label]
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

        class_feature = features.mean(dim=0)
        class_feature = class_feature / class_feature.norm()
        text_features.append(class_feature)

    return torch.stack(text_features, dim=0)


@torch.no_grad()
def evaluate_clip(model, loader, text_features, device, log_every=20):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_scores.append(logits.float().cpu().numpy())
        if log_every and step % log_every == 0:
            print(f"[CLIP zero-shot] batch {step}/{len(loader)}")

    return np.array(all_preds), np.array(all_labels), np.concatenate(all_scores, axis=0)


@torch.no_grad()
def extract_image_features(model, loader, device, log_prefix="split", log_every=20):
    model.eval()
    all_features = []
    all_labels = []

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.float().cpu().numpy())
        all_labels.extend(labels.numpy())
        if log_every and step % log_every == 0:
            print(f"[CLIP features:{log_prefix}] batch {step}/{len(loader)}")

    features = np.concatenate(all_features, axis=0)
    features = np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    features = features.astype(np.float32)
    return features, np.array(all_labels)


def load_or_extract_features(model, dataset, loader, device, args, split_name):
    cache_dir = Path(args.feature_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = Path(PROJECT_ROOT) / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_key = f"{args.model}_{args.pretrained}".replace("/", "_")
    cache_path = cache_dir / f"{model_key}_{args.data_dir.replace('/', '_')}_{split_name}.npz"

    if args.cache_features and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        print(f"[CLIP features:{split_name}] cache load: {cache_path}")
        return cached["features"], cached["labels"]

    features, labels = extract_image_features(
        model, loader, device, log_prefix=split_name, log_every=args.log_every
    )
    if args.cache_features:
        np.savez_compressed(cache_path, features=features, labels=labels)
        print(f"[CLIP features:{split_name}] cache save: {cache_path}")
    return features, labels


def train_linear_probe(train_features, train_labels, val_features, val_labels):
    """Tune a small ridge-classifier head on validation accuracy."""
    best_model = None
    best_alpha = None
    best_acc = -1.0

    for alpha in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        clf = RidgeClassifier(
            alpha=alpha,
            class_weight="balanced",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            clf.fit(train_features, train_labels)
            val_preds = clf.predict(val_features)
        val_acc = float(np.mean(val_preds == val_labels))

        if val_acc > best_acc:
            best_acc = val_acc
            best_alpha = alpha
            best_model = clf

    print(f"Linear probe best alpha: {best_alpha} | Val Accuracy: {best_acc * 100:.2f}%")
    return best_model, best_alpha, best_acc


def evaluate_linear_probe(model, preprocess, label2idx, device, args):
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    train_split = args.train_split
    train_csv = os.path.join(data_dir, f"{train_split}.csv")
    if not os.path.exists(train_csv):
        train_split = "train"
        train_csv = os.path.join(data_dir, "train.csv")

    train_ds = ClipNoteDataset(train_csv, label2idx, preprocess)
    val_ds = ClipNoteDataset(os.path.join(data_dir, "val.csv"), label2idx, preprocess)
    eval_ds = ClipNoteDataset(os.path.join(data_dir, f"{args.split}.csv"), label2idx, preprocess)

    train_loader = make_loader(train_ds, args)
    val_loader = make_loader(val_ds, args)
    eval_loader = make_loader(eval_ds, args)

    print(f"[CLIP] train split: {train_split} ({len(train_ds)} rows)")
    train_features, train_labels = load_or_extract_features(
        model, train_ds, train_loader, device, args, train_split
    )
    val_features, val_labels = load_or_extract_features(
        model, val_ds, val_loader, device, args, "val"
    )
    eval_features, eval_labels = load_or_extract_features(
        model, eval_ds, eval_loader, device, args, args.split
    )

    clf, best_alpha, _ = train_linear_probe(train_features, train_labels, val_features, val_labels)

    if args.split == "test":
        combined_features = np.concatenate([train_features, val_features], axis=0)
        combined_labels = np.concatenate([train_labels, val_labels], axis=0)
        clf = RidgeClassifier(
            alpha=best_alpha,
            class_weight="balanced",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            clf.fit(combined_features, combined_labels)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        preds = clf.predict(eval_features)
        scores = clf.decision_function(eval_features)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return preds, eval_labels, scores, best_alpha


def make_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )


def top_k_accuracy_from_scores(scores, labels, k=3):
    top_k_preds = np.argsort(scores, axis=1)[:, -k:]
    correct = np.array([labels[i] in top_k_preds[i] for i in range(len(labels))])
    return float(correct.mean())


def softmax_np(scores):
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def save_confusion_matrix(y_true, y_pred, target_names, output_dir, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for row in range(cm_norm.shape[0]):
        for col in range(cm_norm.shape[1]):
            value = cm_norm[row, col]
            color = "white" if value > 0.5 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks(range(len(target_names)), target_names, rotation=35, ha="right")
    ax.set_yticks(range(len(target_names)), target_names)
    ax.set_title("CLIP Normalized Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    save_path = output_dir / filename
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[CLIP] confusion matrix saved: {save_path}")


def save_per_class_accuracy(y_true, y_pred, target_names, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(target_names, per_class_acc, color="#4c78a8", edgecolor="white")
    ax.axhline(per_class_acc.mean(), color="#d62728", linestyle="--", label=f"Mean: {per_class_acc.mean():.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("CLIP Per-Class Accuracy")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    save_path = output_dir / "per_class_accuracy.png"
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[CLIP] per-class accuracy saved: {save_path}")


def save_metric_bar(metrics, output_dir, title="CLIP Test Metrics"):
    names = ["accuracy", "macro_f1", "top3_accuracy"]
    values = [metrics[name] for name in names]
    labels = ["Accuracy", "Macro F1", "Top-3 Accuracy"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"], edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}", ha="center", fontsize=10)
    fig.tight_layout()
    save_path = output_dir / "metrics_bar.png"
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[CLIP] metric graph saved: {save_path}")


def save_predictions(eval_df, y_true, y_pred, scores, idx2label, output_dir):
    probs = softmax_np(scores)
    rows = []
    for i, row in eval_df.reset_index(drop=True).iterrows():
        prob = probs[i]
        top3_indices = np.argsort(prob)[-3:][::-1]
        rows.append({
            "image_path": row.get("image_path", ""),
            "name": row.get("name", ""),
            "brand": row.get("brand", ""),
            "actual": idx2label[int(y_true[i])],
            "predicted": idx2label[int(y_pred[i])],
            "confidence": float(prob[int(y_pred[i])]),
            "top3_predictions": " | ".join(idx2label[int(idx)] for idx in top3_indices),
            "top3_correct": bool(int(y_true[i]) in top3_indices),
        })
    save_path = output_dir / "test_predictions.csv"
    pd.DataFrame(rows).to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[CLIP] predictions saved: {save_path}")


def save_outputs(args, y_true, y_pred, scores, target_names, idx2label, best_alpha=None):
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(PROJECT_ROOT) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "top3_accuracy": top_k_accuracy_from_scores(scores, y_true, k=3),
        "random_baseline": 1.0 / len(target_names),
    }
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)

    save_confusion_matrix(y_true, y_pred, target_names, output_dir)
    save_per_class_accuracy(y_true, y_pred, target_names, output_dir)
    save_metric_bar(metrics, output_dir)

    eval_csv = Path(PROJECT_ROOT) / args.data_dir / f"{args.split}.csv"
    save_predictions(pd.read_csv(eval_csv), y_true, y_pred, scores, idx2label, output_dir)

    metrics_payload = {
        "model": f"CLIP {args.model} ({args.pretrained})",
        "mode": args.mode,
        "split": args.split,
        "data_dir": args.data_dir,
        "train_split": args.train_split,
        "best_alpha": best_alpha,
        **metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_text = (
        f"Model        : {metrics_payload['model']}\n"
        f"Mode         : {args.mode}\n"
        f"Split        : {args.split}\n"
        f"Data         : {args.data_dir}\n"
        f"Train split  : {args.train_split}\n"
        f"Accuracy     : {metrics['accuracy']:.4f}\n"
        f"Macro F1     : {metrics['macro_f1']:.4f}\n"
        f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}\n"
        f"Random Baseline: {metrics['random_baseline']:.4f}\n\n"
        f"{report}"
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    print(f"[CLIP] report saved: {output_dir / 'classification_report.txt'}")
    return metrics


def print_target_status(accuracy):
    low, high = TARGET_ACCURACY
    print("\n" + "-" * 60)
    print(f"  Target Accuracy (note, CLIP): {low * 100:.0f}% ~ {high * 100:.0f}%")
    if accuracy < low:
        print(f"  Result: {accuracy * 100:.2f}%  -> 목표 미달")
    elif accuracy <= high:
        print(f"  Result: {accuracy * 100:.2f}%  -> 목표 범위 도달")
    else:
        print(f"  Result: {accuracy * 100:.2f}%  -> 목표 상한 초과")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="CLIP note classification")
    parser.add_argument("--mode", type=str, default="linear_probe",
                        choices=["zero_shot", "linear_probe"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "All"))
    parser.add_argument("--train_split", type=str, default="train_aug")
    parser.add_argument("--model", type=str, default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "clip_linear_probe"))
    parser.add_argument("--feature_cache_dir", type=str, default=os.path.join("results", "clip_linear_probe", "features"))
    parser.add_argument("--no_cache_features", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="CLIP 평가 장치 (기본: cpu, MPS 수치 이슈 회피)")
    args = parser.parse_args()
    args.cache_features = not args.no_cache_features

    try:
        import open_clip
    except ImportError as exc:
        raise SystemExit(
            "open_clip_torch is required. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    train_csv = os.path.join(PROJECT_ROOT, args.data_dir, "train.csv")
    split_csv = os.path.join(PROJECT_ROOT, args.data_dir, f"{args.split}.csv")

    train_df = pd.read_csv(train_csv)
    unique_labels = set(train_df["label"].unique())
    labels = [label for label in NOTE_CLASS_ORDER if label in unique_labels]
    labels.extend(label for label in sorted(unique_labels) if label not in labels)
    label2idx = {label: i for i, label in enumerate(labels)}
    idx2label = {i: label for label, i in label2idx.items()}

    device = get_device(args.device)
    print(f"\nDevice: {device}")
    print(f"Model: CLIP {args.model} ({args.pretrained})")
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Classes: {len(labels)}  ({', '.join(labels)})")
    print(f"Target Accuracy (CLIP note): {TARGET_ACCURACY[0] * 100:.0f}% ~ {TARGET_ACCURACY[1] * 100:.0f}%")

    cache_dir = os.path.join(PROJECT_ROOT, ".torch_cache", "open_clip")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
        cache_dir=cache_dir,
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    if args.mode == "zero_shot":
        dataset = ClipNoteDataset(split_csv, label2idx, preprocess)
        loader = make_loader(dataset, args)
        text_features = build_text_features(model, tokenizer, labels, device)
        preds, true_labels, scores = evaluate_clip(
            model, loader, text_features, device, log_every=args.log_every
        )
        best_alpha = None
    else:
        preds, true_labels, scores, best_alpha = evaluate_linear_probe(
            model, preprocess, label2idx, device, args
        )
    target_names = [idx2label[i] for i in range(len(idx2label))]
    metrics = save_outputs(args, true_labels, preds, scores, target_names, idx2label, best_alpha)

    print("\n" + "=" * 60)
    print("  CLIP Note 결과")
    print("=" * 60)
    print(classification_report(true_labels, preds, target_names=target_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, preds, labels=list(range(len(target_names)))))
    print(f"\n{args.split.title()} Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"{args.split.title()} Macro F1: {metrics['macro_f1'] * 100:.2f}%")
    print(f"{args.split.title()} Top-3 Accuracy: {metrics['top3_accuracy'] * 100:.2f}%")
    print_target_status(metrics["accuracy"])


if __name__ == "__main__":
    main()
