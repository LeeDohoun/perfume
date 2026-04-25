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
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, ".torch_cache"))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))

TARGET_ACCURACY = (0.45, 0.60)

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
        self.df = pd.read_csv(csv_path)
        self.label2idx = label2idx
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.normpath(str(row["image_path"]).replace("\\", os.sep))
        image = Image.open(image_path).convert("RGB")
        label = self.label2idx[row["label"]]
        return self.preprocess(image), label


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
def evaluate_clip(model, loader, text_features, device):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return all_preds, all_labels, accuracy


@torch.no_grad()
def extract_image_features(model, loader, device):
    model.eval()
    all_features = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.float().cpu().numpy())
        all_labels.extend(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    features = np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    features = features.astype(np.float64)
    return features, np.array(all_labels)


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
    train_ds = ClipNoteDataset(os.path.join("data", "note", "train.csv"), label2idx, preprocess)
    val_ds = ClipNoteDataset(os.path.join("data", "note", "val.csv"), label2idx, preprocess)
    eval_ds = ClipNoteDataset(os.path.join("data", "note", f"{args.split}.csv"), label2idx, preprocess)

    train_loader = make_loader(train_ds, args)
    val_loader = make_loader(val_ds, args)
    eval_loader = make_loader(eval_ds, args)

    train_features, train_labels = extract_image_features(model, train_loader, device)
    val_features, val_labels = extract_image_features(model, val_loader, device)
    eval_features, eval_labels = extract_image_features(model, eval_loader, device)

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
    accuracy = float(np.mean(preds == eval_labels))
    return preds, eval_labels, accuracy


def make_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )


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
    parser.add_argument("--model", type=str, default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="CLIP 평가 장치 (기본: cpu, MPS 수치 이슈 회피)")
    args = parser.parse_args()

    try:
        import open_clip
    except ImportError as exc:
        raise SystemExit(
            "open_clip_torch is required. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    train_csv = os.path.join("data", "note", "train.csv")
    split_csv = os.path.join("data", "note", f"{args.split}.csv")

    train_df = pd.read_csv(train_csv)
    labels = sorted(train_df["label"].unique())
    label2idx = {label: i for i, label in enumerate(labels)}
    idx2label = {i: label for label, i in label2idx.items()}

    device = get_device(args.device)
    print(f"\nDevice: {device}")
    print(f"Model: CLIP {args.model} ({args.pretrained})")
    print(f"Mode: {args.mode}")
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
        preds, true_labels, accuracy = evaluate_clip(model, loader, text_features, device)
    else:
        preds, true_labels, accuracy = evaluate_linear_probe(
            model, preprocess, label2idx, device, args
        )
    target_names = [idx2label[i] for i in range(len(idx2label))]

    print("\n" + "=" * 60)
    print("  CLIP Note 결과")
    print("=" * 60)
    print(classification_report(true_labels, preds, target_names=target_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, preds))
    print(f"\n{args.split.title()} Accuracy: {accuracy * 100:.2f}%")
    print_target_status(accuracy)


if __name__ == "__main__":
    main()
