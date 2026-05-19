"""
Note classification with text metadata.

This fixes the main Note-task issue: perfume-family labels are weakly visible
from bottle images, so the model uses metadata that a product page actually
provides (name, brand, description). It deliberately does not use `notes`,
because `notes` are the source used to create the labels.

Usage:
  python scripts/train_note_text.py
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


TARGET_ACCURACY = (0.45, 0.60)
CANDIDATE_C = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def load_split(split):
    df = pd.read_csv(os.path.join("data", "note", f"{split}.csv"))
    required = ["name", "brand", "description", "label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"data/note/{split}.csv is missing {missing}. "
            "Run `python scripts/prepare_data.py` first."
        )

    text = (
        df["name"].fillna("") + " " +
        df["brand"].fillna("") + " " +
        df["description"].fillna("")
    )
    return text, df["label"]


def build_model(c_value):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=c_value,
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def choose_model(train_text, train_labels, val_text, val_labels):
    best_model = None
    best_c = None
    best_val_acc = -1.0

    for c_value in CANDIDATE_C:
        model = build_model(c_value)
        model.fit(train_text, train_labels)
        val_acc = float(np.mean(model.predict(val_text) == val_labels))
        print(f"C={c_value:<4} | Val Accuracy: {val_acc * 100:.2f}%")

        # Tie-break by smaller C to prefer the simpler model.
        if val_acc > best_val_acc:
            best_model = model
            best_c = c_value
            best_val_acc = val_acc

    return best_model, best_c, best_val_acc


def print_target_status(accuracy):
    low, high = TARGET_ACCURACY
    print("\n" + "-" * 60)
    print(f"  Target Accuracy (note, metadata text): {low * 100:.0f}% ~ {high * 100:.0f}%")
    if accuracy < low:
        print(f"  Result: {accuracy * 100:.2f}%  -> 목표 미달")
    elif accuracy <= high:
        print(f"  Result: {accuracy * 100:.2f}%  -> 목표 범위 도달")
    else:
        print(f"  Result: {accuracy * 100:.2f}%  -> 목표 상한 초과")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Note metadata text classifier")
    parser.add_argument("--output", type=str, default=os.path.join("checkpoints", "note_text", "best_model.joblib"))
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    train_text, train_labels = load_split("train")
    val_text, val_labels = load_split("val")
    test_text, test_labels = load_split("test")

    print("\nModel: TF-IDF + LogisticRegression")
    print("Input: name + brand + description")
    print("Leakage guard: notes column is not used")
    print(f"Train: {len(train_labels)} | Val: {len(val_labels)} | Test: {len(test_labels)}")

    model, best_c, best_val_acc = choose_model(train_text, train_labels, val_text, val_labels)
    preds = model.predict(test_text)
    test_acc = float(np.mean(preds == test_labels))

    print("\n" + "=" * 60)
    print("  Note Metadata Text 결과")
    print("=" * 60)
    print(f"Best C: {best_c}")
    print(f"Best Val Accuracy: {best_val_acc * 100:.2f}%")
    print(classification_report(test_labels, preds, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, preds, labels=model.classes_))
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print_target_status(test_acc)

    if not args.no_save:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        joblib.dump({
            "model": model,
            "best_c": best_c,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "input": "name + brand + description",
            "leakage_guard": "notes column is not used",
        }, args.output)
        print(f"\n모델 저장: {args.output}")


if __name__ == "__main__":
    main()
