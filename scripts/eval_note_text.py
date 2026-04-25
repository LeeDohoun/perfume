"""
Evaluate the saved Note metadata text classifier.

Usage:
  python scripts/eval_note_text.py
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from train_note_text import load_split, print_target_status


def main():
    parser = argparse.ArgumentParser(description="저장된 Note 텍스트 모델 평가")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "note_text", "best_model.joblib"))
    args = parser.parse_args()

    bundle = joblib.load(args.checkpoint)
    model = bundle["model"]
    text, labels = load_split(args.split)
    preds = model.predict(text)
    accuracy = float(np.mean(preds == labels))

    print(f"\nModel: TF-IDF + LogisticRegression")
    print(f"Input: {bundle.get('input', 'name + brand + description')}")
    print(f"Leakage guard: {bundle.get('leakage_guard', 'notes column is not used')}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")

    print("\n" + "=" * 60)
    print("  Note Metadata Text 결과")
    print("=" * 60)
    print(classification_report(labels, preds, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds, labels=model.classes_))
    print(f"\n{args.split.title()} Accuracy: {accuracy * 100:.2f}%")
    print_target_status(accuracy)


if __name__ == "__main__":
    main()
