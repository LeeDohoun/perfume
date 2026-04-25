"""
Evaluate a saved EfficientNet-B0 checkpoint.

Usage:
  python eval_checkpoint.py --task note
  python eval_checkpoint.py --task brand
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from train import (
    PerfumeDataset,
    build_model,
    get_transforms,
    print_target_status,
    test_evaluate,
)


def get_device(preferred):
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="저장된 EfficientNet-B0 체크포인트 평가")
    parser.add_argument("--task", type=str, required=True, choices=["note", "brand"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint_path = args.checkpoint or os.path.join("checkpoints", args.task, "best_model.pth")
    csv_path = os.path.join("data", args.task, f"{args.split}.csv")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint["num_classes"]
    idx2label = checkpoint["idx2label"]

    _, eval_tf = get_transforms()
    dataset = PerfumeDataset(csv_path, eval_tf)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"\nDevice: {device}")
    print(f"Task: {args.task.upper()} | Split: {args.split}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {len(dataset)}")

    _, _, accuracy = test_evaluate(model, loader, device, idx2label)
    print_target_status(args.task, accuracy)


if __name__ == "__main__":
    main()
