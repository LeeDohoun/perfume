"""
main.py
-------
학습 / 평가 / 추론을 통합하는 진입점.

사용법
------
# 전체 학습 실행
python main.py --mode train

# 테스트셋 평가
python main.py --mode eval

# 검증셋 평가
python main.py --mode eval --split val

# 특정 체크포인트로 평가
python main.py --mode eval --ckpt checkpoints/stage2_best.pth

# 학습 후 바로 평가까지
python main.py --mode train_eval
"""

import argparse
import os

from config import get_config

cfg = get_config()


def parse_args():
    parser = argparse.ArgumentParser(description="향수 Note 분류 파이프라인")

    parser.add_argument(
        "--mode",
        type=str,
        default="train_eval",
        choices=["train", "eval", "train_eval"],
        help="실행 모드 (기본값: train_eval)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="평가 시 사용할 데이터 분할 (기본값: test)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="평가에 사용할 체크포인트 경로 (기본값: checkpoints/stage2_best.pth)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["efficientnet_b0", "mobilenet_v3_large"],
        help="사용할 backbone (config.py 덮어쓰기)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="배치 크기 (config.py 덮어쓰기)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="랜덤 시드 (config.py 덮어쓰기)",
    )

    return parser.parse_args()


def apply_cli_overrides(args):
    """CLI 인수로 config 값을 동적으로 덮어씁니다."""
    if args.backbone:
        cfg.model.backbone = args.backbone
        print(f"[Config] backbone → {args.backbone}")
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
        print(f"[Config] batch_size → {args.batch_size}")
    if args.seed:
        cfg.train.seed = args.seed
        print(f"[Config] seed → {args.seed}")


def print_config_summary():
    print("\n" + "=" * 55)
    print("  향수 Note 분류 시스템 — EfficientNet-B0")
    print("=" * 55)
    print(f"  Backbone    : {cfg.model.backbone}")
    print(f"  Image size  : {cfg.model.image_size}x{cfg.model.image_size}")
    print(f"  Classes     : {cfg.cls.note_classes}")
    print(f"  Stage1      : {cfg.train.stage1_epochs} epochs | lr={cfg.train.stage1_lr}")
    print(f"  Stage2      : {cfg.train.stage2_epochs} epochs | lr={cfg.train.stage2_lr}")
    print(f"  Batch size  : {cfg.train.batch_size}")
    print(f"  AMP         : {cfg.train.use_amp}")
    print(f"  WR Sampler  : {cfg.train.use_weighted_sampler}")
    print(f"  Train CSV   : {cfg.path.train_csv}")
    print(f"  Val CSV     : {cfg.path.val_csv}")
    print(f"  Test CSV    : {cfg.path.test_csv}")
    print(f"  Image root  : {cfg.path.image_root}")
    print(f"  Checkpoint  : {cfg.path.checkpoint_dir}")
    print(f"  Results     : {cfg.path.result_dir}")
    print("=" * 55 + "\n")


def ensure_directories():
    for d in [cfg.path.checkpoint_dir, cfg.path.log_dir, cfg.path.result_dir]:
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    apply_cli_overrides(args)
    print_config_summary()
    ensure_directories()

    if args.mode in ("train", "train_eval"):
        from train import run_training
        run_training()

    if args.mode in ("eval", "train_eval"):
        from evaluate import run_evaluation

        ckpt = args.ckpt
        if ckpt is None:
            ckpt = os.path.join(cfg.path.checkpoint_dir, "stage2_best.pth")

        metrics = run_evaluation(checkpoint_path=ckpt, split=args.split)
        print("\n[최종 성능 요약]")
        for k, v in metrics.items():
            print(f"  {k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()
