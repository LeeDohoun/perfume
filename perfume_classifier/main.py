"""
main.py
-------
학습 / 평가 / 추론을 통합하는 진입점.

사용법
------
# 완전 처음부터 한 번에 (재분할 → 증강 → 학습 → 평가)
python main.py --mode full

# 증강 → 학습 → 평가 (resplit 생략)
python main.py --mode train_eval

# train_aug.csv 없으면 augment 자동 실행 후 학습
python main.py --mode train

# 테스트셋 평가만
python main.py --mode eval

# 검증셋 평가
python main.py --mode eval --split val

# 특정 체크포인트로 평가
python main.py --mode eval --ckpt checkpoints/stage2_best.pth
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
        choices=["full", "resplit", "train", "eval", "train_eval"],
        help=(
            "실행 모드 (기본값: train_eval)\n"
            "  full       : resplit → train → eval\n"
            "  resplit    : 데이터 재분할만\n"
            "  train      : 학습만\n"
            "  eval       : 평가만\n"
            "  train_eval : 학습 + 평가"
        ),
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
        choices=["efficientnet_v2_s", "efficientnet_b3", "efficientnet_b0", "mobilenet_v3_large"],
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
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="resplit / full 모드: train 비율 (기본 0.8)",
    )

    return parser.parse_args()


def apply_cli_overrides(args):
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
    print("  향수 Note 분류 시스템")
    print("=" * 55)
    print(f"  Backbone    : {cfg.model.backbone}")
    print(f"  Image size  : {cfg.model.image_size}x{cfg.model.image_size}")
    print(f"  Classes     : {cfg.cls.note_classes}")
    print(f"  Text input  : {cfg.text.use_text} | columns={cfg.text.columns} | max_len={cfg.text.max_len}")
    print(f"  Stage1      : {cfg.train.stage1_epochs} epochs | lr={cfg.train.stage1_lr}")
    print(f"  Stage2      : {cfg.train.stage2_epochs} epochs | lr={cfg.train.stage2_lr}")
    print(f"  Batch size  : {cfg.train.batch_size}")
    print(f"  AMP         : {cfg.train.use_amp}")
    print(f"  WR Sampler  : {cfg.train.use_weighted_sampler}")
    aug_exists = os.path.exists(cfg.path.train_aug_csv)
    train_csv_used = cfg.path.train_aug_csv if aug_exists else cfg.path.train_csv
    print(f"  Train CSV   : {os.path.basename(train_csv_used)}"
          + ("  [증강본]" if aug_exists else "  [원본, augment 실행 시 증강본 사용]"))
    print(f"  Val CSV     : {cfg.path.val_csv}")
    print(f"  Test CSV    : {cfg.path.test_csv}")
    print(f"  Image root  : {cfg.path.image_root}")
    print(f"  Checkpoint  : {cfg.path.checkpoint_dir}")
    print(f"  Results     : {cfg.path.result_dir}")
    print("=" * 55 + "\n")


def ensure_directories():
    for d in [cfg.path.checkpoint_dir, cfg.path.log_dir, cfg.path.result_dir]:
        os.makedirs(d, exist_ok=True)


def _run_resplit(args):
    from resplit import run_resplit
    seed = args.seed if args.seed is not None else cfg.train.seed
    run_resplit(train_ratio=args.train, seed=seed)


def _run_train():
    from train import run_training
    run_training()


def _run_eval(args):
    from evaluate import run_evaluation
    ckpt = args.ckpt or os.path.join(cfg.path.checkpoint_dir, "stage2_best.pth")
    metrics = run_evaluation(checkpoint_path=ckpt, split=args.split)
    print("\n[최종 성능 요약]")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")


def main():
    args = parse_args()
    apply_cli_overrides(args)
    print_config_summary()
    ensure_directories()

    if args.mode == "full":
        _run_resplit(args)
        _run_train()
        _run_eval(args)

    elif args.mode == "resplit":
        _run_resplit(args)

    elif args.mode == "train":
        _run_train()

    elif args.mode == "train_eval":
        _run_train()
        _run_eval(args)

    elif args.mode == "eval":
        _run_eval(args)


if __name__ == "__main__":
    main()
