"""
resplit.py
----------
data/raw/all_cleaned.csv 를 기준으로 train/val/test를 재분할합니다.
기본 비율: train 65% / val 17.5% / test 17.5%

사용법
------
  python resplit.py                 # 기본 비율
  python resplit.py --train 0.7    # train 비율 직접 지정
  python main.py --mode resplit
  python main.py --mode resplit --train 0.7
"""

import argparse
import os
import sys
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config

cfg = get_config()

RAW_CSV = os.path.join(cfg.path.image_root, "data", "raw", "all_cleaned.csv")


def _print_dist(df: pd.DataFrame, name: str) -> None:
    c = Counter(df["label"])
    total = len(df)
    max_cnt = max(c.values())
    print(f"\n  [{name}]  {total}개  (클래스 {len(c)}개)")
    for cls, cnt in sorted(c.items()):
        bar = "#" * (cnt * 20 // max_cnt)
        print(f"    {cls:20s}: {cnt:4d}  ({cnt/total*100:4.1f}%)  {bar}")


def run_resplit(train_ratio: float = 0.65, seed: int = 42) -> None:
    val_ratio = (1.0 - train_ratio) / 2

    df = pd.read_csv(RAW_CSV)
    print(f"소스: {RAW_CSV}  ({len(df)}행)")
    print(f"분할 비율: train {train_ratio:.0%} / val {val_ratio:.1%} / test {val_ratio:.1%}")
    _print_dist(df, "전체")

    train_df, temp_df = train_test_split(
        df,
        test_size=round(1.0 - train_ratio, 6),
        stratify=df["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=seed,
    )

    _print_dist(train_df, "train")
    _print_dist(val_df,   "val  ")
    _print_dist(test_df,  "test ")

    min_val  = min(Counter(val_df["label"]).values())
    min_test = min(Counter(test_df["label"]).values())
    print(f"\n  val  최소 클래스: {min_val}개")
    print(f"  test 최소 클래스: {min_test}개")

    out_dir = os.path.dirname(cfg.path.train_csv)
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(cfg.path.train_csv, index=False)
    val_df.to_csv(cfg.path.val_csv,     index=False)
    test_df.to_csv(cfg.path.test_csv,   index=False)

    print(f"\n저장 완료")
    print(f"  train : {cfg.path.train_csv}  ({len(train_df)}행)")
    print(f"  val   : {cfg.path.val_csv}  ({len(val_df)}행)")
    print(f"  test  : {cfg.path.test_csv}  ({len(test_df)}행)")
    print(f"\n* train 증강이 필요하면 augment_offline.py를 다시 실행하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터셋 재분할 (소스: all_cleaned.csv)")
    parser.add_argument("--train", type=float, default=0.65, help="train 비율 (기본 0.65)")
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    run_resplit(train_ratio=args.train, seed=args.seed)
