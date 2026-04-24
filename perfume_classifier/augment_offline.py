"""
augment_offline.py
------------------
소수 클래스(Sweet, Spicy) 오프라인 증강 스크립트.
train CSV만 처리합니다. val / test는 원본 그대로 유지합니다.

증강 배수
---------
- Sweet : 원본 1장당 1장 추가 → 총 2배
- Spicy : 원본 1장당 2장 추가 → 총 3배

저장 위치
---------
  perfume_images/augmented/{split}_aug{i}_{원본파일명}

  split prefix(train_, val_, test_)를 붙여 파일명 충돌을 방지합니다.

사용법
------
  python augment_offline.py          # 직접 실행
  python main.py --mode augment      # main.py 통해 실행
"""

import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config

try:
    import albumentations as A
except ImportError:
    raise ImportError("albumentations가 필요합니다: pip install albumentations")

cfg = get_config()

# ──────────────────────────────────────────────
# 증강 설정
# ──────────────────────────────────────────────

# {클래스명: 원본 1장당 추가할 장수}
AUGMENT_COPIES = {
    "Sweet": 1,   # 총 2배
    "Spicy": 2,   # 총 3배
}

AUG_SUBDIR = os.path.join("perfume_images", "augmented")


# ──────────────────────────────────────────────
# Transform 정의
# ──────────────────────────────────────────────

def _build_train_transform(size: int) -> A.Compose:
    """train용: 강한 augmentation."""
    return A.Compose([
        A.Resize(size + 32, size + 32),
        A.RandomCrop(size, size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1,
            rotate_limit=20, p=0.7,
        ),
        A.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.05, p=0.8,
        ),
        A.CLAHE(clip_limit=2.0, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CoarseDropout(
            max_holes=4,
            max_height=size // 8, max_width=size // 8,
            min_holes=1,
            min_height=size // 16, min_width=size // 16,
            fill_value=0, p=0.3,
        ),
    ])




# ──────────────────────────────────────────────
# 이미지 저장 헬퍼
# ──────────────────────────────────────────────

def _augment_and_save(src_path: str, dst_path: str, transform: A.Compose) -> bool:
    try:
        img_np = np.array(Image.open(src_path).convert("RGB"))
        result = transform(image=img_np)
        Image.fromarray(result["image"]).save(dst_path, quality=95)
        return True
    except Exception as e:
        print(f"  [경고] 처리 실패: {src_path} → {e}")
        return False


# ──────────────────────────────────────────────
# 단일 CSV 증강 처리
# ──────────────────────────────────────────────

def _augment_csv(
    split: str,
    csv_path: str,
    image_root: str,
    aug_abs_dir: str,
    transform: A.Compose,
) -> None:
    """split 하나의 CSV를 읽어 증강 후 저장합니다."""
    print(f"\n{'=' * 55}")
    print(f"  [{split}] {csv_path}")
    print(f"{'=' * 55}")

    df = pd.read_csv(csv_path)

    # 이미 증강된 행이 있으면 중복 실행 방지
    already = df["image_path"].str.contains("augmented", na=False)
    if already.any():
        print(f"  이미 증강 데이터 {already.sum()}건이 있어 건너뜁니다.")
        print(f"  재실행하려면 CSV에서 augmented 행을 제거하세요.")
        return

    new_rows = []
    stats    = {}

    for label, n_copies in AUGMENT_COPIES.items():
        subset = df[df["label"] == label]
        if subset.empty:
            continue

        success_count = 0
        print(f"\n  [{label}] {len(subset)}장 → 목표 {len(subset) * (1 + n_copies)}장")

        for _, row in subset.iterrows():
            src_path  = os.path.join(image_root, row["image_path"])
            orig_stem = os.path.splitext(os.path.basename(row["image_path"]))[0]
            orig_ext  = os.path.splitext(row["image_path"])[1] or ".jpg"

            for i in range(1, n_copies + 1):
                # split prefix로 파일명 충돌 방지
                filename = f"{split}_aug{i}_{orig_stem}{orig_ext}"
                dst_path = os.path.join(aug_abs_dir, filename)
                rel_path = os.path.join(AUG_SUBDIR, filename)

                ok = _augment_and_save(src_path, dst_path, transform)
                if ok:
                    new_row = row.copy()
                    new_row["image_path"] = rel_path
                    new_row["image_url"]  = ""
                    new_rows.append(new_row)
                    success_count += 1

        stats[label] = (len(subset), success_count)
        print(f"    완료: {success_count}장 저장")

    if not new_rows:
        print("  저장된 이미지가 없습니다. 원본 경로를 확인하세요.")
        return

    updated_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    updated_df.to_csv(csv_path, index=False)

    print(f"\n  요약")
    for label, (orig, added) in stats.items():
        print(f"    {label:20s}: {orig}장 → {orig + added}장 (+{added})")
    print(f"  CSV 행 수: {len(df)} → {len(updated_df)}")


# ──────────────────────────────────────────────
# 전체 실행 진입점
# ──────────────────────────────────────────────

def run_offline_augmentation() -> None:
    image_root  = cfg.path.image_root
    img_size    = cfg.model.image_size

    aug_abs_dir = os.path.join(image_root, AUG_SUBDIR)
    os.makedirs(aug_abs_dir, exist_ok=True)

    transform = _build_train_transform(img_size)
    _augment_csv("train", cfg.path.train_csv, image_root, aug_abs_dir, transform)

    print(f"\n[augment] 완료. 저장 위치: {aug_abs_dir}")


# ──────────────────────────────────────────────
# 직접 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    run_offline_augmentation()
