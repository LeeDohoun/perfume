"""
dataset.py
----------
향수병 이미지 데이터셋 정의.

CSV 컬럼 구조:
  image_path | label | name | brand | notes | image_url

- PerfumeDataset : torch.utils.data.Dataset 구현
- get_transforms : train / val·test 용 transform 반환
- make_dataloader : DataLoader 생성 (WeightedRandomSampler 포함)
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBU_AVAILABLE = True
except ImportError:
    _ALBU_AVAILABLE = False

from config import get_config

cfg = get_config()


# ──────────────────────────────────────────────
# Transform
# ──────────────────────────────────────────────

class _AlbuWrapper:
    """albumentations Compose를 torchvision 스타일 callable로 래핑합니다.
    PIL Image를 입력받아 torch.Tensor를 반환합니다."""

    def __init__(self, transform: "A.Compose"):
        self.transform = transform

    def __call__(self, image) -> torch.Tensor:
        img_np = np.array(image)
        return self.transform(image=img_np)["image"]


def get_transforms(split: str):
    """
    split: 'train' | 'val' | 'test'

    Returns
    -------
    callable — torchvision T.Compose 또는 _AlbuWrapper
    """
    aug = cfg.aug
    mean, std = aug.mean, aug.std
    size = cfg.model.image_size

    if split == "train" and aug.use_albumentations:
        if not _ALBU_AVAILABLE:
            raise ImportError(
                "use_albumentations=True이지만 albumentations가 없습니다.\n"
                "  pip install albumentations"
            )
        transform = A.Compose([
            A.Resize(size + 32, size + 32),
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=aug.random_horizontal_flip),
            # ShiftScaleRotate = Rotate + 약간의 스케일·이동 변화
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=aug.random_rotation, p=0.5,
            ),
            A.ColorJitter(
                brightness=aug.color_jitter["brightness"],
                contrast=aug.color_jitter["contrast"],
                saturation=aug.color_jitter["saturation"],
                hue=aug.color_jitter["hue"],
                p=0.8,
            ),
            # 향수병 라벨·유리 질감 강조
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            # 촬영 블러 시뮬레이션
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            # 랜덤 패치 드롭 (RandomErasing 강화판)
            A.CoarseDropout(
                max_holes=8, max_height=size // 8, max_width=size // 8,
                min_holes=1, min_height=size // 16, min_width=size // 16,
                fill_value=0, p=0.3,
            ),
            # 렌즈 왜곡 시뮬레이션
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        return _AlbuWrapper(transform)

    elif split == "train":  # albumentations 미설치 시 torchvision fallback
        return T.Compose([
            T.Resize((size + 32, size + 32)),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(p=aug.random_horizontal_flip),
            T.RandomRotation(degrees=aug.random_rotation),
            T.ColorJitter(**aug.color_jitter),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=aug.random_erasing, scale=(0.02, 0.2)),
        ])

    else:  # val / test
        return T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class PerfumeDataset(Dataset):
    """
    향수병 이미지 + Note 레이블 데이터셋.

    Parameters
    ----------
    csv_path   : train / val / test CSV 파일 경로
    split      : 'train' | 'val' | 'test'
    image_root : image_path 컬럼의 prefix. 절대 경로면 "" 사용.
    transform  : None이면 get_transforms(split) 자동 적용
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        image_root: str = "",
        transform: Optional[T.Compose] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.split = split
        self.image_root = image_root
        self.transform = transform or get_transforms(split)

        # 클래스 → 인덱스 매핑
        self.classes = cfg.cls.note_classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # label 컬럼 정제: classes에 없는 샘플 제거
        valid_mask = self.df["label"].isin(self.class_to_idx)
        dropped = (~valid_mask).sum()
        if dropped > 0:
            print(f"[Dataset:{split}] '{csv_path}' - 알 수 없는 label {dropped}개 제거됨")
        self.df = self.df[valid_mask].reset_index(drop=True)

        self.labels = self.df["label"].map(self.class_to_idx).values

    # ── 기본 메서드 ──────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row["image_path"])

        # 이미지 로드
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # 이미지 로드 실패 시 검은 이미지로 대체 (학습 중단 방지)
            print(f"[Dataset] 이미지 로드 실패: {img_path} → {e}")
            image = Image.new("RGB", (cfg.model.image_size, cfg.model.image_size))

        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])
        return image, label

    # ── 내부 유틸 ────────────────────────────────

    def _resolve_path(self, image_path: str) -> str:
        """image_root와 image_path를 합쳐 절대 경로를 반환합니다."""
        if self.image_root and not os.path.isabs(image_path):
            return os.path.join(self.image_root, image_path)
        return image_path

    def get_class_weights(self) -> torch.Tensor:
        """
        WeightedRandomSampler에 사용할 클래스 가중치를 계산합니다.
        희소 클래스에 더 높은 가중치를 부여합니다.
        """
        counts = np.bincount(self.labels, minlength=len(self.classes))
        counts = np.maximum(counts, 1)                     # 0 나누기 방지
        weights = 1.0 / counts
        sample_weights = weights[self.labels]
        return torch.tensor(sample_weights, dtype=torch.float)

    def class_distribution(self) -> dict:
        """클래스별 샘플 수를 딕셔너리로 반환합니다."""
        counts = np.bincount(self.labels, minlength=len(self.classes))
        return {cls: int(cnt) for cls, cnt in zip(self.classes, counts)}


# ──────────────────────────────────────────────
# DataLoader Factory
# ──────────────────────────────────────────────

def make_dataloader(
    csv_path: str,
    split: str,
    image_root: str = "",
    transform: Optional[T.Compose] = None,
) -> DataLoader:
    """
    PerfumeDataset을 감싼 DataLoader를 생성합니다.

    - train split : WeightedRandomSampler(cfg 설정에 따라) 적용
    - val / test  : shuffle=False, 순서 유지
    """
    tc = cfg.train

    dataset = PerfumeDataset(
        csv_path=csv_path,
        split=split,
        image_root=image_root,
        transform=transform,
    )

    if split == "train" and tc.use_weighted_sampler:
        sample_weights = dataset.get_class_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=tc.batch_size,
            sampler=sampler,
            num_workers=tc.num_workers,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=tc.batch_size,
            shuffle=(split == "train"),
            num_workers=tc.num_workers,
            pin_memory=True,
        )

    print(f"[DataLoader:{split}] {len(dataset)}개 샘플 | 분포: {dataset.class_distribution()}")
    return loader


# ──────────────────────────────────────────────
# 빠른 테스트 (python dataset.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from config import cfg

    loader = make_dataloader(
        csv_path=cfg.path.train_csv,
        split="train",
        image_root=cfg.path.image_root,
    )
    images, labels = next(iter(loader))
    print(f"배치 이미지 shape : {images.shape}")   # (B, 3, 224, 224)
    print(f"배치 레이블       : {labels}")
