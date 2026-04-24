"""
config.py
---------
프로젝트 전반의 하이퍼파라미터 및 경로 설정.
모든 다른 모듈은 이 파일에서 설정을 import해서 사용합니다.
"""

import os
from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class PathConfig:
    train_csv:  str = os.path.join(BASE_DIR, "..", "data","All", "train.csv")
    val_csv:    str = os.path.join(BASE_DIR, "..", "data","All", "val.csv")
    test_csv:   str = os.path.join(BASE_DIR, "..", "data","All", "test.csv")

    # 이미지 루트 디렉터리
    # image_path 컬럼이 절대경로라면 "" 로 두세요
    image_root: str = os.path.join(BASE_DIR, "..")

    # 체크포인트 / 로그 / 결과 저장 디렉터리
    checkpoint_dir: str = os.path.join(BASE_DIR, "..", "checkpoints")
    log_dir:        str = os.path.join(BASE_DIR, "..", "logs")
    result_dir:     str = os.path.join(BASE_DIR, "..", "results")


# ──────────────────────────────────────────────
# 클래스 설정
# ──────────────────────────────────────────────
@dataclass
class ClassConfig:
    # Note 분류 클래스 (6개)
    note_classes: List[str] = field(default_factory=lambda: [
        "Floral", "Woody", "Amber_Oriental", "Citrus", "Sweet", "Spicy"
    ])

    # Brand 분류 태스크를 추가하려면 True로 변경
    use_brand_task: bool = False


# ──────────────────────────────────────────────
# 모델 설정
# ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    # 사용할 백본: "efficientnet_b0" | "mobilenet_v3_large"
    backbone: str = "mobilenet_v3_large"

    # ImageNet pretrained 가중치 사용 여부
    pretrained: bool = True

    # 분류 Head Dropout
    dropout: float = 0.5

    # 입력 이미지 크기
    image_size: int = 224


# ──────────────────────────────────────────────
# 2단계 학습 설정
# ──────────────────────────────────────────────
@dataclass
class TrainConfig:
    # ── Stage 1: Head만 학습 (Backbone Freeze) ──
    stage1_epochs: int   = 5
    stage1_lr:     float = 1e-3

    # ── Stage 2: Gradual Unfreeze + Full Fine-tuning ──
    stage2_epochs: int   = 25
    stage2_lr:     float = 1e-4

    # 공통
    batch_size:      int   = 32
    num_workers:     int   = 4
    weight_decay:    float = 5e-4
    label_smoothing: float = 0.05   # CrossEntropyLoss label smoothing

    # CosineAnnealingLR (stage2 기준)
    t_max:   int   = 25
    eta_min: float = 1e-6

    # Early Stopping (stage2 기준)
    patience:  int   = 7
    min_delta: float = 1e-4

    # 클래스 불균형 보정 WeightedRandomSampler 사용 여부
    use_weighted_sampler: bool = True

    # CutMix: 배치 단위 적용 확률 (0.0 = 비활성화)
    cutmix_prob:  float = 0.5
    cutmix_alpha: float = 1.0   # Beta 분포 파라미터

    # 재현성 시드
    seed: int = 42

    # Mixed Precision (GPU 환경 권장)
    use_amp: bool = False


# ──────────────────────────────────────────────
# Augmentation 설정
# ──────────────────────────────────────────────
@dataclass
class AugConfig:
    random_horizontal_flip: float = 0.5
    color_jitter: dict = field(default_factory=lambda: {
        "brightness": 0.2,
        "contrast":   0.2,
        "saturation": 0.2,
        "hue":        0.05,
    })
    random_rotation: int   = 15
    random_erasing:  float = 0.2   # RandomErasing probability

    # albumentations 강화 augmentation 사용 여부 (pip install albumentations 필요)
    use_albumentations: bool = True

    # ImageNet Normalize 값
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


# ──────────────────────────────────────────────
# 통합 Config
# ──────────────────────────────────────────────
@dataclass
class Config:
    path:  PathConfig  = field(default_factory=PathConfig)
    cls:   ClassConfig = field(default_factory=ClassConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    aug:   AugConfig   = field(default_factory=AugConfig)


# 싱글톤처럼 사용할 기본 인스턴스
cfg = Config()


def get_config() -> Config:
    """다른 모듈에서 cfg를 가져올 때 사용합니다."""
    return cfg
