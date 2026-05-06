"""
model.py
--------
EfficientNet 기반 향수 Note 분류 모델.

- build_model()  : backbone + 커스텀 분류 Head 반환
- freeze_backbone() / unfreeze_last_n_blocks() : 2단계 학습용 유틸
- get_param_groups() : Stage2 용 차등 lr parameter group 반환
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as models

from config import get_config

cfg = get_config()


# ──────────────────────────────────────────────
# 분류 Head
# ──────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Backbone feature → num_classes logit.
    BN → Dropout → Linear 구조로 소규모 데이터 과적합 방지.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class TextEncoder(nn.Module):
    """Embedding encoder for fixed-length brand/name token ids."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)
        mask = (token_ids != self.padding_idx).unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = (emb * mask).sum(dim=1) / lengths
        return self.proj(pooled)


# ──────────────────────────────────────────────
# 모델 빌더
# ──────────────────────────────────────────────

class PerfumeClassifier(nn.Module):
    """
    향수 Note 분류 모델.

    backbone : EfficientNet-V2-S (기본), EfficientNet-B3 (V1), EfficientNet-B0, MobileNetV3-Large
    head     : ClassificationHead (커스텀)
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
        text_vocab_size: int = 0,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.use_text = cfg.text.use_text
        self.image_feature_dim = 0
        self.text_feature_dim = cfg.text.hidden_dim if self.use_text else 0

        # ── Backbone 로드 ────────────────────────────
        weights_arg = "DEFAULT" if pretrained else None

        if backbone == "efficientnet_v2_s":
            base = models.efficientnet_v2_s(weights=weights_arg)
            in_features = base.classifier[1].in_features   # 1280
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)

        elif backbone == "efficientnet_b3":
            base = models.efficientnet_b3(weights=weights_arg)
            in_features = base.classifier[1].in_features   # 1536
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)

        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights_arg)
            in_features = base.classifier[1].in_features   # 1280
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)

        elif backbone == "mobilenet_v3_large":
            base = models.mobilenet_v3_large(weights=weights_arg)
            in_features = base.classifier[0].in_features   # 960
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)

        else:
            raise ValueError(f"지원하지 않는 backbone: {backbone}. "
                             f"'efficientnet_v2_s', 'efficientnet_b3', 'efficientnet_b0', 'mobilenet_v3_large' 중 선택하세요.")

        # ── 분류 Head ────────────────────────────────
        self.image_feature_dim = in_features
        if self.use_text:
            if text_vocab_size <= 0:
                raise ValueError("text_vocab_size must be positive when cfg.text.use_text=True")
            self.text_encoder = TextEncoder(
                vocab_size=text_vocab_size,
                embedding_dim=cfg.text.embedding_dim,
                hidden_dim=cfg.text.hidden_dim,
                dropout=dropout / 2,
            )
        else:
            self.text_encoder = None

        fused_features = self.image_feature_dim + self.text_feature_dim

        self.head = ClassificationHead(
            in_features=fused_features,
            num_classes=num_classes,
            dropout=dropout,
        )

        print(f"[Model] {backbone} 로드 완료 | pretrained={pretrained} | "
              f"image_features={self.image_feature_dim} | text_features={self.text_feature_dim} | "
              f"num_classes={num_classes}")

    def forward(self, x: torch.Tensor, text_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        image_features = torch.flatten(x, 1)
        if self.use_text:
            if text_ids is None:
                raise ValueError("text_ids is required when cfg.text.use_text=True")
            text_features = self.text_encoder(text_ids)
            x = torch.cat([image_features, text_features], dim=1)
        else:
            x = image_features
        x = self.head(x)
        return x

    # ── 2단계 학습 유틸 ───────────────────────────────

    def freeze_backbone(self):
        """Stage 1: Backbone 전체를 Freeze합니다."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone Freeze 완료")

    def unfreeze_last_n_blocks(self, n: int):
        """
        Stage 2: Backbone의 마지막 n개 블록(children)을 Unfreeze합니다.
        EfficientNet 계열 기준 총 9개 블록 (features[0] ~ features[8]).
        n=3 이면 features[6], [7], [8] 을 Unfreeze.
        """
        blocks = list(self.backbone.children())
        total = len(blocks)
        unfreeze_from = max(0, total - n)

        # 우선 전체 Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 마지막 n개 Unfreeze
        for block in blocks[unfreeze_from:]:
            for param in block.parameters():
                param.requires_grad = True

        frozen_cnt   = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        unfrozen_cnt = sum(1 for p in self.backbone.parameters() if p.requires_grad)
        print(f"[Model] Backbone 마지막 {n}블록 Unfreeze | "
              f"frozen={frozen_cnt} / unfrozen={unfrozen_cnt}")

    def unfreeze_all(self):
        """전체 Backbone Unfreeze."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] Backbone 전체 Unfreeze 완료")

    def get_param_groups(self, stage2_lr: float) -> List[dict]:
        """
        Stage 2 용 차등 Learning Rate parameter group을 반환합니다.
        Head > Backbone 마지막 블록 > 나머지 순으로 lr을 낮춥니다.
        """
        head_params     = list(self.head.parameters())
        if self.text_encoder is not None:
            head_params += list(self.text_encoder.parameters())
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]

        return [
            {"params": head_params,     "lr": stage2_lr},
            {"params": backbone_params, "lr": stage2_lr * 0.1},
        ]

    def count_parameters(self) -> dict:
        """학습 가능 / 전체 파라미터 수를 반환합니다."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ──────────────────────────────────────────────
# 편의 함수
# ──────────────────────────────────────────────

def build_model(num_classes: Optional[int] = None) -> PerfumeClassifier:
    """
    config 설정을 읽어 PerfumeClassifier를 생성합니다.
    """
    mc = cfg.model
    nc = num_classes or len(cfg.cls.note_classes)
    text_vocab_size = 0
    if cfg.text.use_text:
        from dataset import get_text_vocab
        text_vocab_size = len(get_text_vocab())

    model = PerfumeClassifier(
        num_classes=nc,
        backbone=mc.backbone,
        pretrained=mc.pretrained,
        dropout=mc.dropout,
        text_vocab_size=text_vocab_size,
    )
    stats = model.count_parameters()
    print(f"[Model] 파라미터: 전체={stats['total']:,} | 학습가능={stats['trainable']:,}")
    return model


# ──────────────────────────────────────────────
# 빠른 테스트 (python model.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(4, 3, 224, 224)
    dummy_text = torch.zeros(4, cfg.text.max_len, dtype=torch.long)

    # Stage 1 시뮬레이션
    model.freeze_backbone()
    out = model(dummy, dummy_text) if cfg.text.use_text else model(dummy)
    print(f"Stage1 출력 shape: {out.shape}")   # (4, 6)

    # Stage 2 시뮬레이션
    model.unfreeze_last_n_blocks(3)
    out = model(dummy, dummy_text) if cfg.text.use_text else model(dummy)
    print(f"Stage2 출력 shape: {out.shape}")
