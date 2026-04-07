# 🌸 향수 이미지 기반 Note 분류 시스템

향수병 이미지를 입력받아 향료 노트 계열(Floral, Woody, Citrus 등)을 자동 분류하는 EfficientNet-B0 기반 딥러닝 파이프라인입니다.

---

## 📁 프로젝트 구조

```
perfume_classifier/
├── config.py       # 모든 하이퍼파라미터 & 경로 설정
├── dataset.py      # Dataset / Augmentation / DataLoader
├── model.py        # EfficientNet-B0 + 커스텀 분류 Head
├── train.py        # 2단계 Fine-tuning 학습 루프
├── evaluate.py     # Accuracy / Macro F1 / Top-3 / Confusion Matrix
├── utils.py        # EarlyStopping / 체크포인트 / 시각화 유틸
├── main.py         # CLI 진입점
└── requirements.txt
```

---

## 🗂️ 데이터 준비

CSV 파일은 아래 컬럼 구조를 따릅니다.

| 컬럼 | 설명 |
|---|---|
| `image_path` | 이미지 파일 경로 (상대 or 절대) |
| `label` | Note 계열 (Floral, Woody, Amber/Oriental, Citrus, Sweet, Spicy) |
| `name` | 향수 제품명 |
| `brand` | 브랜드명 |
| `notes` | 원본 노트 문자열 |
| `image_url` | 원본 이미지 URL |

데이터 디렉터리 예시:
```
data/
├── train.csv
├── val.csv
├── test.csv
└── images/
    ├── 0001.jpg
    └── ...
```

---

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

| 주요 패키지 | 버전 |
|---|---|
| torch | >= 2.0.0 |
| torchvision | >= 0.15.0 |
| scikit-learn | >= 1.2.0 |
| pandas | >= 1.5.0 |
| seaborn | >= 0.12.0 |

---

## 🔧 설정 (`config.py`)

실행 전 `config.py`의 경로를 실제 환경에 맞게 수정하세요.

```python
@dataclass
class PathConfig:
    train_csv:  str = "data/train.csv"
    val_csv:    str = "data/val.csv"
    test_csv:   str = "data/test.csv"
    image_root: str = "data/images"   # image_path가 절대경로면 "" 로
```

주요 하이퍼파라미터:

| 설정 | 기본값 | 설명 |
|---|---|---|
| `backbone` | `efficientnet_b0` | `mobilenet_v3_large` 로 변경 가능 |
| `stage1_epochs` | 5 | Head만 학습하는 Warm-up 단계 |
| `stage2_epochs` | 25 | Gradual Unfreeze Fine-tuning 단계 |
| `stage1_lr` | 1e-3 | Stage 1 학습률 |
| `stage2_lr` | 1e-4 | Stage 2 학습률 |
| `batch_size` | 32 | 배치 크기 |
| `patience` | 7 | Early Stopping 허용 epoch 수 |
| `use_weighted_sampler` | True | 클래스 불균형 보정 |
| `use_amp` | True | Mixed Precision (GPU 권장) |

---

## 🚀 실행

### 학습 + 평가 한번에
```bash
python main.py --mode train_eval
```

### 학습만
```bash
python main.py --mode train
```

### 평가만 (체크포인트 지정)
```bash
python main.py --mode eval --ckpt checkpoints/stage2_best.pth
```

### 검증셋 평가
```bash
python main.py --mode eval --split val
```

### CLI 옵션 덮어쓰기
```bash
# backbone 변경
python main.py --mode train --backbone mobilenet_v3_large

# 배치 크기 변경
python main.py --mode train --batch_size 16

# 시드 고정
python main.py --mode train --seed 123
```

---

## 🧠 모델 아키텍처

```
입력 이미지 (3 × 224 × 224)
        ↓
EfficientNet-B0 Backbone (features)
        ↓
AdaptiveAvgPool2d → Flatten
        ↓
BatchNorm1d → Dropout(0.3) → Linear(1280 → 256)
        ↓
ReLU → BatchNorm1d → Dropout(0.15) → Linear(256 → 6)
        ↓
출력 (6개 클래스 logit)
```

---

## 📈 2단계 학습 전략

### Stage 1 — Head Only (5 epochs)
- Backbone 전체 Freeze
- 분류 Head만 `lr=1e-3`으로 학습
- Adam optimizer

### Stage 2 — Gradual Unfreeze (25 epochs)
- **Epoch 1**: 마지막 3블록 Unfreeze
- **Epoch 6**: 마지막 5블록 Unfreeze
- **Epoch 11**: 전체 Backbone Unfreeze
- Head `lr=1e-4`, Backbone `lr=1e-5` (차등 적용)
- AdamW + CosineAnnealingLR + Early Stopping

---

## 📊 평가 척도

| 지표 | 설명 |
|---|---|
| **Accuracy** | 전체 정분류율 |
| **Macro F1-Score** | 클래스 불균형 보정 F1 |
| **Top-3 Accuracy** | 상위 3개 예측 안에 정답 포함 비율 |
| **Confusion Matrix** | 클래스 간 오분류 패턴 시각화 |

### 예상 성능 범위

| 모델 | 예상 Accuracy | Random Baseline |
|---|---|---|
| EfficientNet-B0 | 35 ~ 50% | 16.7% (1/6) |

---

## 💾 출력 파일

학습 완료 후 아래 파일들이 생성됩니다.

```
checkpoints/
├── stage1_best.pth           # Stage 1 최적 가중치
└── stage2_best.pth           # Stage 2 최적 가중치 (최종)

results/
├── training_curves.png       # 학습 Loss / Accuracy 곡선
├── test_confusion_matrix.png # Confusion Matrix
├── test_per_class_accuracy.png
└── test_metrics.txt          # 수치 결과 요약
```

---

## 🔁 MobileNetV3로 전환 (경량 대안)

GPU 없이 CPU 환경에서 빠른 실험이 필요할 때:

```bash
python main.py --mode train_eval --backbone mobilenet_v3_large
```

또는 `config.py`에서 직접 변경:
```python
backbone: str = "mobilenet_v3_large"
```

---

## 📌 참고

- 데이터셋: [Kaggle — LuckyScent Perfume Dataset](https://www.kaggle.com)
- 시각-후각 상관관계가 낮은 태스크 특성상 절대적 정확도보다 **Random Baseline 대비 유의미한 개선** 여부가 핵심 목표입니다.
- CLIP 기반 Zero-shot 실험은 별도 브랜치에서 진행 예정입니다.
