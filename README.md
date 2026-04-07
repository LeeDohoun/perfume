# 🌸 향수 이미지 기반 Note 분류 시스템

향수병 이미지를 입력받아 향료 노트 계열(Floral, Woody, Citrus 등)을 자동 분류하는 EfficientNet-B0 기반 딥러닝 파이프라인입니다.

---

## 📁 프로젝트 구조

```
PERFUME/
├── data/                          # 전처리된 CSV 데이터
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── all_cleaned.csv            # 전처리 원본
│   └── final_perfume_data.csv     # 최종 전처리 결과
│
├── perfume_images/                # 향수병 이미지 파일
│
├── perfume_classifier/            # 딥러닝 파이프라인
│   ├── config.py                  # 하이퍼파라미터 & 경로 설정
│   ├── dataset.py                 # Dataset / Augmentation / DataLoader
│   ├── model.py                   # EfficientNet-B0 + 분류 Head
│   ├── train.py                   # 2단계 Fine-tuning 학습 루프
│   ├── evaluate.py                # 평가 지표 및 시각화
│   ├── utils.py                   # EarlyStopping / 체크포인트 / 유틸
│   └── main.py                    # CLI 진입점
│
├── train_model.py                 # 기존 학습 코드 (참고용)
├── main.py                        # 기존 진입점 (참고용)
├── requirements.txt
└── README.md
```

---

## 🗂️ 데이터 구조

CSV 파일은 아래 컬럼 구조를 따릅니다.

| 컬럼 | 설명 |
|---|---|
| `image_path` | 이미지 파일 경로 |
| `label` | Note 계열 (Floral / Woody / Amber/Oriental / Citrus / Sweet / Spicy) |
| `name` | 향수 제품명 |
| `brand` | 브랜드명 |
| `notes` | 원본 노트 문자열 |
| `image_url` | 원본 이미지 URL |

### 클래스별 샘플 수

| 클래스 | Train | Val | Test | 합계 |
|---|---|---|---|---|
| Floral | 254 | 32 | 32 | 318 |
| Woody | 253 | 32 | 31 | 316 |
| Amber/Oriental | 166 | 21 | 21 | 208 |
| Citrus | 153 | 19 | 19 | 191 |
| Sweet | 98 | 12 | 13 | 123 |
| Spicy | 82 | 10 | 10 | 102 |
| **합계** | **1,006** | **126** | **126** | **1,258** |

> Stratified Split (80% / 10% / 10%) 적용

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

## 🔧 설정 (`perfume_classifier/config.py`)

CSV를 `data/` 폴더로 이동했으므로 경로가 아래와 같이 설정되어 있습니다.

```python
@dataclass
class PathConfig:
    train_csv:  str = os.path.join(BASE_DIR, "..", "data", "train.csv")
    val_csv:    str = os.path.join(BASE_DIR, "..", "data", "val.csv")
    test_csv:   str = os.path.join(BASE_DIR, "..", "data", "test.csv")
    image_root: str = os.path.join(BASE_DIR, "..", "perfume_images")
```

> `BASE_DIR`은 `perfume_classifier/` 기준이므로 `..`으로 루트를 참조합니다.

주요 하이퍼파라미터:

| 설정 | 기본값 | 설명 |
|---|---|---|
| `backbone` | `efficientnet_b0` | `mobilenet_v3_large` 로 변경 가능 |
| `stage1_epochs` | 5 | Head만 학습하는 Warm-up |
| `stage2_epochs` | 25 | Gradual Unfreeze Fine-tuning |
| `stage1_lr` | 1e-3 | Stage 1 학습률 |
| `stage2_lr` | 1e-4 | Stage 2 학습률 |
| `batch_size` | 32 | 배치 크기 |
| `patience` | 7 | Early Stopping 허용 epoch 수 |
| `use_weighted_sampler` | True | 클래스 불균형 보정 |
| `use_amp` | True | Mixed Precision (GPU 권장) |

---

## 🚀 실행

모든 명령어는 **`perfume_classifier/`** 디렉터리 안에서 실행합니다.

```bash
cd perfume_classifier
```

### 학습 + 평가 한번에
```bash
python main.py --mode train_eval
```

### 학습만
```bash
python main.py --mode train
```

### 평가만
```bash
python main.py --mode eval --ckpt checkpoints/stage2_best.pth
```

### 검증셋 평가
```bash
python main.py --mode eval --split val
```

### CLI 옵션으로 설정 덮어쓰기
```bash
# 경량 backbone으로 변경
python main.py --mode train --backbone mobilenet_v3_large

# 배치 크기 변경
python main.py --mode train --batch_size 16
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
- 분류 Head만 `lr=1e-3`으로 학습 (Adam)

### Stage 2 — Gradual Unfreeze (25 epochs)
- **Epoch 1** : 마지막 3블록 Unfreeze
- **Epoch 6** : 마지막 5블록 Unfreeze
- **Epoch 11** : 전체 Backbone Unfreeze
- Head `lr=1e-4` / Backbone `lr=1e-5` 차등 적용
- AdamW + CosineAnnealingLR + Early Stopping (patience=7)

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
| CLIP (ViT-B/32) | 45 ~ 60% | 16.7% (1/6) |

---

## 💾 출력 파일

학습 완료 후 생성되는 파일들입니다.

```
perfume_classifier/
├── checkpoints/
│   ├── stage1_best.pth              # Stage 1 최적 가중치
│   └── stage2_best.pth              # Stage 2 최적 가중치 (최종)
└── results/
    ├── training_curves.png          # Loss / Accuracy 학습 곡선
    ├── test_confusion_matrix.png    # Confusion Matrix
    ├── test_per_class_accuracy.png  # 클래스별 정확도 막대 그래프
    └── test_metrics.txt             # 수치 결과 요약
```

---

## 📌 참고

- 데이터셋 출처: Kaggle — LuckyScent Perfume Dataset (총 2,067개 유효 샘플)
- 시각-후각 상관관계가 낮은 태스크 특성상, **Random Baseline(16.7%) 대비 유의미한 개선**이 핵심 목표입니다.
- CLIP 기반 Zero-shot / Fine-tuning 실험은 추후 별도로 진행 예정입니다.
