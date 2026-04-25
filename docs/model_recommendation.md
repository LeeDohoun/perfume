# 향수 이미지 분류 모델 추천

## 데이터셋 특성 요약

| 항목 | Note 분류 | Brand 분류 |
|------|-----------|------------|
| 클래스 수 | 6개 | 35개 |
| 학습 샘플 | 1,006개 | 652개 |
| 이미지 종류 | 향수병 제품 이미지 | 향수병 제품 이미지 |
| 난이도 | 높음 (시각-향 상관 낮음) | 중간 (병 디자인이 브랜드 반영) |

> **핵심 제약**: 두 태스크 모두 샘플 수가 매우 적어 **Transfer Learning 필수**

---

## 모델 비교

### 0. TF-IDF + LogisticRegression ✅ Note 정확도 개선안

| 항목 | 내용 |
|------|------|
| 입력 | `name + brand + description` |
| 출력 | Note 6클래스 |
| 특징 | 병 이미지에서 보이지 않는 향 정보를 제품 설명 텍스트로 보완 |

**장점**
- `notes` 컬럼을 쓰지 않아 라벨 직접 누수를 피함
- 현재 데이터에서 Note 목표 정확도에 도달
- 학습이 빠르고 결과 재현이 쉬움

**단점**
- 이미지 단독 분류 모델은 아님
- 제품 설명이 없는 데이터에는 적용하기 어려움

```bash
python scripts/train_note_text.py
```

### 1. EfficientNet-B0 ✅ 최우선 추천

| 항목 | 내용 |
|------|------|
| 파라미터 수 | 5.3M |
| ImageNet Top-1 | 77.1% |
| 특징 | Compound Scaling으로 크기 대비 성능 최고 |

**장점**
- 적은 파라미터로 과적합 위험 낮음 → 소규모 데이터에 적합
- torchvision 기본 제공, 구현 간단
- 빠른 학습 속도

**단점**
- ResNet 계열보다 튜닝 경험이 적음

```python
from torchvision import models
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
```

---

### 2. MobileNetV3-Large

| 항목 | 내용 |
|------|------|
| 파라미터 수 | 5.4M |
| ImageNet Top-1 | 75.3% |
| 특징 | 모바일 최적화, 가벼운 연산 |

**장점**
- GPU 없이도 빠르게 학습 가능
- EfficientNet과 유사한 규모

**단점**
- EfficientNet-B0보다 성능 소폭 낮음

```python
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
```

---

### 3. ResNet50

| 항목 | 내용 |
|------|------|
| 파라미터 수 | 25.6M |
| ImageNet Top-1 | 76.1% |
| 특징 | 가장 검증된 기본 모델 |

**장점**
- 레퍼런스가 많고 디버깅 용이
- 안정적인 학습

**단점**
- 파라미터가 많아 1,000개 수준의 데이터에서 과적합 위험
- Feature Extraction(헤드만 학습) 방식 권장

```python
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

---

### 4. CLIP (ViT-B/32) ⭐ Note 분류 특별 추천

| 항목 | 내용 |
|------|------|
| 파라미터 수 | 151M (Vision Encoder만 87M) |
| 학습 데이터 | 4억 개 이미지-텍스트 쌍 |
| 특징 | 이미지 + 텍스트 멀티모달 |

**장점**
- `"a photo of a floral perfume"` 같은 텍스트 프롬프트로 **Zero-shot 분류** 가능
- Note 분류에서 텍스트 의미를 활용할 수 있어 유리
- 적은 데이터에도 강력한 표현력

**단점**
- 별도 라이브러리 설치 필요 (`pip install openai-clip`)
- 모델 크기가 큼

```python
import clip
model, preprocess = clip.load("ViT-B/32")

# Zero-shot: 텍스트 프롬프트로 분류
text_prompts = clip.tokenize([
    "a floral perfume bottle",
    "a woody perfume bottle",
    "a citrus perfume bottle",
    ...
])
```

---

## 태스크별 최종 추천

### Note 계열 분류 (6클래스, 1,006개)

```
1순위: TF-IDF + LogisticRegression
  → name + brand + description 사용
  → notes 컬럼은 사용하지 않음
  → 현재 Test Accuracy: 59.52%
  → 목표 Test Accuracy: 45 ~ 60%

2순위: CLIP (ViT-B/32)
  → 텍스트 의미 활용 가능, 소규모 데이터에 강함
  → zero-shot은 기준선으로 확인하고, 목표 성능은 frozen CLIP + linear probe 방식으로 접근
  → 목표 Test Accuracy: 45 ~ 60%

3순위: EfficientNet-B0
  → 구현 단순, 안정적
  → Fine-tuning 방식으로 접근
  → 목표 Test Accuracy: 35 ~ 50%
```

### Brand 분류 (35클래스, 652개)

```
1순위: EfficientNet-B0
  → 클래스 수가 많고 샘플이 적은 상황에서 파라미터 수 적은 모델이 유리
  → 브랜드별 병 디자인을 학습하는 데 충분한 표현력
  → 목표 Test Accuracy: 40 ~ 65%

2순위: MobileNetV3-Large
  → GPU 환경이 제한적일 경우
```

---

## 공통 학습 전략

### Fine-tuning 단계별 접근 (소규모 데이터 과적합 방지)

```
1단계 (5 epoch):  Backbone 전체 Freeze → Head만 학습 (lr=1e-3)
2단계 (25 epoch): 마지막 블록 + Head Unfreeze → 전체 Fine-tune (lr=1e-4)
```

### 필수 적용 기법

| 기법 | 이유 |
|------|------|
| WeightedRandomSampler | 클래스 불균형 보정 |
| RandomHorizontalFlip, ColorJitter | 데이터 부족 보완 |
| CosineAnnealingLR | 안정적인 lr 감소 |
| Early Stopping | 과적합 조기 차단 |

---

## 목표 성능

> 향수병 이미지만으로 향 계열/브랜드를 분류하는 태스크 특성상 절대적 수치보다 **학습 가능성 확인**이 목표

| 모델 | 태스크 | 목표 Test Accuracy | 비고 |
|------|--------|--------------------|------|
| TF-IDF + LogisticRegression | Note 계열 | 45 ~ 60% | 현재 59.52%, notes 미사용 |
| EfficientNet-B0 | Note 계열 | 35 ~ 50% | Random baseline: 16.7% |
| CLIP ViT-B/32 | Note 계열 | 45 ~ 60% | frozen CLIP + linear probe 기준 |
| EfficientNet-B0 | Brand | 40 ~ 65% | Random baseline: 2.9% |

## 실행 명령

```bash
# EfficientNet-B0 Note
python scripts/train_efficientnet.py --task note

# EfficientNet-B0 Brand
python scripts/train_efficientnet.py --task brand

# CLIP Note linear probe
python scripts/eval_clip_note.py --mode linear_probe --split test

# CLIP Note zero-shot baseline
python scripts/eval_clip_note.py --mode zero_shot --split test

# Note metadata text classifier
python scripts/train_note_text.py
```

`scripts/train_efficientnet.py`와 `scripts/eval_clip_note.py`는 평가 후 현재 Test Accuracy가 목표 범위에 들어왔는지 함께 출력한다.
