# CLIP 이미지 단독 향수 계열 분류 실험

실험일: 2026-05-18  
담당: 찬준 (feature/ChanJoon)

---

## 1. 실험 목적

석현님 브랜치(`feature/SeokHyeon02`)의 CLIP 모델이 4분류에서 76.10%를 기록했으나,  
텍스트 입력으로 `notes` 컬럼을 사용한 것이 **데이터 누수**임을 확인.

> `label`은 `notes` 키워드 매칭으로 생성된 값이므로,  
> `notes`를 모델 입력으로 쓰면 정답을 직접 힌트로 주는 것과 같음.

본 실험의 목적: **이미지만 넣었을 때 CLIP의 실제 성능 측정**

---

## 2. 실험 설정

### 베이스 모델

| 항목 | 내용 |
|------|------|
| 모델 | `openai/clip-vit-base-patch32` |
| 학습 방식 | CLIP 전체 freeze → 임베딩 추출 후 classifier head만 학습 |
| Feature 추출 | Vision encoder → `visual_projection` → L2 normalize → 512d 벡터 |
| Classifier | LayerNorm → Linear(512→512) → GELU → Dropout(0.25) → Linear(512→4) |

### 텍스트 입력

| 항목 | 내용 |
|------|------|
| text_mode | `none` |
| 실제 입력 | `"a perfume bottle"` (고정 중립 문장) |
| 사용 feature | **이미지 임베딩만** (텍스트 임베딩 미사용) |

> 석현님 코드(`fragrance notes: {notes}`)와의 차이: notes 완전 제거

### 데이터셋

| Split | 전체 | Amber | Floral | Fresh | Woody |
|-------|------|-------|--------|-------|-------|
| Train | 22,557 | 1,758 | 9,480 | 3,346 | 7,973 |
| Val   | 2,820  | 220   | 1,185  | 418   | 997   |
| Test  | 2,820  | 220   | 1,185  | 418   | 997   |

- 소규모 원본(1,258개) 대비 **약 22배** 증가한 대용량 데이터
- 클래스 불균형 존재 (Floral 9,480 vs Amber 1,758 → 약 5.4배 차이)

### 학습 하이퍼파라미터

| 항목 | 값 |
|------|----|
| Epochs | 20 (Early stopping patience=7) |
| Batch size | 64 |
| Optimizer | AdamW |
| LR | 1e-3 |
| LR Scheduler | CosineAnnealingLR |
| Weight decay | 1e-4 |
| Label smoothing | 0.05 |
| Loss | CrossEntropyLoss (클래스 가중치 적용) |
| WeightedRandomSampler | 적용 (클래스 불균형 보정) |

### 실행 환경

| 항목 | 내용 |
|------|------|
| 머신 | MacBook Pro M1 Pro 32GB |
| 연산 장치 | MPS (Apple Silicon) |
| 실행 커맨드 | `python clip_classifier/train.py --text-mode none --rebuild-embedding-cache` |

---

## 3. 학습 과정

Early stopping이 **epoch 2**에서 best val acc 기록 후 epoch 9에서 조기 종료.

| Epoch | Train Loss | Train Acc | Val Acc | Val Top-3 |
|-------|-----------|-----------|---------|-----------|
| 1 | 1.0448 | 0.3641 | 0.2103 | 0.8128 |
| **2** | **0.9949** | **0.4022** | **0.3489** | **0.8443** |
| 3 | 0.9933 | 0.4041 | 0.2248 | 0.7993 |
| 4 | 1.0070 | 0.3929 | 0.2050 | 0.7844 |
| 5 | 1.0147 | 0.3856 | 0.1787 | 0.8057 |
| 6 | 1.0020 | 0.3896 | 0.2220 | 0.7738 |
| 7 | 0.9793 | 0.4036 | 0.2021 | 0.7635 |
| 8 | 0.9759 | 0.3981 | 0.1890 | 0.7730 |
| 9 | 0.9435 | 0.4243 | 0.1628 | 0.7894 |

val_acc가 epoch 2 이후 지속 하락 → 수렴하지 못하고 조기 종료.

---

## 4. 최종 결과 (Test set)

### 전체 지표

| 지표 | 값 |
|------|----|
| **Test Accuracy** | **34.61%** |
| Test Top-3 Accuracy | 84.47% |
| Macro F1 | 0.3100 |
| Weighted F1 | 0.3431 |
| Random Baseline (4분류) | 25.00% |

### 클래스별 성능

| 클래스 | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| Amber  | 0.1149 | 0.4591 | 0.1838 | 220 |
| Floral | 0.7259 | 0.4203 | 0.5323 | 1,185 |
| Fresh  | 0.2675 | 0.7225 | 0.3904 | 418 |
| Woody  | 0.5952 | 0.0752 | 0.1336 | 997 |

---

## 5. 분석

### 전체 비교

| 모델 | 인풋 | Accuracy | Macro F1 | 누수 여부 |
|------|------|----------|----------|---------|
| 석현님 CLIP (ViT-L/14) | 이미지 + notes | 76.10% | 0.7058 | **있음** |
| 찬준님 EfficientNet-B0 | 이미지 | 51.77% | 0.4593 | 없음 |
| **CLIP ViT-B/32 (본 실험)** | **이미지만** | **34.61%** | **0.3100** | **없음** |

### 주요 발견

**1. notes 누수의 영향이 압도적**  
석현님 CLIP 76.10% → 이미지 단독 CLIP 34.61%로 **41.49%p 급락**.  
76.10%의 성능은 이미지가 아닌 notes 텍스트로 분류한 결과에 가깝다.

**2. 이미지 단독에서는 EfficientNet-B0이 CLIP보다 우수**  
같은 조건(이미지만, 누수 없음)에서 EfficientNet(51.77%) > CLIP(34.61%).  
CLIP의 vision encoder가 향수 계열 분류 태스크에 특화되어 있지 않기 때문.

**3. Woody recall 7.5% — 심각한 쏠림 현상**  
Woody(997개)가 가장 많은 클래스 중 하나임에도 recall이 7.5%에 불과.  
Fresh로 예측이 쏠리는 현상(Fresh recall 72.3%)과 맞물려 클래스 간 혼동이 큼.

**4. 수렴 실패**  
Train accuracy는 epoch마다 40% 근방에서 정체, val accuracy는 epoch 2 이후 지속 하락.  
이미지만으로는 Note 계열을 학습하기 어렵다는 것을 학습 곡선에서도 확인.

---

## 6. 결론

> CLIP 이미지 단독 분류는 **34.61%**로 랜덤 베이스라인(25%) 대비 소폭 높은 수준에 그침.  
> 석현님의 76.10%는 notes 데이터 누수에 의한 결과이며, 신뢰할 수 있는 이미지 기반 성능으로 볼 수 없음.  
> 향수 계열 분류에서 이미지 단독 접근의 한계를 정량적으로 확인한 실험.

---

## 7. 다음 실험 제안

- `--text-mode name_brand`: 이미지 + 제품명/브랜드명 (누수 없는 멀티모달)
- ViT-L/14 모델로 이미지 단독 재실험 (더 강력한 vision encoder)
- CLIP zero-shot: 학습 없이 프롬프트 기반 분류
