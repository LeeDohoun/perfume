# 향수 이미지 기반 분류 프로젝트 보고서

## 1. 프로젝트 개요

본 프로젝트는 향수 제품 데이터를 활용하여 두 가지 분류 태스크를 수행하는 것을 목표로 한다.

| 태스크 | 입력 | 출력 |
|--------|------|------|
| Note 분류 | 향수병 이미지 또는 제품 메타데이터 | 향 계열 6종 |
| Brand 분류 | 향수병 이미지 | 브랜드 35종 |

초기 목표는 향수병 이미지만으로 향 계열과 브랜드를 분류하는 것이었다. 그러나 실험 결과, 브랜드는 병 디자인과 로고 등 시각적 단서가 강해 높은 정확도를 보였지만, 향 계열은 이미지 단독으로는 충분한 정보를 얻기 어려웠다. 따라서 Note 분류의 정확도 문제를 해결하기 위해 제품명, 브랜드명, 설명 텍스트를 활용한 보조 모델을 추가하였다.

## 2. 데이터 구성

사용한 데이터는 LuckyScent 기반 향수 데이터이다.

| 데이터 | 내용 |
|--------|------|
| 원본 CSV | `data/raw/final_perfume_data.csv` |
| 정제 CSV | `data/raw/all_cleaned.csv` |
| 이미지 폴더 | `perfume_images/` |
| 이미지 수 | 2,067장 |

최종 학습 데이터 분할은 다음과 같다.

| 태스크 | Train | Validation | Test | 클래스 수 |
|--------|------:|-----------:|-----:|----------:|
| Note | 1,006 | 126 | 126 | 6 |
| Brand | 652 | 81 | 82 | 35 |

## 3. 데이터 전처리

전처리는 `prepare.py`에서 수행하였다.

### 3.1 공통 전처리

- `data/raw/all_cleaned.csv`를 기준 데이터로 사용하였다.
- 이미지 경로를 macOS 환경에서 정상적으로 읽을 수 있도록 `\`를 `/`로 정규화하였다.
- 원본 CSV에만 있던 `Description` 컬럼을 `image_url` 기준으로 병합하였다.
- Stratified 80 / 10 / 10 방식으로 train, validation, test를 분할하였다.
- `random_state=42`를 사용하여 분할 재현성을 확보하였다.

### 3.2 Note 라벨 생성

Note 분류 라벨은 `notes` 문자열에서 키워드 기반 규칙으로 생성하였다.

| 처리 | 내용 |
|------|------|
| 라벨 방식 | Notes 문자열을 콤마 단위로 분리 후 키워드 매칭 |
| 동점 처리 | 2개 이상 계열이 동점인 샘플 제거 |
| 무매칭 처리 | 어떤 계열에도 매칭되지 않는 샘플 제거 |
| Fresh 처리 | 샘플 수 부족으로 별도 클래스에서 제외 |
| 최종 클래스 | 6개 |

Fresh 클래스는 샘플 수가 적어 제거하였고, 일부 키워드는 유사 계열로 이동하였다.

- `lavender` → `Floral`
- `lemongrass` → `Citrus`

Note 텍스트 모델에서는 `notes` 컬럼을 입력으로 사용하지 않았다. `notes`는 라벨 생성의 원천이기 때문에 입력으로 사용하면 라벨 누수가 발생할 수 있기 때문이다.

### 3.3 Brand 라벨 생성

Brand 분류는 `brand` 컬럼을 그대로 라벨로 사용하였다. 단, 샘플 수가 너무 적은 브랜드는 학습이 어렵기 때문에 15개 이상 샘플을 가진 브랜드만 사용하였다.

## 4. 모델 및 학습 방법

## 4.1 EfficientNet-B0

EfficientNet-B0는 Note 분류와 Brand 분류 모두에 사용하였다. 학습 방식은 기존 계획에 맞춰 전이학습 방식으로 구성하였다.

학습 전략:

- ImageNet 사전학습 가중치 사용
- Stage 1: Backbone freeze 후 classifier head만 학습
- Stage 2: 마지막 3개 block unfreeze 후 fine-tuning
- WeightedRandomSampler로 클래스 불균형 보정
- 데이터 증강 적용
- CosineAnnealingLR 사용
- Early Stopping 적용

이미지 증강은 학습 중에만 적용하였고, 증강된 이미지를 파일로 따로 저장하지는 않았다.

사용한 증강:

- Resize
- RandomCrop
- RandomHorizontalFlip
- RandomRotation
- ColorJitter
- RandomAffine
- Normalize

## 4.2 CLIP

CLIP은 Note 분류에서 이미지와 텍스트 의미를 활용할 수 있는 모델로 실험하였다.

사용 방식:

- CLIP ViT-B/32 계열 모델 사용
- Frozen CLIP image encoder 사용
- Linear probe 방식으로 Note 분류 평가
- 입력은 이미지 중심으로 사용

CLIP은 강력한 사전학습 모델이지만, 현재 실험에서는 향수병 이미지만으로 향 계열을 충분히 분리하지 못하였다.

## 4.3 TF-IDF + LogisticRegression

Note 분류 정확도 문제를 해결하기 위해 텍스트 메타데이터 기반 모델을 추가하였다.

입력:

- `name`
- `brand`
- `description`

사용하지 않은 입력:

- `notes`

모델 구조:

```text
name + brand + description
→ TF-IDF
→ LogisticRegression
→ Note label
```

이 방식은 이미지 단독 모델이 아니라, 제품 페이지에 존재하는 텍스트 정보를 활용하는 보조 모델이다. 향 계열은 병 이미지보다 제품 설명 텍스트에 더 직접적으로 드러나기 때문에 Note 분류 성능 개선에 효과적이었다.

## 5. 실험 결과

목표 정확도와 실제 결과는 다음과 같다.

| 모델 | 태스크 | 목표 Test Accuracy | 실제 Test Accuracy | 판정 |
|------|--------|--------------------|--------------------|------|
| EfficientNet-B0 | Note | 35 ~ 50% | 24.60% | 목표 미달 |
| EfficientNet-B0 | Brand | 40 ~ 65% | 95.12% | 목표 초과 |
| CLIP linear probe | Note | 45 ~ 60% | 26.98% | 목표 미달 |
| TF-IDF + LogisticRegression | Note | 45 ~ 60% | 59.52% | 목표 달성 |

## 6. 결과 분석

### 6.1 Brand 분류

Brand 분류는 EfficientNet-B0에서 95.12%의 Test Accuracy를 기록하였다. 이는 목표 범위인 40~65%를 크게 초과한 결과이다.

높은 정확도가 나온 이유는 브랜드 분류가 이미지와 직접적으로 연결되어 있기 때문이다. 향수병 디자인, 라벨, 로고, 패키지 스타일은 브랜드마다 고유한 경우가 많다. 따라서 이미지 기반 CNN 모델이 브랜드를 학습하기에 충분한 시각적 단서를 얻을 수 있었다.

다만 test 데이터가 82개로 크지 않고, 클래스당 샘플 수가 적은 경우도 있으므로 실제 일반화 성능은 추가 데이터로 더 검증할 필요가 있다.

### 6.2 Note 이미지 분류

EfficientNet-B0 Note 분류는 24.60%, CLIP Note 분류는 26.98%의 정확도를 기록하였다. 두 모델 모두 목표 정확도에 도달하지 못했다.

가장 큰 원인은 입력 정보의 한계이다. Note 라벨은 향수의 실제 향 계열을 의미하지만, 입력 이미지는 병의 외형이다. 병 디자인만으로 향이 Floral인지, Woody인지, Sweet인지 구분하기는 어렵다.

또한 Note 라벨은 사람이 직접 부여한 정답이 아니라 `notes` 텍스트에서 규칙 기반으로 생성한 라벨이다. 향수는 여러 향 계열을 동시에 가질 수 있으므로 single-label 분류로 단순화하는 과정에서 라벨 노이즈가 발생할 수 있다.

### 6.3 Note 텍스트 분류

텍스트 메타데이터 기반 모델은 59.52%의 Test Accuracy를 기록하여 목표 범위인 45~60%에 도달하였다.

이 결과는 Note 분류에서 제품명, 브랜드명, 제품 설명이 이미지보다 더 유용한 정보임을 보여준다. 특히 제품 설명에는 향의 분위기, 주요 재료, 향조에 대한 표현이 포함되어 있어 향 계열 분류에 직접적인 단서를 제공한다.

## 7. Note 태스크의 문제점

Note 분류에서 확인된 주요 문제는 다음과 같다.

1. 이미지와 라벨의 정보 연결이 약하다.
2. 향수는 여러 향 계열을 동시에 가지는 multi-label 성격이 강하다.
3. 현재 라벨은 키워드 규칙으로 만든 single-label이라 노이즈가 존재한다.
4. 일부 클래스의 샘플 수가 적고 불균형하다.
5. 이미지 모델 학습 로그에서 과적합 경향이 나타난다.
6. CLIP도 이미지 단독 입력에서는 향 계열 정보를 충분히 분리하지 못했다.

## 8. 개선 방향

향후 개선 방향은 다음과 같다.

1. Note 분류를 이미지 단독이 아니라 `image + name + brand + description` 멀티모달 모델로 확장한다.
2. `notes`를 입력으로 사용하는 실험은 라벨 누수 가능성이 있으므로 별도로 구분한다.
3. Note 라벨을 single-label이 아니라 multi-label 방식으로 재정의한다.
4. 클래스별 대표 샘플을 수동 검수하여 라벨 품질을 높인다.
5. Note 이미지 모델은 더 보수적인 fine-tuning을 적용한다.
   - unfreeze 범위 축소
   - learning rate 감소
   - early stopping 강화
   - class-weight loss 적용
6. Brand 분류는 높은 정확도가 나온 만큼 더 큰 test set으로 일반화 성능을 재검증한다.

## 9. 재현 방법

전처리:

```bash
.venv/bin/python prepare.py
```

EfficientNet-B0 Note 학습:

```bash
.venv/bin/python train.py --task note --batch_size 32
```

EfficientNet-B0 Brand 학습:

```bash
.venv/bin/python -u train.py --task brand --batch_size 32
```

CLIP Note 평가:

```bash
.venv/bin/python -u clip_note_eval.py --mode linear_probe --split test --batch_size 64 --device cpu
```

Note 텍스트 모델 학습:

```bash
.venv/bin/python train_note_text.py
```

저장된 모델 재평가:

```bash
.venv/bin/python eval_checkpoint.py --task note --device cpu
.venv/bin/python eval_checkpoint.py --task brand --device cpu
.venv/bin/python eval_note_text.py
```

## 10. 결론

본 프로젝트에서는 향수병 이미지를 활용한 Brand 분류와 Note 분류를 수행하였다. Brand 분류는 EfficientNet-B0를 통해 95.12%의 높은 정확도를 달성하였다. 이는 브랜드가 병 디자인과 강하게 연결되어 있기 때문이다.

반면 Note 분류는 이미지 단독으로는 목표 정확도에 도달하지 못하였다. 이는 향 계열이 병 이미지에서 직접 관찰되기 어려운 정보이기 때문이다. 이를 해결하기 위해 제품명, 브랜드명, 설명 텍스트를 활용한 TF-IDF + LogisticRegression 모델을 추가하였고, 59.52%의 정확도로 목표 범위에 도달하였다.

따라서 본 프로젝트의 핵심 결론은 다음과 같다.

- Brand 분류는 이미지 기반 모델이 효과적이다.
- Note 분류는 이미지 단독보다 텍스트 메타데이터 활용이 필요하다.
- 향수 Note 분류는 향후 multi-label 및 멀티모달 방식으로 확장하는 것이 적절하다.
