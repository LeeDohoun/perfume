# 향수 Note 분류 프로젝트 보고서

## 1. 프로젝트 개요

본 프로젝트는 향수 상품 이미지와 제품 메타데이터를 활용해 향수의 Note 계열을 분류하는 것을 목표로 한다. 현재 실험 기준은 찬준 브랜치에서 가져온 `data/All` 데이터와 `perfume_classifier` 파이프라인이다.

현재 모델 입력:

- 향수 이미지
- 브랜드명 `brand`
- 제품명 `name`

사용하지 않는 입력:

- `notes`

`notes`는 Note 라벨의 직접 근거가 될 수 있으므로, 최종 모델 입력에 넣지 않는다.

## 2. 데이터 구성

현재 학습/평가 데이터는 `data/All` 아래 CSV를 사용한다.

| split | rows |
|---|---:|
| train | 22,557 |
| validation | 2,820 |
| test | 2,820 |
| train_aug | 41,061 |

클래스는 7개다.

| label | test rows |
|---|---:|
| Floral | 1,097 |
| Woody | 946 |
| Amber_Oriental | 220 |
| Fresh | 212 |
| Citrus | 207 |
| Sweet | 87 |
| Spicy | 51 |

데이터는 불균형하다. 특히 `Sweet`, `Spicy`는 테스트 샘플이 적고, `Floral`, `Woody`가 전체의 큰 비중을 차지한다.

## 3. 모델 구조

현재 모델은 EfficientNet-B0 기반 멀티 입력 분류기다.

| 구성 | 내용 |
|---|---|
| 이미지 backbone | EfficientNet-B0 |
| 이미지 feature | 1,280차원 |
| 텍스트 입력 | `brand`, `name` |
| 텍스트 인코더 | token embedding + 평균 pooling + projection |
| fusion | image feature와 text feature concat |
| classifier | BatchNorm, Dropout, Linear, ReLU 기반 head |
| 출력 | 7개 Note 클래스 |

학습은 2단계로 구성되어 있다.

1. Stage 1: backbone freeze 후 head 중심 학습
2. Stage 2: backbone 일부 unfreeze 후 fine-tuning

학습에는 offline augmentation, weighted sampler, focal loss가 사용된다.

## 4. 평가 결과

2026-05-11에 현재 작업트리에서 `checkpoints/stage2_best.pth`를 재평가했다.

| metric | value |
|---|---:|
| Accuracy | 0.5191 |
| Macro F1 | 0.3226 |
| Top-3 Accuracy | 0.8766 |
| Random Baseline | 0.1429 |

클래스별 결과는 다음과 같다.

| label | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| Floral | 0.63 | 0.69 | 0.66 | 1,097 |
| Woody | 0.50 | 0.54 | 0.52 | 946 |
| Amber_Oriental | 0.29 | 0.19 | 0.23 | 220 |
| Citrus | 0.38 | 0.35 | 0.37 | 207 |
| Sweet | 0.19 | 0.07 | 0.10 | 87 |
| Spicy | 0.07 | 0.04 | 0.05 | 51 |
| Fresh | 0.33 | 0.33 | 0.33 | 212 |

## 5. 결과 해석

Top-1 정확도는 51.91%다. 7개 클래스 랜덤 기준선인 14.29%보다는 충분히 높지만, Macro F1은 32.26%로 낮다. 이는 모델이 전체적으로는 다수 클래스를 어느 정도 맞히지만, 소수 클래스까지 균형 있게 맞히지는 못한다는 뜻이다.

`Floral`과 `Woody`는 상대적으로 성능이 높다. 반면 `Sweet`, `Spicy`, `Amber_Oriental`은 recall이 낮다. 이 클래스들은 샘플 수가 적거나 다른 계열과 note가 많이 겹치기 때문에 top-1 예측이 어렵다.

Top-3 Accuracy는 87.66%다. 따라서 모델을 "하나의 정답 계열을 확정하는 분류기"로 쓰기보다는, 향 계열 후보를 3개 정도 추천하는 방식으로 쓰면 실용성이 높다.

## 6. 한계

1. 향수 Note는 본질적으로 multi-label 성격이 강하다.
2. 현재 라벨은 single-label이어서 여러 향 계열을 동시에 가진 향수를 충분히 표현하지 못한다.
3. 병 이미지와 제품명/브랜드명만으로 실제 향 계열을 구분하는 데 한계가 있다.
4. 클래스 불균형 때문에 소수 클래스 성능이 낮다.
5. `notes`를 입력에 넣으면 정확도는 오를 수 있지만, 라벨 누수 문제가 생긴다.

## 7. 개선 방향

1. Top-1 분류보다 Top-3 후보 추천 방식으로 결과를 제공한다.
2. `Sweet`, `Spicy`, `Amber_Oriental` 오분류 샘플을 우선 검토한다.
3. 라벨을 single-label에서 multi-label로 바꾸는 방안을 검토한다.
4. `image only`, `image + brand/name`, `notes only`를 분리해 ablation 실험을 진행한다.
5. `notes only`는 최종 모델이 아니라 라벨 품질 진단용 baseline으로만 사용한다.

## 8. 재현 방법

테스트셋 평가:

```bash
cd perfume_classifier
../.venv/bin/python main.py --mode eval
```

검증셋 평가:

```bash
cd perfume_classifier
../.venv/bin/python main.py --mode eval --split val
```

학습 후 평가:

```bash
cd perfume_classifier
../.venv/bin/python main.py --mode train_eval
```

평가 결과는 `results/test_metrics.txt`에 저장된다.
