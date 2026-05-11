# 향수 Note 분류 모델 추천

현재 기준은 찬준 브랜치에서 가져온 `data/All` 7클래스 데이터와 `perfume_classifier` 파이프라인이다. 이전 6클래스 `scripts/` 실험 결과는 참고용으로만 본다.

## 현재 데이터 특성

| 항목 | 값 |
|---|---:|
| 클래스 수 | 7 |
| train | 22,557 |
| val | 2,820 |
| test | 2,820 |
| train_aug | 41,061 |

테스트셋은 `Floral`, `Woody`가 많고 `Sweet`, `Spicy`가 적다. 따라서 단순 accuracy만 보면 다수 클래스 성능이 크게 반영되므로, Macro F1과 클래스별 recall을 같이 봐야 한다.

## 현재 추천 모델

### 1. EfficientNet-B0 + brand/name text encoder

현재 기본 추천 모델이다.

| 항목 | 내용 |
|---|---|
| 입력 | image + `brand` + `name` |
| 제외 입력 | `notes` |
| backbone | EfficientNet-B0 |
| classifier | image feature와 text feature concat 후 7클래스 분류 |
| 최신 Test Accuracy | 51.91% |
| 최신 Macro F1 | 32.26% |
| 최신 Top-3 Accuracy | 87.66% |

장점:

- 현재 데이터와 체크포인트가 이미 맞춰져 있다.
- 이미지 단서와 제품명/브랜드 단서를 함께 사용한다.
- Top-3 성능이 높아 후보 추천 방식에 적합하다.

단점:

- `Sweet`, `Spicy`, `Amber_Oriental` 같은 소수/경계 클래스 성능이 낮다.
- top-1 단일 정답 분류기로 쓰기에는 클래스 불균형 영향을 많이 받는다.

### 2. MobileNetV3-Large

경량 비교 실험용 후보로 추천한다.

| 항목 | 내용 |
|---|---|
| 입력 | image + `brand` + `name` |
| 장점 | CPU 환경에서 상대적으로 가볍다 |
| 용도 | EfficientNet-B0 대비 속도/성능 비교 |

### 3. Notes-only diagnostic baseline

최종 모델이 아니라 진단용 baseline으로만 사용한다.

`notes`는 라벨 생성의 근거가 될 수 있으므로 입력에 넣으면 최종 성능으로 보고하기 어렵다. 다만 `notes only` 모델을 따로 돌리면 현재 라벨 자체가 얼마나 일관적인지 확인할 수 있다.

## 권장 평가 기준

| 지표 | 이유 |
|---|---|
| Accuracy | 전체 top-1 성능 확인 |
| Macro F1 | 소수 클래스까지 균형 있게 보는 지표 |
| Top-3 Accuracy | 추천/검색 후보 방식에 적합 |
| Per-class recall | `Sweet`, `Spicy`, `Amber_Oriental` 개선 확인 |

현재 최신 결과:

| metric | value |
|---|---:|
| Accuracy | 0.5191 |
| Macro F1 | 0.3226 |
| Top-3 Accuracy | 0.8766 |
| Random Baseline | 0.1429 |

## 실행 명령

현재 체크포인트 평가:

```bash
cd perfume_classifier
../.venv/bin/python main.py --mode eval
```

학습 후 평가:

```bash
cd perfume_classifier
../.venv/bin/python main.py --mode train_eval
```

backbone 비교:

```bash
cd perfume_classifier
../.venv/bin/python main.py --mode train_eval --backbone mobilenet_v3_large
```

## 다음 실험 우선순위

1. `image only`와 `image + brand/name` ablation을 분리한다.
2. `notes only`를 diagnostic baseline으로만 측정한다.
3. `Sweet`, `Spicy`, `Amber_Oriental` 오분류 샘플을 검토한다.
4. top-1 단일 분류 대신 top-3 후보 추천 UX를 고려한다.
5. single-label 대신 multi-label 라벨 구조를 검토한다.
