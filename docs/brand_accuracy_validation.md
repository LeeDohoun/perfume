# 브랜드 분류 정확도 검증

검증일: 2026-05-05

## 핵심 결론

현재 기록된 브랜드 분류 Test Accuracy 95.12%는 전체 242개 브랜드에 대한 성능이 아니다. `scripts/prepare_data.py`에서 샘플 수가 15개 이상인 브랜드만 남긴 뒤, 그 35개 브랜드 안에서 train/val/test를 나눈 closed-set 평가 성능이다.

따라서 이 점수는 다음처럼 해석해야 한다.

- 맞는 해석: 35개 빈도 높은 브랜드가 모두 학습에 등장한 상태에서, 새 향수병 이미지를 그 35개 중 하나로 맞히는 성능
- 틀린 해석: 전체 242개 브랜드 또는 처음 보는 희소 브랜드까지 맞히는 일반 브랜드 분류 성능

## 데이터 범위

`data/raw/all_cleaned.csv` 기준:

| 항목 | 값 |
|---|---:|
| 전체 샘플 | 2,067 |
| 전체 브랜드 수 | 242 |
| 평가에 남긴 브랜드 수 | 35 |
| 평가에 남긴 샘플 수 | 815 |
| 평가에서 제외된 브랜드 수 | 207 |
| 평가에서 제외된 샘플 수 | 1,252 |

샘플 기준으로도 전체의 39.43%만 브랜드 분류 데이터에 남아 있고, 60.57%는 희소 브랜드라 제외되어 있다. 제외된 브랜드를 모두 `unknown/unsupported`로 보면, 현재 95.12%를 전체 원본 데이터 관점으로 환산했을 때 이론상 상한은 약 37.50%다.

## Test Set 구조

`data/brand/test.csv` 기준:

| 항목 | 값 |
|---|---:|
| Test 샘플 수 | 82 |
| Test 브랜드 수 | 35 |
| 브랜드별 Test 샘플 수 | 1-4 |
| 랜덤 기준선 | 2.86% |
| 최빈 클래스 기준선 | 4.88% |
| 95.12%에서 맞힌 개수 | 78 / 82 |
| 95.12%에서 틀린 개수 | 4 / 82 |

이 구조에서는 특정 다수 브랜드가 Test Accuracy를 크게 지배한다고 보기는 어렵다. Test split이 stratified라서 브랜드별 테스트 샘플이 1-4개 수준으로 작고 비교적 분산되어 있다. 즉 95.12%는 "상위 몇 개 브랜드 샘플만 많이 맞혀서 나온 점수"라기보다는, 선택된 35개 브랜드 안에서는 대부분의 브랜드를 잘 맞힌 결과에 가깝다.

다만 클래스당 테스트 샘플이 너무 적다. 브랜드 하나에 Test 샘플이 1개뿐인 경우도 8개라서, 브랜드별 성능을 안정적으로 말하기에는 Test set이 작다.

## Split 누수 점검

정확히 같은 샘플이 train/val/test에 중복으로 들어간 흔적은 보이지 않았다.

| 기준 | train-val | train-test | val-test |
|---|---:|---:|---:|
| `image_path` | 0 | 0 | 0 |
| `image_url` | 0 | 0 | 0 |
| `name` | 0 | 0 | 1 |

`name` 중복 1건은 `New York Intense Eau de Parfum`인데, 서로 다른 브랜드와 서로 다른 이미지 URL이다. 현재 확인한 범위에서는 이미지 자체의 exact duplicate leakage는 없다.

## 과적합 여부 판단

현재 작업공간에는 `checkpoints/brand/best_model.pth`가 없어서 저장된 모델의 예측값을 다시 뽑아 per-brand confusion matrix까지 재검증할 수는 없었다.

그래도 `docs/results.md`에 기록된 수치만 보면:

| 지표 | 값 |
|---|---:|
| Stage 1 Best Val Accuracy | 96.30% |
| Stage 2 Best Val Accuracy | 97.53% |
| Test Accuracy | 95.12% |

Val과 Test가 비슷하므로, 선택된 35개 브랜드 안에서의 전형적인 train overfitting 신호는 강하지 않다. 하지만 더 중요한 한계는 overfitting보다 평가 범위 제한이다. 모델은 전체 브랜드 세계를 배운 것이 아니라, 로고/라벨/병 형태가 반복되는 35개 frequent brand의 시각적 패턴을 배운 것으로 보는 게 맞다.

브랜드 분류에서는 이 자체가 부정적인 것은 아니다. 향수병 이미지에는 브랜드 로고와 패키지 디자인이 직접 들어 있으므로 Note 분류보다 훨씬 쉬운 태스크다. 다만 결과 보고에서는 반드시 "35개 frequent brand closed-set"이라는 조건을 붙여야 한다.

## 최종 판정

1. "많은 브랜드만 검증해서 높게 나온 것인가?"
   - 일부 맞다. 정확히는 많은 샘플을 가진 브랜드만 남겨서, 전체 242개 브랜드 중 35개만 평가했다.

2. "Test가 상위 브랜드에 치우쳐서 accuracy가 부풀었는가?"
   - 현재 split 안에서는 크게 그렇지 않다. Test는 브랜드당 1-4개라 최빈 클래스 baseline도 4.88%뿐이다.

3. "과적합인가?"
   - 저장 체크포인트가 없어 확정은 어렵다. 다만 기록된 Val 97.53%, Test 95.12%의 차이만 보면 closed-set 내부의 과적합 증거는 약하다.

4. "현재 점수를 어떻게 써야 하는가?"
   - `Brand EfficientNet-B0: 35개 frequent brand closed-set Test Accuracy 95.12%`
   - 전체 브랜드 일반화 성능처럼 쓰면 안 된다.

## 재현 명령

분포와 split 검증은 아래 명령으로 다시 확인할 수 있다.

```powershell
.\venv\Scripts\python.exe scripts\analyze_brand_accuracy.py
```

저장된 체크포인트가 있는 환경에서는 아래 명령으로 실제 모델 평가를 다시 돌린다.

```powershell
.\venv\Scripts\python.exe scripts\eval_efficientnet.py --task brand --device cpu
```

추가로 제대로 검증하려면 체크포인트 평가 결과에서 per-brand precision/recall, macro F1, confusion matrix를 저장하고, train 샘플 수가 12-15개인 브랜드와 25개 이상인 브랜드의 성능을 따로 비교해야 한다.
