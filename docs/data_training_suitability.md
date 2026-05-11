# 데이터 추가 및 학습 방식 적합성 판단

검토일: 2026-05-11

## 결론

현재 추가된 `data/All` 데이터에 대해서는 기존 6클래스 `scripts/` 파이프라인보다 `perfume_classifier` 파이프라인이 더 적합하다. 이유는 현재 데이터가 7클래스 Note 분류, `brand/name` 텍스트 입력, `train_aug.csv` 기반 소수 클래스 증강을 전제로 구성되어 있기 때문이다.

다만 완전히 문제가 없는 것은 아니다. 현재 구조는 top-1 단일 분류보다 top-3 후보 추천에 더 적합하고, `Sweet`, `Spicy`, `Amber_Oriental` 같은 소수/경계 클래스는 추가 개선이 필요하다.

## 데이터 추가 방식 점검

현재 분할:

| split | rows |
|---|---:|
| train | 22,557 |
| val | 2,820 |
| test | 2,820 |
| train_aug | 41,061 |

분할 누수 점검:

| 기준 | train-val | train-test | val-test |
|---|---:|---:|---:|
| `image_path` 중복 | 0 | 0 | 0 |
| `name + brand + notes` 중복 | 0 | 0 | 0 |

이 기준에서는 train/val/test 사이의 직접 중복은 보이지 않는다.

## 증강 방식 판단

`train_aug.csv`는 train 원본 22,557개를 포함하고, 추가 증강 18,504개를 더해 총 41,061개로 구성되어 있다. 증강 경로는 `perfume_images/augmented` 아래에 있으며 val/test에는 섞이지 않는다.

증강 후 train 분포:

| label | train_aug rows |
|---|---:|
| Floral | 8,777 |
| Woody | 7,566 |
| Amber_Oriental | 5,274 |
| Fresh | 5,082 |
| Citrus | 4,953 |
| Sweet | 4,921 |
| Spicy | 4,488 |

판단:

- 소수 클래스 보강 목적에는 적합하다.
- val/test를 증강하지 않으므로 평가 오염 가능성은 낮다.
- 다만 offline augmentation과 WeightedRandomSampler, FocalLoss를 동시에 쓰고 있어 소수 클래스 보정이 과하게 걸릴 수 있다.
- 다음 실험에서는 `train_aug + sampler + focal loss`를 한 번에 고정하지 말고, 각각을 분리한 ablation이 필요하다.

## 기존 학습 방식 적합성

적합한 부분:

- 현재 데이터는 7클래스이므로 `perfume_classifier`의 7클래스 설정이 맞다.
- 이미지 단독보다 `brand/name` 텍스트를 함께 쓰는 현재 모델이 Note 분류에 더 현실적이다.
- 검증 기준을 accuracy가 아니라 macro F1로 checkpoint 저장하는 점은 클래스 불균형 상황에 적합하다.
- Stage 1 freeze 후 Stage 2 gradual unfreeze 방식은 데이터가 커진 현재도 무난하다.
- `notes`를 입력에서 제외한 것은 라벨 누수를 피하는 측면에서 적합하다.

주의할 부분:

- 현재 `data/raw/all_cleaned.csv`는 2,067행이고, `data/All` 총 28,197행의 원천과 맞지 않는다.
- 따라서 `resplit.py`를 그대로 실행하면 큰 데이터셋을 구 데이터 기준으로 축소할 위험이 있다.
- 이 위험을 줄이기 위해 현재 코드에서는 source CSV 행 수가 현재 split보다 현저히 작으면 resplit을 중단하도록 막았다.
- `Sweet`, `Spicy` recall이 낮아 top-1 성능만 보고 모델을 충분하다고 말하기 어렵다.

## 최신 평가 결과

체크포인트: `checkpoints/stage2_best.pth`  
평가 split: `data/All/test.csv`

| metric | value |
|---|---:|
| Accuracy | 0.5191 |
| Macro F1 | 0.3226 |
| Top-3 Accuracy | 0.8766 |
| Random Baseline | 0.1429 |

판단:

- top-1 분류 성능은 기본 기준선보다 충분히 높다.
- Macro F1이 낮아 클래스별 균형 성능은 아직 부족하다.
- Top-3 Accuracy가 높으므로, 실서비스나 발표에서는 "정답 하나"보다 "향 계열 후보 3개 추천"으로 해석하는 편이 더 적합하다.

## 최종 권장

1. 현재 데이터에는 `perfume_classifier` 파이프라인을 사용한다.
2. `scripts/train_efficientnet.py` 기반 구 6클래스 실험은 현재 데이터의 주 실험으로 쓰지 않는다.
3. `notes`는 최종 입력에서 제외하고, 진단용 baseline에만 사용한다.
4. 다음 개선은 모델 변경보다 ablation 우선으로 진행한다.
   - 원본 train vs train_aug
   - WeightedRandomSampler on/off
   - FocalLoss vs CrossEntropy
   - image only vs image + brand/name
5. 결과 보고는 Accuracy와 Top-3 Accuracy를 함께 제시하고, Macro F1과 소수 클래스 한계를 명확히 적는다.
