# 시행착오 및 문제 정리

정리일: 2026-05-13

이 문서는 프로젝트를 진행하면서 겪은 혼동, 문제, 시행착오와 최종 정리 방향을 따로 기록한다.

## 1. 모델이 여러 개라 현재 기준이 헷갈림

처음에는 프로젝트 안에 여러 실험 코드가 함께 남아 있어서, 현재 실제로 사용하는 모델이 무엇인지 불명확했다.

확인된 모델/실험은 다음과 같았다.

| 항목 | 내용 | 현재 상태 |
|---|---|---|
| `perfume_classifier` EfficientNet-B0 | image + `brand` + `name`으로 7개 Note 분류 | 최종 비교용 |
| `scripts/train_clip_finetune.py` CLIP | CLIP image-only 2단계 fine-tuning | 최종 비교용 |
| `scripts/eval_clip_note.py` CLIP | CLIP image embedding 기반 Note 분류 | baseline |
| 구 `scripts/train_efficientnet.py` | 예전 `data/note`, `data/brand` 실험용 | deprecated |
| `train_note_text.py` | metadata text baseline | 참고용 |
| Brand EfficientNet-B0 | 브랜드 분류 | 참고용 |
| SeokHyeon CLIP fusion | image + text + `notes` 사용 | leakage 가능성 있는 참고 결과 |

최종적으로 현재 프로젝트의 기준은 `data/All` 7클래스 Note 분류로 맞췄다.

## 2. 추가 데이터 기준으로 돌렸는지 확인이 필요했음

처음에는 `data/note`와 `data/All`이 같이 있어서 어떤 split을 쓰는지 헷갈렸다.

현재 최종 비교는 아래 데이터를 기준으로 정리했다.

| split | rows |
|---|---:|
| `data/All/train.csv` | 22,557 |
| `data/All/train_aug.csv` | 41,061 |
| `data/All/val.csv` | 2,820 |
| `data/All/test.csv` | 2,820 |

EfficientNet-B0는 현재 체크포인트를 `data/All/test.csv`로 평가했고, CLIP은 `data/All/train_aug.csv` 기준으로 image-only 2단계 fine-tuning까지 돌렸다. 별도로 CLIP image embedding linear probe도 baseline으로 남겼다.

## 3. EfficientNet 평가와 재학습을 구분해야 했음

EfficientNet-B0는 이번에 처음부터 새로 학습한 것이 아니라, 현재 프로젝트에 남아 있던 `checkpoints/stage2_best.pth`를 사용해 추가 데이터 기준 test set으로 재평가했다.

따라서 결과 표현은 다음처럼 구분해야 한다.

- 맞는 표현: 현재 추가 데이터 기준 EfficientNet-B0 체크포인트 재평가 결과
- 조심할 표현: EfficientNet-B0를 이번에 새로 학습한 결과

현재 성능:

| Metric | Value |
|---|---:|
| Accuracy | 0.5191 |
| Macro F1 | 0.3226 |
| Top-3 Accuracy | 0.8766 |

## 4. CLIP 결과가 석현 브랜치보다 낮아서 원인을 확인함

현재 CLIP image-only 결과는 석현 브랜치의 CLIP 결과보다 낮았다.

| 모델 | Accuracy | Macro F1 | Top-3 Accuracy |
|---|---:|---:|---:|
| 현재 CLIP image linear probe | 0.4723 | 0.2563 | 0.8106 |
| 현재 CLIP two-stage fine-tune | 0.4489 | 0.2815 | 0.8177 |
| 현재 CLIP two-stage fine-tune (MPS) | 0.4404 | 0.2894 | 0.8160 |
| 석현 브랜치 CLIP fusion | 0.6241 | 0.4663 | 0.9138 |

원인은 데이터 차이가 아니라 입력 방식 차이였다. `data/All` split 자체는 같았지만, 석현 브랜치는 `name + brand + notes` 텍스트 embedding과 이미지 embedding을 함께 사용했다.

`notes`에는 `Rose`, `Jasmine`, `Cedarwood`, `Bergamot`, `Vanilla`처럼 정답 Note 계열을 직접 암시하는 단어가 들어 있다. 그래서 성능이 높더라도 최종 공정 비교에는 넣기 어렵다.

정리:

- 석현 CLIP fusion은 참고/상한선 baseline으로 본다.
- 최종 비교에는 `notes`를 입력으로 쓰지 않는 모델만 넣는다.

## 5. `notes` 사용은 과적합보다 데이터 누수에 가까움

`notes`는 향 계열 라벨의 근거가 될 수 있다. 따라서 `notes`를 입력으로 넣으면 모델이 향수병 이미지나 일반 metadata를 학습한다기보다, 정답을 암시하는 텍스트를 보고 맞힐 가능성이 크다.

예시:

| notes 단어 | 암시되는 라벨 |
|---|---|
| Rose, Jasmine, Iris | Floral |
| Cedarwood, Sandalwood, Oud | Woody |
| Lemon, Bergamot, Orange | Citrus |
| Vanilla, Caramel, Honey | Sweet |

이 문제는 train 성능만 높은 전형적 overfitting이라기보다, test에도 정답 힌트가 들어가는 feature leakage 문제로 해석하는 것이 맞다.

## 6. Brand 분류 결과는 높지만 해석 범위가 제한됨

이전에 돌린 Brand EfficientNet-B0 결과가 남아 있었다.

| 항목 | 값 |
|---|---:|
| 브랜드 수 | 35 |
| train / val / test | 652 / 81 / 82 |
| Stage 2 Best Val Accuracy | 0.9753 |
| 기록된 Test Accuracy | 0.9512 |

하지만 이 결과는 전체 242개 브랜드 일반화 성능이 아니다. 샘플 수가 15개 이상인 35개 frequent brand만 남긴 closed-set 평가다.

따라서 보고서에는 다음처럼 써야 한다.

> Brand EfficientNet-B0: 35개 frequent brand closed-set Test Accuracy 95.12%

전체 브랜드를 모두 맞히는 모델처럼 해석하면 안 된다.

## 7. 결과 이미지가 흩어져 있어 발표용으로 보기 어려웠음

처음에는 metric 텍스트, confusion matrix, per-class accuracy가 각각 따로 저장되어 있었다. 발표나 보고서에 바로 넣기에는 한눈에 보기 어렵다.

그래서 모델별 요약 이미지를 새로 만들었다.

| 모델 | 요약 이미지 |
|---|---|
| EfficientNet-B0 | `results/efficientnet_b0/evaluation_summary.png` |
| CLIP two-stage fine-tune | `results/clip_finetune/evaluation_summary.png` |
| CLIP linear probe baseline | `results/clip_linear_probe/evaluation_summary.png` |
| 두 모델 비교 | `results/model_metric_comparison.png` |

## 8. 실행 환경 문제도 있었음

기본 `python`에는 `torch`가 없었고, 프로젝트의 `.venv`를 써야 했다.

또한 그래프 저장에 필요한 `matplotlib`, `seaborn`이 빠져 있어서 추가 설치가 필요했다.

정리된 실행 기준:

```bash
.venv/bin/python ...
```

그래프 캐시는 `.matplotlib_cache/`에 생기므로 `.gitignore`에 추가했다.

이후 GPU/MPS 여부를 확인했을 때, `.venv`와 `.venv-mps` 모두 기본 Codex 샌드박스에서는 `torch.backends.mps.is_available()`가 `False`로 나왔다. 하지만 같은 명령을 샌드박스 밖에서 실행하면 MPS가 정상 동작했다. 따라서 로컬 Apple Silicon GPU를 쓰려면 MPS 학습 명령은 샌드박스 밖 실행 권한으로 돌려야 한다.

MPS로 CLIP Stage 2를 더 길게 돌려 보니, train 성능은 계속 상승했지만 val Macro F1은 Stage 2 epoch 2의 0.2960 이후 개선되지 않았다. 그래서 단순히 GPU로 오래 학습한다고 결과가 크게 좋아지는 구조는 아니었고, CLIP image-only 방식은 class imbalance와 입력 정보 한계가 남아 있는 것으로 보인다.

## 9. 중간 산출물과 최종 산출물을 구분해야 했음

CLIP feature cache는 다시 만들 수 있는 중간 산출물인데, 크기가 크고 최종 보고에 필요하지 않다.

정리 기준:

| 항목 | 처리 |
|---|---|
| `results/clip_linear_probe/features/` | 재생성 가능하므로 삭제 및 ignore |
| `.matplotlib_cache/` | 실행 캐시라 삭제 및 ignore |
| `__pycache__/` | 삭제 |
| 루트 `results/test_confusion_matrix.png` 등 중복 파일 | 모델별 폴더로 정리 후 삭제 |

## 10. 최종 정리 방향

최종 보고/발표에서는 아래 두 모델만 같은 조건의 Note 분류 비교로 사용한다.

| 모델 | 입력 | 사용 이유 |
|---|---|---|
| EfficientNet-B0 + text encoder | image + `brand` + `name` | 현재 프로젝트 기본 모델 |
| CLIP two-stage fine-tune | image only | EfficientNet과 비슷하게 2단계 학습한 CLIP 비교 모델 |
| CLIP linear probe | image embedding | 이미지 기반 CLIP baseline |

아래 결과들은 참고용으로만 둔다.

- Brand EfficientNet-B0: Note 분류가 아니라 Brand closed-set 분류
- SeokHyeon CLIP fusion: `notes` 사용으로 leakage 가능성 있음
- 구 `data/note` 실험: 현재 7클래스 `data/All` 기준과 다름

## 한 줄 회고

이번 프로젝트의 가장 큰 시행착오는 "성능이 높은 모델"과 "공정하게 비교 가능한 모델"을 구분하는 일이었다. 특히 `notes`처럼 정답을 직접 암시하는 입력을 쓰면 수치는 좋아지지만, 실제 이미지/metadata 기반 Note 분류 성능으로 보기 어렵다. 따라서 최종 결과는 같은 데이터 기준, 같은 누수 방지 원칙으로 정리해야 한다.
