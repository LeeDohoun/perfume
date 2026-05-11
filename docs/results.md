# Experiment Results

이 문서는 현재 데이터(`data/All`)와 찬준 브랜치 기반 `perfume_classifier` 파이프라인으로 재평가한 결과를 정리한다.

## 현재 기준

- 날짜: 2026-05-11
- 브랜치: `feature/dohoon`
- 데이터 기준: 찬준 브랜치에서 가져온 `data/All`
- 모델 코드: `perfume_classifier`
- 체크포인트: `checkpoints/stage2_best.pth`
- 입력: 향수 이미지 + `brand` + `name`
- 제외 입력: `notes`

`notes`는 라벨의 직접 근거가 될 수 있으므로 최종 모델 입력에는 사용하지 않는다.

## 데이터 분할

| split | rows |
|---|---:|
| train | 22,557 |
| val | 2,820 |
| test | 2,820 |
| train_aug | 41,061 |

클래스 수는 7개다.

| label | test rows |
|---|---:|
| Floral | 1,097 |
| Woody | 946 |
| Amber_Oriental | 220 |
| Fresh | 212 |
| Citrus | 207 |
| Sweet | 87 |
| Spicy | 51 |

## 결과 요약

| 모델 | 태스크 | 입력 | Test Accuracy | Macro F1 | Top-3 Accuracy |
|---|---|---|---:|---:|---:|
| EfficientNet-B0 + text encoder | Note | image + brand + name | 51.91% | 32.26% | 87.66% |

랜덤 기준선은 7클래스 기준 14.29%다.

## 클래스별 결과

| label | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| Floral | 0.63 | 0.69 | 0.66 | 1,097 |
| Woody | 0.50 | 0.54 | 0.52 | 946 |
| Amber_Oriental | 0.29 | 0.19 | 0.23 | 220 |
| Citrus | 0.38 | 0.35 | 0.37 | 207 |
| Sweet | 0.19 | 0.07 | 0.10 | 87 |
| Spicy | 0.07 | 0.04 | 0.05 | 51 |
| Fresh | 0.33 | 0.33 | 0.33 | 212 |

## 재현 명령

현재 체크포인트로 테스트셋을 재평가한다.

```bash
cd perfume_classifier
../.venv/bin/python main.py --mode eval
```

검증셋을 평가하려면 아래처럼 실행한다.

```bash
cd perfume_classifier
../.venv/bin/python main.py --mode eval --split val
```

학습부터 다시 돌리려면 아래 명령을 사용한다.

```bash
cd perfume_classifier
../.venv/bin/python main.py --mode train_eval
```

## 실행 중 조정한 부분

현재 macOS/샌드박스 환경에서 재평가가 가능하도록 다음 호환성 수정을 반영했다.

- CSV의 Windows식 경로(`perfume_images\...`)를 `os.sep` 기준으로 정규화
- DataLoader `num_workers=0`으로 설정
- `TORCH_HOME`을 프로젝트 내부 `.torch_cache`로 설정
- `matplotlib`/`seaborn`이 없으면 그래프 저장만 생략

## 해석

Top-1 Accuracy는 51.91%로 이전 6클래스 이미지 단독 실험보다 높다. 다만 Macro F1은 32.26%라서 소수 클래스 성능은 아직 약하다. 특히 `Sweet`, `Spicy`, `Amber_Oriental`은 recall이 낮고, 다수 클래스인 `Floral`, `Woody` 쪽으로 예측이 쏠리는 경향이 있다.

Top-3 Accuracy는 87.66%로 높다. 따라서 이 모델은 "정답 Note 하나를 단정"하는 용도보다, 향 계열 후보 3개를 추천하거나 검색 필터 후보를 보여주는 용도에 더 잘 맞는다.

데이터 추가 방식과 학습 방식의 세부 적합성 판단은 `docs/data_training_suitability.md`에 별도로 정리했다.
