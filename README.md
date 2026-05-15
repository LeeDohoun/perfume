# PERFUME

향수 상품 이미지와 간단한 메타데이터를 입력으로 받아 향수의 Note 계열을 분류하는 프로젝트입니다. 

## Current Pipeline

1. 학습/평가 데이터는 `data/All/train.csv`, `data/All/val.csv`, `data/All/test.csv`입니다.
2. 학습 시 `data/All/train_aug.csv`가 있으면 원본 train 대신 증강 train을 사용합니다.
3. 모델은 EfficientNet-B0 이미지 feature와 `brand`, `name` 텍스트 feature를 결합해 7개 Note 클래스를 예측합니다.
4. `notes` 컬럼은 CSV에 남아 있지만 모델 입력에는 사용하지 않습니다. Note 라벨의 근거라서 입력으로 쓰면 데이터 누수에 가깝습니다.
5. 평가 결과는 `results/test_metrics.txt`에 저장됩니다.

## Current Data

| split | rows |
|---|---:|
| train | 22,557 |
| val | 2,820 |
| test | 2,820 |
| train_aug | 41,061 |

현재 Note 클래스는 7개입니다.

- `Floral`
- `Woody`
- `Amber_Oriental`
- `Citrus`
- `Sweet`
- `Spicy`
- `Fresh`

테스트셋 분포:

| label | test rows |
|---|---:|
| Floral | 1,097 |
| Woody | 946 |
| Amber_Oriental | 220 |
| Fresh | 212 |
| Citrus | 207 |
| Sweet | 87 |
| Spicy | 51 |

## Latest Evaluation

실행일: 2026-05-11  
체크포인트: `checkpoints/stage2_best.pth`  
평가 split: `data/All/test.csv`

| metric | value |
|---|---:|
| Accuracy | 0.5191 |
| Macro F1 | 0.3226 |
| Top-3 Accuracy | 0.8766 |
| Random Baseline | 0.1429 |

클래스별 결과:

| label | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| Floral | 0.63 | 0.69 | 0.66 | 1,097 |
| Woody | 0.50 | 0.54 | 0.52 | 946 |
| Amber_Oriental | 0.29 | 0.19 | 0.23 | 220 |
| Citrus | 0.38 | 0.35 | 0.37 | 207 |
| Sweet | 0.19 | 0.07 | 0.10 | 87 |
| Spicy | 0.07 | 0.04 | 0.05 | 51 |
| Fresh | 0.33 | 0.33 | 0.33 | 212 |

해석하면 top-1 정확도는 약 52%이고, 정답이 상위 3개 후보 안에 들어가는 비율은 약 88%입니다. 추천/검색 보조처럼 후보를 보여주는 방식에서는 Top-3 결과가 특히 유용합니다.

데이터 추가 방식과 현재 학습 방식의 적합성 판단은 `docs/data_training_suitability.md`에 정리했습니다.

## How To Run

현재 체크포인트 재평가:

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

옵션 예시:

```bash
python main.py --mode train --backbone mobilenet_v3_large
python main.py --mode train --batch_size 16
python main.py --mode train --seed 123
```

## Project Structure

```text
perfume/
├── checkpoints/
│   ├── stage1_best.pth
│   └── stage2_best.pth
├── data/
│   ├── All/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── train_aug.csv
│   ├── brand/
│   ├── note/
│   └── raw/
├── perfume_classifier/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── main.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── perfume_images/
├── results/
│   └── test_metrics.txt
└── docs/
```

## Notes

- `checkpoints/`는 `.gitignore` 대상이라 Git에는 올라가지 않습니다.
- macOS에서도 CSV의 `perfume_images\...` 경로를 읽을 수 있도록 `perfume_classifier/dataset.py`에서 경로를 정규화합니다.
- 현재 환경에 `matplotlib`/`seaborn`이 없으면 그래프 저장은 건너뛰고, 정확도와 classification report는 정상 저장합니다.
