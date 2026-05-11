# PERFUME

향수 상품 이미지를 입력으로 받아 향수의 note 계열을 분류하는 프로젝트입니다. 현재 저장소 기준 파이프라인은 `data/All`에 있는 CSV를 읽어서 `perfume_classifier`에서 이미지 기반 분류를 학습하고 평가하는 구조입니다.

## Current Pipeline

1. 원본 메타데이터는 `data/raw/final_perfume_data.csv`, `data/raw/all_cleaned.csv`에 있습니다.
2. 현재 학습/평가에 쓰는 분할 데이터는 `data/All/train.csv`, `data/All/val.csv`, `data/All/test.csv`입니다.
3. `perfume_classifier/config.py`는 위 `data/All` CSV를 읽도록 설정되어 있습니다.
4. `perfume_classifier`는 현재 이미지와 `brand`, `name` 텍스트를 함께 사용해서 note 분류를 수행합니다.
5. 학습 결과는 `checkpoints/`, 평가 결과는 `results/`에 저장됩니다.

중요:
`notes` 컬럼은 CSV에 보관되어 있지만 현재 모델 입력으로 사용하지 않습니다. 최종 목표가 note 계열 예측이라면 `notes`를 입력에 넣는 것은 정답 근거를 보는 데이터 누수에 가까우므로, 회의 전후 실험에서는 진단용 baseline으로만 다룹니다.

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
│   │   └── test.csv
│   ├── brand/
│   ├── note/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── raw/
│       ├── all_cleaned.csv
│       └── final_perfume_data.csv
├── perfume_classifier/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── main.py
│   ├── model.py
│   ├── requirements.txt
│   ├── train.py
│   └── utils.py
├── perfume_images/
├── results/
└── README.md
```

## Data Format

현재 `data/All/*.csv`는 아래 컬럼을 가집니다.

| column | description |
|---|---|
| `image_path` | 향수 이미지 경로 |
| `label` | note 라벨 |
| `name` | 제품명 |
| `brand` | 브랜드명 |
| `notes` | 원본 note 문자열 |
| `image_url` | 원본 이미지 URL |

예시:

```csv
image_path,label,name,brand,notes,image_url
perfume_images\01577_Kyoto_Eau_de_Toilette.jpg,Woody,Kyoto Eau de Toilette,Comme des Garcons: Incense,"incense, cypress oil, coffee, teak wood, vetiver, patchouli, amber, everlasting flower, Virginian cedar",https://...
```

## Current Split Sizes

현재 `data/All` 기준 샘플 수는 아래와 같습니다.

| split | rows |
|---|---:|
| train | 22557 |
| val | 2820 |
| test | 2820 |

`data/All`에는 아래 7개 라벨이 들어 있습니다.

- `Floral`
- `Woody`
- `Amber_Oriental`
- `Citrus`
- `Sweet`
- `Spicy`
- `Fresh`

현재 학습용 증강 파일 `data/All/train_aug.csv`가 있으면 학습 시 원본 `train.csv` 대신 사용됩니다. 현재 `train_aug.csv`는 41061개 샘플을 포함합니다.

원본 전체 데이터(`data/raw/all_cleaned.csv`) 기준 클래스별 샘플 수:

| label | rows |
|---|---:|
| `Floral` | 10972 |
| `Woody` | 9457 |
| `Amber_Oriental` | 2198 |
| `Fresh` | 2118 |
| `Citrus` | 2064 |
| `Sweet` | 878 |
| `Spicy` | 510 |

## Install

루트가 아니라 `perfume_classifier/requirements.txt`를 사용합니다.

```bash
pip install -r perfume_classifier/requirements.txt
```

주요 패키지:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `Pillow`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Current Config

현재 기본 설정의 핵심은 아래와 같습니다.

- 데이터 경로: `data/All/train.csv`, `data/All/val.csv`, `data/All/test.csv`
- 이미지 루트: 저장소 루트
- backbone 기본값: `efficientnet_b0`
- batch size: `32`
- AMP: `False`
- weighted sampler: `True`

현재 경로 설정:

```python
class PathConfig:
    train_csv = os.path.join(BASE_DIR, "..", "data", "All", "train.csv")
    val_csv   = os.path.join(BASE_DIR, "..", "data", "All", "val.csv")
    test_csv  = os.path.join(BASE_DIR, "..", "data", "All", "test.csv")
```

## How To Run

가장 안전한 방법은 `perfume_classifier` 폴더로 들어가서 실행하는 것입니다.

```bash
cd perfume_classifier
```

학습:

```bash
python main.py --mode train
```

평가:

```bash
python main.py --mode eval
```

검증셋 평가:

```bash
python main.py --mode eval --split val
```

학습 후 바로 평가:

```bash
python main.py --mode train_eval
```

옵션 예시:

```bash
python main.py --mode train --backbone mobilenet_v3_large
python main.py --mode train --batch_size 16
python main.py --mode train --seed 123
```

## Model

현재 모델은 이미지 기반 단일 입력 분류기입니다.

- backbone: `EfficientNet-B0` 또는 `MobileNetV3-Large`
- pooling: `AdaptiveAvgPool2d`
- classifier head:
  - `BatchNorm1d`
  - `Dropout`
  - `Linear(1280 -> 256)`
  - `ReLU`
  - `BatchNorm1d`
  - `Dropout`
  - `Linear(256 -> num_classes)`

현재는 이미지 feature와 `brand`, `name` 텍스트 feature를 결합해 분류합니다. `notes`는 모델 입력으로 사용하지 않습니다.

## Training Strategy

현재 학습은 2단계입니다.

### Stage 1

- backbone freeze
- head만 학습
- optimizer: `Adam`

### Stage 2

- 처음에는 backbone 마지막 3개 block만 unfreeze
- 이후 일정 epoch에서 추가 unfreeze
- optimizer: `AdamW`
- scheduler: `CosineAnnealingLR`
- early stopping 사용

현재 코드상 stage2 unfreeze 스케줄:

- epoch 1: 마지막 3개 block
- epoch 6: 마지막 5개 block
- epoch 11: 마지막 9개 block

## Known Caveats

현재 파이프라인에서 주의할 점:

- `notes`는 라벨의 직접 근거가 될 수 있으므로 최종 모델 입력에 넣지 않는 것이 안전합니다.
- stage2에서 추가 unfreeze를 할 때 optimizer를 다시 만들지 않기 때문에, 특정 실행에서는 epoch 11 부근에서 `NaN`이 발생할 수 있습니다.
- 현재 결과를 보면 이미지와 제품명/브랜드만으로 note 계열을 top-1로 분류하는 난도가 높습니다.

## Evaluation Outputs

평가 시 아래 파일들이 생성됩니다.

```text
results/
├── training_curves.png
├── test_confusion_matrix.png
├── test_metrics.txt
└── test_per_class_accuracy.png
```

현재 `results/test_metrics.txt` 기준 최신 성능:

- Accuracy: `0.5191`
- Macro F1: `0.3226`
- Top-3 Accuracy: `0.8766`
- Random Baseline: `0.1429`

즉, 정답이 상위 3개 후보 안에는 자주 들어가지만 top-1 결정은 아직 불안정합니다. 제품 추천이나 검색 보조처럼 후보를 보여주는 사용 방식이라면 top-3 결과를 활용할 여지가 있습니다.

## Current Interpretation

현재 파이프라인의 해석은 아래에 가깝습니다.

- 병 이미지 자체에서 얻을 수 있는 정보가 note 라벨과 직접적으로 강하게 연결되지 않습니다.
- 흰 배경 상품 이미지 특성상 제품 모양, 라벨 디자인, 브랜드 스타일에 더 끌릴 가능성이 있습니다.
- `brand`, `name`을 함께 쓰더라도 향 계열 자체를 설명하는 단서는 제한적입니다.
- `Amber_Oriental`은 데이터 수가 `Citrus`, `Fresh`와 비슷하지만 정확도가 낮습니다. 현재 confusion matrix 기준 Amber 샘플은 `Amber_Oriental`로 맞는 비율이 19%이고, `Woody`로 41%, `Floral`로 28% 예측됩니다.
- Amber의 주요 note는 `vanilla`, `amber`, `musk`, `bergamot`, `patchouli`, `sandalwood` 등인데, 이 성분들이 `Woody`, `Floral`의 주요 성분과 많이 겹칩니다. 따라서 Amber 문제는 단순한 샘플 수 부족보다 클래스 경계와 입력 단서 부족의 영향이 커 보입니다.

## Meeting Decision Points

내일 회의에서 결정할 만한 방향은 아래와 같습니다.

1. 최종 제품 목표를 정합니다.
   - 이미지/브랜드/제품명만 보고 note 계열을 맞히는 모델이 목표라면 `notes`는 최종 입력에서 제외합니다.
   - 사용자가 note 정보를 이미 입력하거나 제공하는 서비스라면, 이 프로젝트는 note 예측이 아니라 note 기반 계열 정규화/검색 문제로 바뀝니다.

2. `notes only` baseline은 진단용으로만 실험합니다.
   - `notes`만으로도 Amber가 낮으면 라벨 기준 자체가 애매한 것입니다.
   - `notes`만으로 Amber가 높으면 라벨은 어느 정도 일관적이지만, 이미지/브랜드/제품명 입력에 Amber를 구분할 단서가 부족한 것입니다.
   - 이 결과는 최종 성능으로 보고하지 않고 oracle 또는 diagnostic baseline으로 분리합니다.

3. 입력 조합별 ablation을 먼저 비교합니다.
   - `image only`
   - `image + brand/name`
   - diagnostic: `notes only`
   - 목표는 큰 모델을 바로 바꾸기보다, 어떤 입력이 실제로 성능을 올리는지 확인하는 것입니다.

4. 라벨 구조를 다시 검토합니다.
   - Amber/Woody/Floral처럼 note가 겹치는 클래스는 단일 라벨보다 multi-label 또는 top-k 추천이 자연스러울 수 있습니다.
   - 특히 현재 Top-3 Accuracy가 87.66%라서, top-1 분류보다 후보 계열 3개를 보여주는 방식이 더 현실적일 수 있습니다.

5. 학습 전략은 단순한 증강 확대보다 비교 실험으로 정합니다.
   - 현재는 offline augmentation, WeightedRandomSampler, FocalLoss가 함께 들어가 있습니다.
   - 다음 실험에서는 원본 train + WeightedSampler, 증강 train + WeightedSampler, weighted CE/FocalLoss를 나눠 비교하는 편이 좋습니다.
   - Sweet/Spicy처럼 증강 후에도 낮은 클래스는 이미지 증강만으로 해결하기 어렵다는 신호일 수 있습니다.

## Next Ideas

다음 단계로 고려할 만한 방향:

- `image only`와 `image + brand/name` 성능 차이 확인하기
- diagnostic baseline으로 `notes only` 실험하기
- Amber/Woody/Floral 오분류 샘플을 직접 검토해 라벨 기준 점검하기
- top-1 분류 대신 top-3 후보 제안 방식 검토하기
- stage2 unfreeze 스케줄 단순화하기
- augmentation, sampler, loss 조합을 분리해서 ablation 실험하기
