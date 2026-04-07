# PERFUME

향수 상품 이미지를 입력으로 받아 향수의 note 계열을 분류하는 프로젝트입니다. 현재 저장소 기준 파이프라인은 `data/All`에 있는 CSV를 읽어서 `perfume_classifier`에서 이미지 기반 분류를 학습하고 평가하는 구조입니다.

## Current Pipeline

1. 원본 메타데이터는 `data/raw/final_perfume_data.csv`, `data/raw/all_cleaned.csv`에 있습니다.
2. 현재 학습/평가에 쓰는 분할 데이터는 `data/All/train.csv`, `data/All/val.csv`, `data/All/test.csv`입니다.
3. `perfume_classifier/config.py`는 위 `data/All` CSV를 읽도록 설정되어 있습니다.
4. `perfume_classifier`는 현재 `image_path`와 `label`만 사용해서 note 분류를 수행합니다.
5. 학습 결과는 `checkpoints/`, 평가 결과는 `results/`에 저장됩니다.

중요:
`data/All` CSV에는 `brand` 컬럼이 들어 있지만, 현재 모델은 brand를 입력으로 사용하지 않습니다.

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
| train | 1653 |
| val | 207 |
| test | 207 |

`data/All`에는 아래 7개 라벨이 들어 있습니다.

- `Floral`
- `Woody`
- `Amber_Oriental`
- `Citrus`
- `Sweet`
- `Spicy`
- `Fresh`

중요:
현재 모델 설정 `perfume_classifier/config.py`의 `note_classes`에는 `Fresh`가 포함되어 있지 않습니다. 그래서 `dataset.py`에서 `Fresh` 샘플은 자동으로 제거됩니다.

실제로 현재 모델이 학습/평가에 쓰는 유효 샘플 수:

| split | usable rows |
|---|---:|
| train | 1617 |
| val | 202 |
| test | 203 |

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

현재는 `brand`, `name`, `notes` 컬럼을 입력으로 쓰지 않고, CSV 안에 보관만 하고 있습니다.

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

- `data/All`에는 `Fresh` 라벨이 있지만, 현재 모델은 6클래스만 사용합니다.
- `brand` 컬럼이 있어도 현재 모델 입력에는 연결되어 있지 않습니다.
- stage2에서 추가 unfreeze를 할 때 optimizer를 다시 만들지 않기 때문에, 특정 실행에서는 epoch 11 부근에서 `NaN`이 발생할 수 있습니다.
- 현재 결과를 보면 이미지 만으로 note를 분류하는 난도가 높아서 성능이 낮습니다.

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

- Accuracy: `0.1823`
- Macro F1: `0.1601`
- Top-3 Accuracy: `0.6256`
- Random Baseline: `0.1667`

즉, 현재 모델은 랜덤보다는 약간 높지만 아직 실용적인 수준은 아닙니다.

## Current Interpretation

현재 파이프라인의 해석은 아래에 가깝습니다.

- 병 이미지 자체에서 얻을 수 있는 정보가 note 라벨과 직접적으로 강하게 연결되지 않습니다.
- 흰 배경 상품 이미지 특성상 제품 모양, 라벨 디자인, 브랜드 스타일에 더 끌릴 가능성이 있습니다.
- `brand`나 텍스트 메타데이터를 함께 쓰는 멀티모달/멀티입력 방향이 다음 실험 후보입니다.

## Next Ideas

다음 단계로 고려할 만한 방향:

- `Fresh` 클래스를 실제 학습 클래스에 포함하기
- `image + brand` 입력으로 확장하기
- `image + brand + notes text` 조합 실험하기
- stage2 unfreeze 스케줄 단순화하기
- `val loss` 대신 `macro F1` 기준 체크포인트 저장 실험하기
