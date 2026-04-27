# 향수 분류 프로젝트 학습 가이드

이 문서는 이 프로젝트에서 데이터 전처리부터 모델 학습까지 어떤 흐름으로 진행되는지 초심자 기준으로 정리한 설명입니다.

## 1. 전체 흐름

이 프로젝트는 크게 3단계로 진행됩니다.

```text
1. 원본 데이터 정리
2. 이미지 모델 학습
3. 결과 평가 및 모델 저장
```

프로젝트에서 다루는 분류 문제는 두 가지입니다.

| 작업 | 의미 | 예시 정답 |
|---|---|---|
| Note 분류 | 향 계열 맞히기 | Floral, Woody, Sweet, Citrus |
| Brand 분류 | 브랜드 맞히기 | Le Labo, Xerjoff, Byredo |

원본 데이터에는 대략 이런 정보가 들어 있습니다.

```csv
image_path,label,name,brand,notes,image_url
perfume_images\00000_Tihota_Eau_de_Parfum.jpg,Sweet,Tihota Eau de Parfum,Indult,"Vanilla bean, musks",...
```

주요 컬럼의 의미는 다음과 같습니다.

| 컬럼 | 의미 |
|---|---|
| `image_path` | 향수 병 이미지 경로 |
| `brand` | 브랜드 이름 |
| `notes` | 바닐라, 장미, 우디 같은 향 구성 정보 |
| `description` | 상품 설명 |
| `label` | 모델이 맞혀야 하는 정답 |

## 2. 가상환경에서 실행하기

PowerShell에서 가상환경 활성화가 막히면, 활성화하지 않고도 아래처럼 직접 실행할 수 있습니다.

```powershell
.\venv\Scripts\python.exe scripts\prepare_data.py
```

가상환경을 활성화한 경우에는 이렇게 실행해도 됩니다.

```powershell
python scripts\prepare_data.py
```

## 3. 데이터 전처리

전처리 코드는 다음 파일에 있습니다.

```text
scripts/prepare_data.py
```

실행 명령은 다음과 같습니다.

```powershell
.\venv\Scripts\python.exe scripts\prepare_data.py
```

전처리 결과는 아래 위치에 저장됩니다.

```text
data/note/train.csv
data/note/val.csv
data/note/test.csv

data/brand/train.csv
data/brand/val.csv
data/brand/test.csv
```

### 3.1 Note 라벨 만들기

핵심 함수는 `notes_to_label()`입니다.

```python
def notes_to_label(notes: str):
    ...
```

이 함수는 `notes` 안에 들어 있는 단어를 보고 향 계열 라벨을 만듭니다.

예를 들어 코드에는 이런 규칙이 있습니다.

```python
LABEL_RULES = {
    "Citrus": ["bergamot", "lemon", "orange", "grapefruit"],
    "Floral": ["rose", "jasmine", "lily", "magnolia"],
    "Woody": ["sandalwood", "cedar", "vetiver", "oud"],
    "Sweet": ["vanilla", "caramel", "chocolate", "honey"],
}
```

즉, 다음과 같은 방식으로 정답을 만듭니다.

```text
notes에 rose, jasmine이 많다       -> Floral
notes에 vanilla, caramel이 많다    -> Sweet
notes에 cedar, sandalwood가 많다   -> Woody
```

중요한 점은 이 라벨이 사람이 하나하나 직접 붙인 정답이 아니라, 코드 규칙으로 만든 정답이라는 것입니다. 그래서 실제 향과 완전히 일치하지 않을 수 있습니다.

### 3.2 Train, Val, Test 분리

데이터는 세 부분으로 나뉩니다.

| 이름 | 의미 |
|---|---|
| `train` | 모델이 공부하는 데이터 |
| `val` | 학습 중간에 성능을 확인하는 데이터 |
| `test` | 마지막에 최종 실력을 확인하는 데이터 |

비율은 다음과 같습니다.

```text
train : val : test = 80 : 10 : 10
```

이 작업은 `split_and_save()` 함수에서 수행합니다.

```python
def split_and_save(df: pd.DataFrame, out_dir: str):
    ...
```

현재 Note 데이터의 학습 분포는 대략 다음과 같습니다.

| 클래스 | 학습 데이터 수 |
|---|---:|
| Floral | 254 |
| Woody | 253 |
| Amber_Oriental | 166 |
| Citrus | 153 |
| Sweet | 98 |
| Spicy | 82 |

Brand 데이터는 학습 데이터 652개, 브랜드 클래스 35개로 구성되어 있습니다.

## 4. 이미지 모델 학습

이미지 학습 코드는 다음 파일에 있습니다.

```text
scripts/train_efficientnet.py
```

Note 이미지 모델 학습:

```powershell
.\venv\Scripts\python.exe scripts\train_efficientnet.py --task note
```

Brand 이미지 모델 학습:

```powershell
.\venv\Scripts\python.exe scripts\train_efficientnet.py --task brand
```

## 5. Dataset 클래스

이미지를 읽는 핵심 코드는 `PerfumeDataset` 클래스입니다.

```python
class PerfumeDataset(Dataset):
    ...
```

이 클래스가 하는 일은 다음과 같습니다.

```text
1. CSV 한 줄 읽기
2. image_path로 이미지 열기
3. label을 숫자로 바꾸기
4. 모델에 이미지와 정답 전달
```

모델은 문자열 라벨을 그대로 이해하지 못합니다. 그래서 라벨을 숫자로 바꿉니다.

```text
Floral -> 0
Woody  -> 1
Sweet  -> 2
...
```

## 6. 이미지 전처리와 데이터 증강

이미지 변환은 `get_transforms()` 함수에서 정의합니다.

```python
def get_transforms():
    ...
```

학습할 때는 이미지에 약간의 변형을 줍니다.

```text
크기 조정
랜덤 자르기
좌우 뒤집기
조금 회전
색감 변화
```

이것을 `Data Augmentation`, 한국어로는 데이터 증강이라고 합니다.

목적은 모델이 특정 이미지 하나를 외우는 것이 아니라, 조금 다른 이미지가 들어와도 잘 맞히게 만드는 것입니다.

## 7. EfficientNet-B0 모델

모델 생성은 `build_model()` 함수에서 합니다.

```python
def build_model(num_classes, freeze_backbone=True):
    ...
```

이 프로젝트는 `EfficientNet-B0`를 사용합니다.

쉽게 말하면 EfficientNet-B0는 이미지 분류를 잘하는 이미 만들어진 모델입니다.

처음부터 모델을 새로 만들면 데이터가 많이 필요합니다. 그래서 이미 큰 데이터로 학습된 모델을 가져온 다음, 마지막 분류 부분만 우리 문제에 맞게 바꿉니다.

이 방식을 `Transfer Learning`, 한국어로는 전이학습이라고 합니다.

코드에서는 마지막 분류기를 아래처럼 교체합니다.

```python
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, num_classes),
)
```

의미는 다음과 같습니다.

```text
기존 모델의 마지막 분류기 제거
-> 우리 클래스 개수에 맞는 분류기로 교체
```

Note 분류라면 클래스 수는 6개이고, Brand 분류라면 클래스 수는 35개입니다.

## 8. 학습 방식

이 프로젝트는 모델을 두 단계로 학습합니다.

### 8.1 Stage 1

```text
기존 EfficientNet 부분은 고정
마지막 분류기만 학습
```

처음에는 이미 잘 학습된 부분을 건드리지 않고, 새로 붙인 마지막 분류기만 학습합니다.

### 8.2 Stage 2

```text
EfficientNet의 마지막 일부 블록도 같이 학습
```

이 단계에서는 모델 본체의 마지막 부분도 조금 풀어서 우리 데이터에 더 맞게 조정합니다.

이 작업은 `unfreeze_backbone()` 함수에서 합니다.

```python
def unfreeze_backbone(model, unfreeze_from=-3):
    ...
```

관련 용어는 다음과 같습니다.

| 용어 | 의미 |
|---|---|
| `backbone` | 이미지 특징을 뽑는 모델 본체 |
| `head` | 마지막 분류기 |
| `freeze` | 학습 중 값이 바뀌지 않게 고정 |
| `unfreeze` | 다시 학습되게 풀기 |
| `fine-tuning` | 이미 배운 모델을 내 데이터에 맞게 조금 더 조정 |

## 9. 실제 학습 루프

실제로 학습이 일어나는 핵심 함수는 `train_one_epoch()`입니다.

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    ...
```

한 번의 학습 과정은 다음과 같습니다.

```text
1. 이미지 묶음을 모델에 넣음
2. 모델이 예측함
3. 정답과 비교해서 loss 계산
4. loss를 줄이도록 모델 값을 수정
```

`loss`는 모델이 얼마나 틀렸는지를 나타내는 값입니다.

`accuracy`는 전체 중 몇 퍼센트를 맞혔는지를 나타내는 값입니다.

## 10. 클래스 불균형 보정

현재 Note 데이터는 클래스마다 개수가 다릅니다.

```text
Floral: 254
Woody: 253
Sweet: 98
Spicy: 82
```

그냥 학습하면 모델이 데이터가 많은 클래스 위주로 배울 수 있습니다.

그래서 `WeightedRandomSampler`를 사용합니다.

핵심 함수는 `make_weighted_sampler()`입니다.

```python
def make_weighted_sampler(dataset):
    ...
```

의미는 다음과 같습니다.

```text
데이터가 적은 클래스는 더 자주 뽑기
데이터가 많은 클래스는 상대적으로 덜 뽑기
```

## 11. 모델 저장

학습이 끝난 모델은 아래 위치에 저장됩니다.

```text
checkpoints/note/best_model.pth
checkpoints/brand/best_model.pth
```

`.pth`는 PyTorch 모델 저장 파일입니다.

## 12. Note는 이미지보다 텍스트 모델이 더 잘 맞음

이 프로젝트에서 중요한 점은 Note 분류가 이미지로는 어렵다는 것입니다.

향수 병 이미지만 보고 그 향이 `Woody`인지 `Sweet`인지 맞히기는 어렵습니다. 병 디자인은 브랜드 정보와는 관련이 많지만, 실제 향 계열과는 직접적인 관련이 약할 수 있습니다.

그래서 Note 분류에는 텍스트 기반 모델도 있습니다.

코드는 다음 파일입니다.

```text
scripts/train_note_text.py
```

실행 명령은 다음과 같습니다.

```powershell
.\venv\Scripts\python.exe scripts\train_note_text.py
```

이 모델은 이미지를 사용하지 않고 다음 정보를 사용합니다.

```text
name + brand + description
```

하지만 `notes`는 일부러 사용하지 않습니다.

이유는 `notes`로 정답 라벨을 만들었기 때문입니다.

만약 `notes`를 입력으로 사용하면 사실상 답을 알려주는 것과 비슷합니다.

예를 들어:

```text
입력: vanilla, caramel, honey
정답: Sweet
```

이런 상황은 모델이 진짜로 일반화해서 배운 것이 아니라, 답에 가까운 정보를 그대로 본 것입니다.

이런 문제를 `data leakage`, 한국어로는 데이터 누수라고 합니다.

## 13. 텍스트 모델 구조

텍스트 모델의 핵심은 `build_model()` 함수입니다.

```python
def build_model(c_value):
    return Pipeline([
        ("tfidf", TfidfVectorizer(...)),
        ("clf", LogisticRegression(...)),
    ])
```

여기서 사용하는 두 가지 핵심 도구는 다음과 같습니다.

| 도구 | 의미 |
|---|---|
| `TF-IDF` | 문장을 숫자로 바꾸는 방법 |
| `LogisticRegression` | 숫자를 보고 클래스를 고르는 분류 모델 |

예를 들어 설명문에 `rose`, `jasmine`, `flower` 같은 단어가 자주 나오면 Floral 쪽 점수가 올라갈 수 있습니다.

## 14. 추천 실행 순서

처음부터 다시 실행한다면 아래 순서로 진행하면 됩니다.

### 14.1 데이터 전처리

```powershell
.\venv\Scripts\python.exe scripts\prepare_data.py
```

### 14.2 이미지 기반 Note 학습

```powershell
.\venv\Scripts\python.exe scripts\train_efficientnet.py --task note
```

### 14.3 이미지 기반 Brand 학습

```powershell
.\venv\Scripts\python.exe scripts\train_efficientnet.py --task brand
```

### 14.4 텍스트 기반 Note 학습

```powershell
.\venv\Scripts\python.exe scripts\train_note_text.py
```

CPU만 사용하는 환경이라면 EfficientNet 학습은 느릴 수 있습니다.

먼저 구조를 이해하고 빠르게 결과를 보고 싶다면 `train_note_text.py`부터 실행하는 것을 추천합니다.

## 15. 핵심 요약

이 프로젝트는 `CSV + 이미지` 데이터를 정리해서 `train`, `val`, `test`로 나눈 뒤, EfficientNet으로 향수 병 이미지를 분류합니다.

Brand 분류는 병 디자인, 로고, 패키지 특징이 브랜드와 관련이 있어서 이미지 모델이 잘 맞을 가능성이 높습니다.

반면 Note 분류는 병 이미지만으로 향 계열을 알기 어려워서 이미지 모델의 한계가 있습니다. 그래서 `name + brand + description`을 사용하는 텍스트 모델이 더 적합할 수 있습니다.

