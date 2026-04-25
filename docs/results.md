# Experiment Results

이 문서는 현재 프로젝트 상태에서 얻은 모델 결과와, 같은 결과를 나중에 다시 확인하기 위한 실행 방법을 정리한다.

## 현재 기준

- 날짜: 2026-04-25
- 브랜치: `feature/dohoon`
- Python 환경: `.venv`
- 주요 패키지:
  - `torch 2.11.0`
  - `torchvision 0.26.0`
  - `pandas 3.0.2`
  - `scikit-learn 1.8.0`
  - `numpy 1.26.4`
  - `open_clip_torch 3.3.0`

## 데이터 분할

| 태스크 | Train | Val | Test | 클래스 수 |
|--------|------:|----:|-----:|----------:|
| Note | 1,006 | 126 | 126 | 6 |
| Brand | 652 | 81 | 82 | 35 |

데이터는 `scripts/prepare_data.py` 실행 결과인 `data/note/*.csv`, `data/brand/*.csv`를 기준으로 한다.

```bash
.venv/bin/python scripts/prepare_data.py
```

## 데이터 처리 및 변형 내역

현재 결과를 만들기 위해 데이터에는 아래 처리를 적용했다.

### 원본 데이터

- 원본 CSV: `data/raw/final_perfume_data.csv`
- 이미지 다운로드 완료 CSV: `data/raw/all_cleaned.csv`
- 이미지 폴더: `perfume_images/`
- 원본 향수병 이미지 파일 자체는 새 이미지로 변형해서 저장하지 않았다.

### 공통 처리

- `data/raw/all_cleaned.csv`를 기준 데이터로 사용했다.
- macOS에서 이미지 경로가 깨지지 않도록 `perfume_images\...` 형태를 `perfume_images/...` 형태로 정규화했다.
- `data/raw/final_perfume_data.csv`에만 있던 `Description` 컬럼을 `image_url` 기준으로 다시 병합했다.
- 병합 후 split CSV에는 필요한 메타데이터를 보존했다.
  - `image_path`
  - `label`
  - `name`
  - `brand`
  - `description`
  - `image_url`

### Note 데이터 처리

- `notes` 문자열을 콤마로 나누고, 키워드 규칙으로 6개 계열 라벨을 생성했다.
- 2개 이상 계열이 동점으로 잡히는 샘플은 제거했다.
- 어떤 계열에도 매칭되지 않는 샘플도 제거했다.
- 기존 `Fresh` 계열은 샘플 수가 너무 적어 별도 클래스로 사용하지 않았다.
  - `lavender`는 `Floral` 규칙에 포함했다.
  - `lemongrass`는 `Citrus` 규칙에 포함했다.
- 클래스당 20개 미만인 라벨은 제거했다.
- 최종 Note 데이터는 6클래스, 총 1,258개다.
- `notes`는 라벨 생성에는 사용했지만, Note 텍스트 모델 입력에는 사용하지 않았다.
  - 이유: `notes`는 라벨을 만든 원천이므로 입력으로 쓰면 라벨 누수가 된다.

### Brand 데이터 처리

- `brand` 컬럼을 라벨로 사용했다.
- 샘플 수가 15개 이상인 브랜드만 남겼다.
- 최종 Brand 데이터는 35클래스, 총 815개다.

### 데이터 분할

- Note와 Brand 모두 Stratified 80 / 10 / 10 split을 사용했다.
- `random_state=42`로 분할했다.
- 출력 파일:
  - `data/note/train.csv`
  - `data/note/val.csv`
  - `data/note/test.csv`
  - `data/brand/train.csv`
  - `data/brand/val.csv`
  - `data/brand/test.csv`

### 학습 중 이미지 증강

EfficientNet-B0 학습에서는 이미지 증강을 적용했다. 단, 증강 이미지를 파일로 따로 저장하지 않고 학습 중에만 적용했다.

- `Resize`
- `RandomCrop`
- `RandomHorizontalFlip`
- `RandomRotation`
- `ColorJitter`
- `RandomAffine`
- `Normalize`

## 결과 요약

| 모델 | 태스크 | 목표 Test Accuracy | 현재 Test Accuracy | 판정 |
|------|--------|--------------------|--------------------|------|
| EfficientNet-B0 | Note | 35 ~ 50% | 24.60% | 목표 미달 |
| EfficientNet-B0 | Brand | 40 ~ 65% | 95.12% | 목표 상한 초과 |
| CLIP ViT-B/32 linear probe | Note | 45 ~ 60% | 26.98% | 목표 미달 |
| TF-IDF + LogisticRegression | Note | 45 ~ 60% | 59.52% | 목표 범위 도달 |

## 재현 명령

### EfficientNet-B0 Note 학습

```bash
.venv/bin/python scripts/train_efficientnet.py --task note --batch_size 32
```

결과:

- Stage 1 Best Val Accuracy: 33.33%
- Stage 2 Best Val Accuracy: 31.75%
- Test Accuracy: 24.60%
- 저장 위치: `checkpoints/note/best_model.pth`

### EfficientNet-B0 Brand 학습

```bash
.venv/bin/python -u scripts/train_efficientnet.py --task brand --batch_size 32
```

결과:

- Stage 1 Best Val Accuracy: 96.30%
- Stage 2 Best Val Accuracy: 97.53%
- Test Accuracy: 95.12%
- 저장 위치: `checkpoints/brand/best_model.pth`

### CLIP Note 평가

```bash
.venv/bin/python -u scripts/eval_clip_note.py --mode linear_probe --split test --batch_size 64 --device cpu
```

결과:

- Linear probe best alpha: `0.03`
- Val Accuracy: 26.19%
- Test Accuracy: 26.98%

### Note 정확도 개선 모델

```bash
.venv/bin/python scripts/train_note_text.py
```

입력:

- `name`
- `brand`
- `description`

사용하지 않는 입력:

- `notes`

`notes`는 Note 라벨을 만든 원천이므로 입력으로 사용하지 않는다.

결과:

- Best C: `1.0`
- Val Accuracy: 58.73%
- Test Accuracy: 59.52%
- 저장 위치: `checkpoints/note_text/best_model.joblib`

## 저장된 체크포인트 재평가

학습을 다시 돌리면 증강, 샘플링, 연산 장치 차이 때문에 수치가 약간 달라질 수 있다. 현재 저장된 EfficientNet-B0 체크포인트를 그대로 평가하려면 아래 명령을 사용한다.

```bash
.venv/bin/python scripts/eval_efficientnet.py --task note --device cpu
.venv/bin/python scripts/eval_efficientnet.py --task brand --device cpu
.venv/bin/python scripts/eval_note_text.py
```

체크포인트는 `.gitignore`에 의해 Git에는 포함되지 않는다. 다른 컴퓨터에서 같은 결과를 보려면 아래 파일도 함께 복사해야 한다.

- `checkpoints/note/best_model.pth`
- `checkpoints/brand/best_model.pth`
- `checkpoints/note_text/best_model.joblib`

## Note 태스크의 문제점

### 1. 이미지와 라벨의 정보 연결이 약함

현재 Note 분류는 향수병 이미지만 입력으로 사용한다. 그러나 라벨은 `Notes` 텍스트에서 만들어진 향 계열이다. 병 디자인만 보고 실제 향이 Floral, Woody, Sweet인지 맞히는 것은 정보적으로 어렵다.

Brand 분류가 95.12%까지 오른 이유는 로고, 병 모양, 라벨 디자인이 브랜드와 직접 연결되기 때문이다. Note 분류는 이런 직접 단서가 약하다.

### 2. 규칙 기반 라벨이라 노이즈가 있음

Note 라벨은 수동 정답이 아니라 키워드 규칙으로 만든 라벨이다. 예를 들어 한 향수가 `rose`, `oud`, `vanilla`, `amber`를 동시에 갖고 있으면 여러 계열에 해당하지만, 현재는 하나의 대표 라벨만 선택한다. 이 과정에서 애매하거나 틀린 라벨이 생길 수 있다.

### 3. 향수는 본질적으로 multi-label 문제임

향수 하나는 보통 여러 향 계열을 동시에 가진다. 현재는 single-label 분류로 단순화되어 있어 실제 문제 구조와 맞지 않는다. 이 때문에 모델 입장에서는 같은 이미지/비슷한 제품인데 라벨 기준이 불안정하게 보일 수 있다.

### 4. 클래스별 데이터가 적고 불균형함

Note 학습 데이터는 총 1,006개이고, `Sweet`, `Spicy` 쪽은 상대적으로 적다. 테스트셋도 클래스당 10~32개 수준이라 정확도 변동이 크다.

### 5. 학습 로그상 과적합이 보임

EfficientNet-B0 Note 학습에서 Train Accuracy는 65% 근처까지 올라갔지만, Val Accuracy는 31.75%, Test Accuracy는 24.60%에 머물렀다. 즉 이미지에서 일반화 가능한 Note 단서를 잘 배우지 못하고 있다.

### 6. CLIP도 이미지 단독으로는 한계가 있음

CLIP linear probe도 Test Accuracy 26.98%에 그쳤다. CLIP의 장점은 이미지-텍스트 연결인데, 현재 평가는 최종 입력을 이미지 중심으로 사용한다. `name`, `brand`, `description` 같은 텍스트를 함께 쓰지 않으면 Note 목표 정확도까지 올리기 어렵다.

## 정확도 상승 방향

현재 적용한 해결책은 텍스트 메타데이터 활용이다. `image` 단독 대신 `name + brand + description`을 사용한 TF-IDF + LogisticRegression 모델이 Note Test Accuracy 59.52%를 달성했다.

1. Note 태스크를 `image + name + brand + description` 멀티모달 입력으로 바꾼다.
2. `notes`를 입력으로 쓰는 것은 라벨 누수이므로 피하거나 별도 실험으로 분리한다.
3. Note 라벨을 single-label에서 multi-label로 바꾼다.
4. 클래스별 대표 샘플을 수동 검수해서 라벨 노이즈를 줄인다.
5. Note 전용으로 더 보수적인 fine-tuning을 적용한다.
   - unfreeze 범위 축소
   - learning rate 감소
   - early stopping 강화
   - class-weight loss 적용
6. 결과 보고 시 Brand는 “시각 단서가 강한 태스크”, Note는 “이미지 단독으로는 한계가 있는 태스크”로 분리해서 해석한다.
