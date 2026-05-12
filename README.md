# Perfume CLIP Classifier

CLIP 기반 향수 계열 분류 프로젝트입니다. `data/All` 원본 CSV와 `perfume_images` 이미지를 사용해 향수 계열을 분류합니다. 현재는 원본 라벨을 4개 계열로 줄인 `data/modified` 데이터셋과 그 학습 결과도 함께 관리합니다.

## 1. 프로젝트 구조

- `data/All`: 원본 train/val/test CSV
- `data/modified`: 4분류로 변환한 train/val/test CSV와 변환 스크립트
- `data_classifier`: CLIP 분류기 학습 코드
- `perfume_images`: 향수 이미지
- `results/clip_classifier`: 일반 학습 결과 저장 위치
- `results/clip_embedding_cache`: CLIP embedding cache 저장 위치
- `results/modified`: 4분류 실험 결과

## 2. 기본 학습 코드

학습 스크립트는 `data_classifier/train_clip_classifier.py`입니다.

기본 입력:

- 학습 데이터: `data/All/train.csv`
- 검증 데이터: `data/All/val.csv`
- 테스트 데이터: `data/All/test.csv`
- 결과 저장 위치: `results/clip_classifier/<실행시간>`
- CLIP embedding cache 위치: `results/clip_embedding_cache`

기본 실행:

```bash
python data_classifier/train_clip_classifier.py
```

기본 설정은 CLIP 모델을 freeze하고, 이미지 embedding과 텍스트 embedding을 합친 뒤 classifier만 학습합니다.

## 3. 주요 옵션

```bash
python data_classifier/train_clip_classifier.py --batch-size 64
python data_classifier/train_clip_classifier.py --weighted-loss
python data_classifier/train_clip_classifier.py --rebuild-embedding-cache
python data_classifier/train_clip_classifier.py --no-cache-embeddings
python data_classifier/train_clip_classifier.py --train-clip --lr 1e-5
```

- `--batch-size`: batch 크기 조정
- `--weighted-loss`: 클래스 불균형을 loss에 반영
- `--rebuild-embedding-cache`: 기존 embedding cache를 무시하고 새로 추출
- `--no-cache-embeddings`: cache 없이 매 epoch CLIP을 직접 실행
- `--train-clip`: CLIP까지 fine-tuning
- `--log-every`: batch 진행 로그 출력 간격

`--train-clip`은 속도가 느리고 GPU 메모리를 많이 사용합니다. 먼저 기본 cache 방식으로 classifier 성능을 확인한 뒤 사용하는 것을 권장합니다.

## 4. Embedding Cache 방식

기본 실행은 CLIP embedding cache를 사용합니다.

첫 실행:

```text
이미지/텍스트 로드
-> CLIP image/text embedding 추출
-> results/clip_embedding_cache에 저장
-> 저장된 embedding으로 classifier 학습
```

두 번째 실행부터:

```text
저장된 embedding 로드
-> classifier만 학습
```

첫 실행은 embedding 추출 때문에 시간이 걸리지만, 이후 반복 실험은 훨씬 빨라집니다.

## 5. 성능 평가 척도

학습과 테스트 결과에는 다음 지표가 포함됩니다.

- Accuracy: 가장 높은 확률의 예측 1개가 정답이면 성공
- Top-3 Accuracy: 확률 상위 3개 예측 안에 정답이 있으면 성공
- Precision, Recall, F1-score: 클래스별 분류 성능
- Confusion Matrix: 실제 클래스와 예측 클래스의 혼동 정도

Top-3 Accuracy는 향수 계열처럼 계열 경계가 겹치는 분류 문제에서 유용합니다. 예를 들어 실제 정답이 `Woody`이고 모델의 상위 3개 예측에 `Woody`가 포함되면 Top-3에서는 맞은 것으로 계산합니다.

## 6. 기본 결과 파일

일반 학습 결과는 `results/clip_classifier/<실행시간>`에 저장됩니다.

- `best_clip_classifier.pt`: 가장 좋은 validation accuracy의 모델
- `summary.json`: 최고 epoch, validation accuracy, test accuracy, test top-3 accuracy 요약
- `training_history.csv`: epoch별 loss, accuracy, top-3 accuracy
- `classification_report.txt`: test accuracy, test top-3 accuracy, 클래스별 precision/recall/f1-score
- `test_predictions.csv`: 테스트 샘플별 예측 결과, confidence, top-3 예측
- `training_curves.png`: loss, accuracy, top-3 accuracy 학습 곡선
- `confusion_matrix.png`: confusion matrix
- `label_map.json`: 클래스 이름과 인덱스 매핑

## 7. 기본 실험 결과

기본 실험 결과는 `results` 루트에 저장되어 있습니다.

- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/training_curves.png`
- `results/accuracy_improvement_suggestions.txt`

`results/classification_report.txt` 기준 성능:

| Metric | Value |
|---|---:|
| Test accuracy | 0.6241 |
| Test top-3 accuracy | 0.9138 |
| Macro F1-score | 0.4663 |
| Weighted F1-score | 0.6326 |

클래스별 성능:

| Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Amber_Oriental | 0.3513 | 0.5045 | 0.4142 | 220 |
| Citrus | 0.4060 | 0.5217 | 0.4567 | 207 |
| Floral | 0.7827 | 0.7484 | 0.7651 | 1097 |
| Fresh | 0.4516 | 0.6604 | 0.5364 | 212 |
| Spicy | 0.1463 | 0.1176 | 0.1304 | 51 |
| Sweet | 0.2990 | 0.3333 | 0.3152 | 87 |
| Woody | 0.7355 | 0.5761 | 0.6461 | 946 |

기본 실험은 7개 라벨을 그대로 사용했습니다. `Floral`과 `Woody`는 상대적으로 높게 나오지만, `Spicy`, `Sweet`, `Amber_Oriental`, `Citrus`는 데이터 수와 계열 경계 문제 때문에 낮은 성능을 보입니다. 이 결과를 근거로 향 계열을 4개로 줄이는 실험을 진행했습니다.

## 8. 현재 변경사항 요약

현재 실험에서는 기본 학습 코드에 다음 변경사항을 적용했습니다.

- CLIP 텍스트 입력을 향수명/브랜드가 아니라 `notes`만 사용하도록 수정
- 기존 향 계열을 `Fresh`, `Floral`, `Amber`, `Woody` 4개로 축소
- `Sweet` 라벨은 노트 키워드 기반으로 보정 분배
- 4분류 데이터셋을 `data/modified`에 저장
- 4분류 학습 결과를 `results/modified`에 저장

## 9. CLIP 텍스트 입력 변경

이미지를 보고 향 계열을 예측하는 목적에 맞게 `data_classifier/train_clip_classifier.py`의 CLIP 텍스트 입력을 `notes`만 사용하도록 수정했습니다.

현재 텍스트 입력 형식:

```python
fragrance notes: {notes}
```

`notes`가 비어 있으면 다음 문장을 사용합니다.

```python
fragrance notes unavailable
```

이 변경으로 모델이 향수명이나 브랜드명에 의존하지 않고, 이미지 특징과 노트 정보 기반의 임베딩을 사용합니다.

## 10. 향 계열 4분류 축소

최종 라벨은 다음 4개입니다.

- `Fresh`
- `Floral`
- `Amber`
- `Woody`

기본 매핑 방향:

- `Citrus` -> `Fresh`
- `Amber_Oriental` -> `Amber`
- `Spicy` -> `Amber`
- `Sweet` -> 노트 키워드 기반 보정 분배

`Sweet`는 단일 계열로 고정하지 않고, 노트에 포함된 키워드를 기준으로 `Amber`, `Floral`, `Fresh`, `Woody` 중 하나로 재분류합니다.

## 11. Sweet 라벨 보정

초기 방식은 단순 부분 문자열 검색을 사용해서 `cedarwood`, `ambergris` 같은 단어가 중복 계산될 수 있었습니다. 이를 보정하기 위해 다음 방식으로 수정했습니다.

- 정규식 기반 단어/구문 매칭 사용
- 같은 계열 안에서 겹치는 키워드 span은 중복 점수로 계산하지 않음
- `Sweet`는 기본적으로 `Amber` 성격이 강하다고 보고, 다른 계열 점수가 `Amber`보다 충분히 높을 때만 이동
- 이동 기준은 `NON_AMBER_SWITCH_MARGIN = 2`

보정 후 `Sweet` 재분배 결과:

| Split | Amber | Floral | Fresh | Woody |
|---|---:|---:|---:|---:|
| Train | 568 | 79 | 27 | 29 |
| Val | 72 | 10 | 3 | 3 |
| Test | 72 | 9 | 2 | 4 |

최종 `data/modified/train.csv` 라벨 분포:

| Label | Count |
|---|---:|
| Amber | 2734 |
| Floral | 8856 |
| Fresh | 3372 |
| Woody | 7595 |

## 12. 4분류 데이터 변환

4분류로 변환된 데이터는 `data/modified` 폴더에 저장됩니다.

- `data/modified/remap_to_four_families.py`
- `data/modified/train.csv`
- `data/modified/val.csv`
- `data/modified/test.csv`
- `data/modified/label_remap_summary.json`

변환 스크립트 실행:

```bash
python data/modified/remap_to_four_families.py
```

## 13. Colab 실행 방법

Colab에서 Google Drive를 마운트합니다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

프로젝트 폴더로 이동합니다.

```python
%cd /content/drive/MyDrive/perfume
```

필요 패키지를 설치합니다.

```python
!pip install -r data_classifier/requirements.txt
```

4분류 데이터셋을 사용하려면 반드시 `data/modified/*.csv`를 지정해야 합니다. 기본값은 `data/All/*.csv`입니다.

```bash
!python data_classifier/train_clip_classifier.py \
  --train-csv data/modified/train.csv \
  --val-csv data/modified/val.csv \
  --test-csv data/modified/test.csv \
  --clip-model openai/clip-vit-large-patch14 \
  --num-head-layers 3 \
  --hidden-dim 1024 \
  --lr 1e-3 \
  --epochs 20 \
  --patience 7 \
  --batch-size 16 \
  --rebuild-embedding-cache \
  --no-auto-copy-data-to-local
```

주의사항:

- `--rebuild-embedding-cache`는 유지합니다. 텍스트 입력과 라벨 구성이 바뀌었기 때문에 기존 임베딩 캐시를 재사용하면 안 됩니다.
- `--rebuild-embedding-cache \`처럼 옵션 뒤에 공백을 두고 줄바꿈합니다.
- `openai/clip-vit-large-patch14`에서 GPU 메모리가 부족하면 `--batch-size 8`로 낮춥니다.
- Colab에서 프로젝트가 `/content/drive/MyDrive/perfume` 아래에 있으면 기본적으로 `data`와 `perfume_images`를 `/content/perfume_local`로 자동 복사합니다. 위 실행 명령은 `--no-auto-copy-data-to-local`을 사용하므로 Drive 경로에서 직접 읽습니다.

## 14. 4분류 결과 파일

현재 4분류 실험 결과는 `results/modified`에 저장되어 있습니다.

- `results/modified/classification_report.txt`
- `results/modified/confusion_matrix.png`
- `results/modified/training_curves.png`

## 15. 4분류 실험 결과

`results/modified/classification_report.txt` 기준 성능:

| Metric | Value |
|---|---:|
| Test accuracy | 0.7610 |
| Test top-3 accuracy | 0.9638 |
| Macro F1-score | 0.7058 |
| Weighted F1-score | 0.7577 |

클래스별 성능:

| Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Amber | 0.5559 | 0.4781 | 0.5141 | 343 |
| Floral | 0.8211 | 0.8671 | 0.8434 | 1106 |
| Fresh | 0.7075 | 0.6722 | 0.6894 | 421 |
| Woody | 0.7732 | 0.7789 | 0.7761 | 950 |

해석:

- 전체 정확도는 `0.7610`입니다.
- Top-3 정확도는 `0.9638`로 높아, 정답 계열이 상위 후보 안에는 대부분 포함됩니다.
- `Floral`과 `Woody`는 상대적으로 안정적입니다.
- `Amber`는 precision/recall이 가장 낮아, `Sweet`, `Spicy`, `Amber_Oriental`이 합쳐진 계열 내부의 다양성이 성능 저하 원인일 수 있습니다.
