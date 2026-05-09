# CLIP 향수 계열 분류기

`data/All` 폴더의 CSV와 `perfume_images` 이미지를 사용해 향수 계열을 분류하는 CLIP 기반 학습 코드입니다.

## 기본 입력

- 학습 데이터: `data/All/train.csv`
- 검증 데이터: `data/All/val.csv`
- 테스트 데이터: `data/All/test.csv`
- 결과 저장 위치: `results/clip_classifier/<실행시간>`
- CLIP embedding cache 위치: `results/clip_embedding_cache`

## 실행 방법

```powershell
python data_classifier/train_clip_classifier.py
```

기본 설정은 CLIP 모델을 freeze하고, 이미지 embedding과 텍스트 embedding을 합친 뒤 작은 classifier만 학습합니다. 향수 계열은 이미지보다 `notes` 텍스트와 더 직접적으로 관련될 수 있으므로, 이미지 단독 모델보다 더 적합한 baseline입니다.

## Colab 실행 예시

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/perfume
```

```python
!pip install -r data_classifier/requirements.txt
```

```python
!python data_classifier/train_clip_classifier.py --epochs 10 --batch-size 128 --log-every 20
```

GPU 메모리가 부족하면 batch size를 낮추세요.

```python
!python data_classifier/train_clip_classifier.py --epochs 10 --batch-size 64 --log-every 20
```

Colab에서 프로젝트가 `/content/drive/MyDrive/perfume` 아래에 있으면 기본적으로 `data`와 `perfume_images`를 `/content/perfume_local`로 자동 복사한 뒤 학습합니다. Google Drive에서 이미지를 직접 읽는 병목을 줄이기 위한 설정입니다.

## 속도 개선 방식

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

## 주요 옵션

```powershell
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

## 성능 평가 척도

학습과 테스트 결과에는 다음 지표가 포함됩니다.

- Accuracy: 가장 높은 확률의 예측 1개가 정답이면 성공
- Top-3 Accuracy: 확률 상위 3개 예측 안에 정답이 있으면 성공
- Precision, Recall, F1-score: 클래스별 분류 성능
- Confusion Matrix: 실제 클래스와 예측 클래스의 혼동 정도

Top-3 Accuracy는 향수 계열처럼 계열 경계가 겹치는 분류 문제에서 유용합니다. 예를 들어 실제 정답이 `Woody`이고 모델의 상위 3개 예측에 `Woody`가 포함되면 Top-3에서는 맞은 것으로 계산합니다.

## 결과 파일

학습 결과는 `results/clip_classifier/<실행시간>`에 저장됩니다.

- `best_clip_classifier.pt`: 가장 좋은 validation accuracy의 모델
- `summary.json`: 최고 epoch, validation accuracy, test accuracy, test top-3 accuracy 요약
- `training_history.csv`: epoch별 loss, accuracy, top-3 accuracy
- `classification_report.txt`: test accuracy, test top-3 accuracy, 클래스별 precision/recall/f1-score
- `test_predictions.csv`: 테스트 샘플별 예측 결과, confidence, top-3 예측
- `training_curves.png`: loss, accuracy, top-3 accuracy 학습 곡선
- `confusion_matrix.png`: confusion matrix
- `label_map.json`: 클래스 이름과 인덱스 매핑
