# Model Run Inventory

정리일: 2026-05-13

이 문서는 현재 프로젝트에서 실제로 돌린 모델과, 지금 기준으로 사용할 결과/사용하지 않을 실험을 구분한다.

## 현재 최종 비교에 사용하는 모델

| 모델 | 데이터 | 입력 | 결과 위치 | Accuracy | Macro F1 | Top-3 Accuracy | 상태 |
|---|---|---|---|---:|---:|---:|---|
| EfficientNet-B0 + text encoder | `data/All` | image + `brand` + `name` | `results/efficientnet_b0` | 0.5191 | 0.3226 | 0.8766 | 최종 비교용 |
| CLIP two-stage fine-tune (MPS) | `data/All` | image only | `results/clip_finetune_mps` | 0.4404 | 0.2894 | 0.8160 | 최종 비교용 |
| CLIP two-stage fine-tune (CPU 3e) | `data/All` | image only | `results/clip_finetune` | 0.4489 | 0.2815 | 0.8177 | 참고 |
| CLIP linear probe | `data/All` | image embedding | `results/clip_linear_probe` | 0.4723 | 0.2563 | 0.8106 | baseline |

최종 발표/보고용 이미지는 아래 파일을 사용한다.

- `results/efficientnet_b0/evaluation_summary.png`
- `results/clip_finetune_mps/evaluation_summary.png`
- `results/model_metric_comparison.png`

## 현재 실행 방식

### EfficientNet-B0

현재 7클래스 Note 분류의 기본 모델이다. `notes`는 입력에서 제외하고, 이미지와 `brand`, `name`만 사용한다.

```bash
.venv/bin/python perfume_classifier/main.py --mode eval --split test
```

핵심 파일:

- `perfume_classifier/main.py`
- `perfume_classifier/model.py`
- `perfume_classifier/dataset.py`
- `perfume_classifier/evaluate.py`
- `checkpoints/stage2_best.pth`

### CLIP two-stage fine-tune

EfficientNet-B0와 맞춰 보기 위해 CLIP도 2단계 방식으로 돌렸다.

Stage 1:

- CLIP visual encoder freeze
- image embedding 추출
- classifier head만 학습

Stage 2:

- Stage 1 best head 로드
- CLIP visual transformer 마지막 1개 block unfreeze
- 낮은 learning rate로 fine-tuning
- Macro F1 기준 best checkpoint 저장

```bash
.venv/bin/python -u scripts/train_clip_finetune.py \
  --stage1-epochs 5 \
  --stage2-epochs 3 \
  --train-split train_aug
```

MPS 환경에서는 별도 Python 3.11 환경을 만들고, Stage 1을 8 epoch, Stage 2를 최대 12 epoch 설정으로 실행했다. Stage 2는 val Macro F1이 2 epoch 이후 개선되지 않고 train 성능만 계속 올라 과적합 신호가 보여 중단했으며, best checkpoint는 Stage 2 epoch 2이다.

```bash
.venv-mps/bin/python scripts/train_clip_finetune.py \
  --device mps \
  --stage1-epochs 8 \
  --stage2-epochs 12 \
  --stage2-batch-size 32 \
  --unfreeze-last-blocks 1 \
  --output-dir results/clip_finetune_mps \
  --checkpoint-dir checkpoints/clip_finetune_mps
```

핵심 파일:

- `scripts/train_clip_finetune.py`
- `checkpoints/clip_finetune_mps/clip_stage2_best.pt`
- `results/clip_finetune_mps/evaluation_summary.png`
- `results/clip_finetune_mps/classification_report.txt`

참고: Codex 기본 샌드박스에서는 MPS 접근이 막혀 CPU처럼 보였지만, 샌드박스 밖 실행에서는 MPS가 정상 동작했다.

### CLIP linear probe baseline

초기 비교용 CLIP baseline이다. `data/All/train_aug.csv`에서 CLIP image embedding을 추출하고, Ridge classifier로 평가한다.

```bash
.venv/bin/python -u scripts/eval_clip_note.py \
  --mode linear_probe \
  --split test \
  --data_dir data/All \
  --train_split train_aug \
  --batch_size 64 \
  --device cpu
```

핵심 파일:

- `scripts/eval_clip_note.py`
- `results/clip_linear_probe/evaluation_summary.png`
- `results/clip_linear_probe/classification_report.txt`

참고: CLIP embedding cache는 `results/clip_linear_probe/features/`에 생성되지만, 재생성 가능한 중간 산출물이므로 현재 정리 대상에서 제외하고 `.gitignore` 처리한다.

## 참고용 결과

### Brand EfficientNet-B0

브랜드 분류 결과는 남아 있지만, 현재 Note 분류 최종 비교에는 포함하지 않는다.

| 항목 | 값 |
|---|---:|
| 데이터 | `data/brand` |
| 브랜드 수 | 35 |
| train / val / test | 652 / 81 / 82 |
| Stage 2 Best Val Accuracy | 0.9753 |
| 기록된 Test Accuracy | 0.9512 |

해석 조건: 전체 242개 브랜드 일반화 성능이 아니라, 35개 frequent brand closed-set 성능이다.

관련 파일:

- `checkpoints/brand/best_model.pth`
- `docs/brand_accuracy_validation.md`
- `scripts/analyze_brand_accuracy.py`

### SeokHyeon CLIP fusion

석현 브랜치의 CLIP 결과는 현재 CLIP image-only baseline보다 높다.

| 모델 | Accuracy | Macro F1 | Top-3 Accuracy |
|---|---:|---:|---:|
| SeokHyeon CLIP fusion | 0.6241 | 0.4663 | 0.9138 |

단, 해당 방식은 `name + brand + notes` 텍스트 embedding과 이미지 embedding을 함께 사용한다. `notes`는 Note 라벨을 직접 암시하므로 최종 공정 비교에는 넣지 않고, leakage 가능성이 있는 참고/상한선 baseline으로만 본다.

## 현재 안 쓰는 레거시 실험

아래 항목은 현재 최종 비교 방식에서는 쓰지 않는다. 재현 필요가 없다면 삭제 후보지만, 지금은 추적 가능한 참고 코드로 남겨둔다.

| 파일/폴더 | 이유 | 권장 상태 |
|---|---|---|
| `scripts/train_efficientnet.py` | 구 `data/note`, `data/brand`용 EfficientNet 이미지-only 실험 | deprecated |
| `scripts/eval_efficientnet.py` | 구 체크포인트 평가용 스크립트 | deprecated |
| `scripts/train_note_text.py` | `name + brand + description` 텍스트 baseline | reference only |
| `scripts/eval_note_text.py` | 텍스트 baseline 평가 | reference only |
| `data/note` | 현재 7클래스 `data/All` 이전의 Note split | deprecated |
| `checkpoints/note` | 구 Note 이미지-only 체크포인트 | deprecated |
| `checkpoints/note_text` | 텍스트 baseline 체크포인트 | reference only |

## 정리 원칙

최종 비교는 `data/All` 기준의 아래 모델을 중심으로 사용한다.

1. EfficientNet-B0 + image/brand/name
2. CLIP two-stage fine-tune
3. CLIP image embedding linear probe baseline

`notes`를 입력으로 쓰는 실험은 성능이 높더라도 leakage 가능성이 있으므로 최종 성능 표에는 넣지 않는다.
