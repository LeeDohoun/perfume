# 향수 이미지 기반 Note 분류 프로젝트

## 프로젝트 개요
향수병 이미지 + 텍스트(brand/name)를 입력받아 7개 Note 클래스로 분류하는 딥러닝 프로젝트.
- 7 클래스: Floral, Woody, Amber_Oriental, Citrus, Sweet, Spicy, Fresh
- 데이터: train 22,557 / train_aug 41,061 / val 2,820 / test 2,820

## 실행 환경
- Python 가상환경: `.venv` (메인), `.venv-mps` (MPS 실험용)
- 메인 실행: `perfume_classifier/` 디렉토리에서 `../.venv/bin/python main.py`
- 스크립트 실행: 프로젝트 루트에서 `.venv/bin/python scripts/`

## 핵심 명령어 (run.sh 참고)
```bash
# EfficientNet + 텍스트 (메인 모델) 평가
cd perfume_classifier && ../.venv/bin/python main.py --mode eval

# EfficientNet 이미지만 평가
cd perfume_classifier && ../.venv/bin/python main.py --mode eval --no_text --ckpt ../checkpoints_image_only/stage2_best.pth

# EfficientNet + 텍스트 학습+평가 (처음부터)
cd perfume_classifier && ../.venv/bin/python main.py --mode train_eval

# EfficientNet 이미지만 학습+평가 (처음부터)
cd perfume_classifier && ../.venv/bin/python main.py --mode train_eval --no_text

# CLIP + 텍스트 학습+평가
.venv/bin/python scripts/train_clip_text.py --device auto

# CLIP 이미지만 학습+평가 (기존 실험 재실행)
.venv/bin/python scripts/train_clip_finetune.py --device auto
```

## 실험 현황 (2026-05-18 기준)

| 모델 | 상태 | Accuracy | Macro F1 | Top-3 |
|---|---|---|---|---|
| EfficientNet-B0 + 텍스트 | ✅ 완료 | 0.5191 | 0.3226 | 0.8766 |
| EfficientNet-B0 이미지만 | ✅ 완료 | 0.4592 | 0.2243 | 0.8223 |
| CLIP 이미지만 (linear probe) | ✅ 완료 | 0.4723 | 0.2563 | 0.8106 |
| CLIP 이미지만 (fine-tune CPU) | ✅ 완료 | 0.4489 | 0.2815 | 0.8177 |
| CLIP 이미지만 (fine-tune MPS) | ✅ 완료 | 0.4404 | 0.2894 | 0.8160 |
| CLIP + 텍스트 | ✅ 완료 | 0.4851 | 0.3384 | 0.8532 |

## 체크포인트 경로
- EfficientNet + 텍스트: `checkpoints/stage1_best.pth`, `checkpoints/stage2_best.pth`
- EfficientNet 이미지만: `checkpoints_image_only/stage1_best.pth`, `checkpoints_image_only/stage2_best.pth`
- CLIP 이미지만: `checkpoints/clip_finetune/`, `checkpoints/clip_finetune_mps/`
- CLIP + 텍스트: `checkpoints/clip_text/` (미실행)

## 결과 경로
- `results/test_metrics.txt` — EfficientNet + 텍스트
- `results/efficientnet_image_only/` — EfficientNet 이미지만 (평가 후 생성)
- `results/clip_linear_probe/` — CLIP linear probe
- `results/clip_finetune/` — CLIP fine-tune CPU
- `results/clip_finetune_mps/` — CLIP fine-tune MPS
- `results/clip_text/` — CLIP + 텍스트 (미실행)

## 남은 작업
1. ~~EfficientNet 이미지만 — 평가 실행~~ ✅
2. ~~CLIP + 텍스트 — 학습+평가 실행~~ ✅
3. 전체 6개 모델 성능 비교표 작성
4. 노션 업데이트

## 프로젝트 구조
```
perfume/
├── perfume_classifier/     # 메인 모듈 (EfficientNet)
│   ├── main.py             # 진입점 (--mode, --no_text, --backbone 등)
│   ├── config.py           # 하이퍼파라미터 설정
│   ├── model.py            # PerfumeClassifier (EfficientNet + TextEncoder)
│   ├── dataset.py          # PerfumeDataset, TextVocab, Transforms
│   ├── train.py            # 2단계 학습 (Stage1: Freeze, Stage2: Gradual Unfreeze)
│   ├── evaluate.py         # 평가 (Accuracy, Macro F1, Top-3, Confusion Matrix)
│   └── augment_offline.py  # 소수 클래스 오프라인 증강
├── scripts/                # 실험용 스크립트
│   ├── train_clip_finetune.py   # CLIP 이미지만 2단계 fine-tune
│   ├── train_clip_text.py       # CLIP + 텍스트 2단계 fine-tune (신규)
│   └── eval_clip_note.py        # CLIP zero-shot / linear probe
├── data/All/               # train.csv, val.csv, test.csv, train_aug.csv
├── checkpoints/            # EfficientNet+텍스트 체크포인트
├── checkpoints_image_only/ # EfficientNet 이미지만 체크포인트
├── perfume_images/         # 향수 이미지 (~28,000장)
└── results/                # 평가 결과
```
