#!/bin/bash
# 향수 분류 프로젝트 실행 스크립트
# 사용법: bash run.sh <명령>

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"

show_help() {
    echo ""
    echo "사용법: bash run.sh <명령>"
    echo ""
    echo "  eval_main       EfficientNet+텍스트 평가 (기존 체크포인트)"
    echo "  eval_imgonly    EfficientNet 이미지만 평가"
    echo "  train_main      EfficientNet+텍스트 학습+평가 (처음부터)"
    echo "  train_imgonly   EfficientNet 이미지만 학습+평가 (처음부터)"
    echo "  train_clip_text CLIP+텍스트 학습+평가"
    echo "  train_clip_img  CLIP 이미지만 학습+평가"
    echo ""
}

case "$1" in
    eval_main)
        cd "$ROOT/perfume_classifier"
        $PYTHON main.py --mode eval
        ;;
    eval_imgonly)
        cd "$ROOT/perfume_classifier"
        $PYTHON main.py --mode eval --no_text --ckpt ../checkpoints_image_only/stage2_best.pth
        ;;
    train_main)
        cd "$ROOT/perfume_classifier"
        $PYTHON main.py --mode train_eval
        ;;
    train_imgonly)
        cd "$ROOT/perfume_classifier"
        $PYTHON main.py --mode train_eval --no_text
        ;;
    train_clip_text)
        cd "$ROOT"
        $PYTHON scripts/train_clip_text.py --device auto
        ;;
    train_clip_img)
        cd "$ROOT"
        $PYTHON scripts/train_clip_finetune.py --device auto
        ;;
    *)
        show_help
        ;;
esac
