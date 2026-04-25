# Perfume Image Classification

향수 병 이미지를 기반으로 **향 계열(Note)** 또는 **브랜드(Brand)** 를 분류하는 딥러닝 프로젝트입니다.

---

## 프로젝트 목표

| 태스크 | 입력 | 출력 |
|--------|------|------|
| Note 분류 | 향수병 이미지 | 향 계열 6종 (Floral, Woody, Citrus 등) |
| Brand 분류 | 향수병 이미지 | 브랜드 35종 (BYREDO, Le Labo 등) |

## 목표 정확도

| 모델 | 태스크 | 목표 Test Accuracy |
|------|--------|--------------------|
| EfficientNet-B0 | Note 분류 | 35 ~ 50% |
| EfficientNet-B0 | Brand 분류 | 40 ~ 65% |
| CLIP ViT-B/32 | Note 분류 | 45 ~ 60% |
| TF-IDF + LogisticRegression | Note 분류 | 45 ~ 60% |

---

## 현재 진행 상황

```
[완료] 1단계: 데이터 수집 및 이미지 다운로드   main.py
[완료] 2단계: 전처리 및 분할                  prepare.py
[완료] 3단계: 모델 선정                       model_recommendation.md 참고
[진행] 4단계: EfficientNet-B0 학습            train.py
[진행] 5단계: CLIP Note 평가                  clip_note_eval.py
[완료] 6단계: Note 정확도 개선 모델           train_note_text.py
```

---

## 프로젝트 구조

```
perfume/
├── data/
│   ├── raw/
│   │   ├── final_perfume_data.csv   # 원본 데이터 (2,191개)
│   │   └── all_cleaned.csv          # 이미지 다운로드 완료 데이터 (2,067개)
│   ├── note/
│   │   ├── train.csv                # 1,006개
│   │   ├── val.csv                  # 126개
│   │   └── test.csv                 # 126개
│   └── brand/
│       ├── train.csv                # 652개
│       ├── val.csv                  # 81개
│       └── test.csv                 # 82개
├── perfume_images/                  # 다운로드된 향수병 이미지 (2,067장)
├── main.py                          # 1단계: 데이터 수집 및 이미지 다운로드
├── prepare.py                       # 2단계: 전처리 및 태스크별 분할
├── train.py                         # EfficientNet-B0 학습 및 평가
├── clip_note_eval.py                # CLIP zero-shot / linear-probe Note 평가
├── train_note_text.py               # Note 개선용 메타데이터 텍스트 모델
├── model_recommendation.md          # 모델 추천 및 비교
├── requirements.txt
└── README.md
```

---

## 데이터 파이프라인

### 1단계 - 데이터 수집 (`main.py`)

- 출처: LuckyScent 향수 데이터 (`final_perfume_data.csv`)
- 원본 2,191개 중 이미지 다운로드 성공 **2,067개** 사용
- 검증: 100×100px 미만 이미지 제거, RGB 변환

### 2단계 - 전처리 (`prepare.py`)

**Note 분류 전처리**

| 처리 | 내용 |
|------|------|
| 라벨 방식 | Notes 문자열을 콤마로 분리 → 노트별 키워드 매칭 |
| 동점 제거 | 2개 이상 계열이 동점이면 제거 (기존 44% 동점 문제 해결) |
| Fresh 제거 | 45개로 샘플 부족 → lavender는 Floral, lemongrass는 Citrus로 이전 |
| 최소 기준 | 클래스당 20개 미만 제거 |
| 결과 | 6클래스, 총 1,258개 |

**Note 클래스 분포 (train 기준)**

| 클래스 | 샘플 수 |
|--------|---------|
| Floral | 254 |
| Woody | 253 |
| Amber_Oriental | 166 |
| Citrus | 153 |
| Sweet | 98 |
| Spicy | 82 |

**Brand 분류 전처리**

| 처리 | 내용 |
|------|------|
| 필터 기준 | 샘플 15개 이상 브랜드만 사용 |
| 결과 | 35개 브랜드, 총 815개 |

**데이터 분할** (공통): Stratified 80 / 10 / 10

---

## 설치 및 실행

```bash
# 환경 설치
pip install -r requirements.txt

# 전처리 실행 (data/ 폴더 내 CSV 생성)
python prepare.py

# EfficientNet-B0: Note 분류
python train.py --task note

# EfficientNet-B0: Brand 분류
python train.py --task brand

# CLIP: Note linear-probe 평가 (목표 45~60%)
python clip_note_eval.py --mode linear_probe --split test

# CLIP: Note zero-shot 기준선 확인
python clip_note_eval.py --mode zero_shot --split test

# Note 정확도 개선: name + brand + description 텍스트 모델
python train_note_text.py

# 저장된 모델 재평가
python eval_checkpoint.py --task note --device cpu
python eval_checkpoint.py --task brand --device cpu
python eval_note_text.py
```

> `main.py` (이미지 다운로드)는 이미 완료된 단계입니다.  
> `data/raw/all_cleaned.csv` 와 `perfume_images/` 가 존재하면 `prepare.py` 부터 실행하면 됩니다.

---

## 모델 추천

`model_recommendation.md` 참고

현재 실험 결과와 Note 태스크의 문제점은 `RESULTS.md` 참고

| 태스크 | 추천 모델 | 이유 |
|--------|-----------|------|
| Note 분류 | TF-IDF + LogisticRegression, CLIP, EfficientNet-B0 | 이미지 단독은 한계가 있어 메타데이터 텍스트 모델이 현재 목표 도달 |
| Brand 분류 | EfficientNet-B0 | 브랜드 병 디자인 학습 목표 40~65% |

---

## 요구사항

```
torch >= 2.0.0
torchvision >= 0.15.0
Pillow >= 9.0.0
pandas >= 1.5.0
scikit-learn >= 1.2.0
requests >= 2.28.0
numpy < 2.0
open_clip_torch >= 2.24.0
joblib >= 1.3.0
```
