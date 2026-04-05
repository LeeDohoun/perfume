# Perfume Image Classification

향수 병 이미지를 기반으로 **향 계열(Note)** 또는 **브랜드(Brand)** 를 분류하는 딥러닝 프로젝트입니다.

---

## 프로젝트 목표

| 태스크 | 입력 | 출력 |
|--------|------|------|
| Note 분류 | 향수병 이미지 | 향 계열 6종 (Floral, Woody, Citrus 등) |
| Brand 분류 | 향수병 이미지 | 브랜드 35종 (BYREDO, Le Labo 등) |

---

## 현재 진행 상황

```
[완료] 1단계: 데이터 수집 및 이미지 다운로드   main.py
[완료] 2단계: 전처리 및 분할                  prepare.py
[예정] 3단계: 모델 선정                       model_recommendation.md 참고
[예정] 4단계: 모델 학습 및 평가
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
```

> `main.py` (이미지 다운로드)는 이미 완료된 단계입니다.  
> `data/raw/all_cleaned.csv` 와 `perfume_images/` 가 존재하면 `prepare.py` 부터 실행하면 됩니다.

---

## 모델 추천

`model_recommendation.md` 참고

| 태스크 | 추천 모델 | 이유 |
|--------|-----------|------|
| Note 분류 | CLIP (ViT-B/32) | 텍스트 프롬프트로 시각-향 상관 낮은 문제 보완 |
| Brand 분류 | EfficientNet-B0 | 소규모 데이터에서 과적합 방지, 높은 성능 |

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
```
