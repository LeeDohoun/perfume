# Perfume Note Classification

향수 이미지를 기반으로 향료 노트(Note)를 분류하는 딥러닝 프로젝트

---

## 1. 프로젝트 주제

향수병 이미지를 입력으로 받아 향료 노트를 분류하는 모델을 개발한다.

- Floral
- Woody
- Citrus
- Amber
- Sweet
- Spicy

---

## 2. 데이터셋

- Kaggle LuckyScent 데이터 사용
- 약 2000개 이미지 데이터

### 추가 계획
- 노트 데이터 확장
- 브랜드 데이터 추가 확보
- 데이터 불균형 개선

---

## 3. 모델

- EfficientNet-B0 (baseline)
- MobileNetV3 (경량 모델)
- CLIP (멀티모달 확장)

---

## 4. 학습 전략

- Transfer Learning
- Fine-tuning
- Data Augmentation
- Early Stopping

---

## 5. 평가 지표

- Accuracy
- Macro F1-score
- Confusion Matrix
- Top-3 Accuracy

---

## 6. 목표

- Random baseline 대비 성능 향상
- 이미지 기반 노트 분류 가능성 검증
- 멀티모달 적용 시 성능 개선 확인
