"""
데이터 증강 시각화 데모
- 원본 향수 이미지 1장에 다양한 증강을 적용한 결과를 보여줍니다.
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# 재현성
random.seed(None)
np.random.seed(None)

# ── 샘플 이미지 선택 ──
IMG_DIR = "perfume_images"
sample_images = [
    "00013_LHomme_A_La_Rose_Eau_de_Parfum.jpg",  # 장미 향수
    "00042_Basilico_Fellini_Eau_de_Parfum.jpg",   # 특이한 병
    "00287_Eau_Capitale_Eau_de_Parfum.jpg",        # diptyque
]

# 첫 번째 존재하는 이미지 사용
img_path = None
for name in sample_images:
    path = os.path.join(IMG_DIR, name)
    if os.path.exists(path):
        img_path = path
        break

if img_path is None:
    # 아무 이미지나 사용
    files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    img_path = os.path.join(IMG_DIR, files[0])

print(f"샘플 이미지: {img_path}")
original = Image.open(img_path).convert("RGB")

# ── 증강 기법 정의 ──
augmentations = {
    "1. Original": transforms.Compose([
        transforms.Resize((224, 224)),
    ]),
    "2. HorizontalFlip": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),  # 확실히 뒤집기
    ]),
    "3. Rotation(15)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
    ]),
    "4. ColorJitter": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    ]),
    "5. RandomCrop": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
    ]),
    "6. Affine": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ]),
    "7. Perspective": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomPerspective(distortion_scale=0.3, p=1.0),
    ]),
    "8. Combined\n(Train용)": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ]),
}

# ── 결과 이미지 그리드 생성 ──
TILE_SIZE = 224
PADDING = 10
LABEL_H = 30
COLS = 4
ROWS = 2
GRID_W = COLS * TILE_SIZE + (COLS + 1) * PADDING
GRID_H = ROWS * (TILE_SIZE + LABEL_H) + (ROWS + 1) * PADDING

grid = Image.new("RGB", (GRID_W, GRID_H), (255, 255, 255))
draw = ImageDraw.Draw(grid)

for i, (name, aug) in enumerate(augmentations.items()):
    row = i // COLS
    col = i % COLS
    x = PADDING + col * (TILE_SIZE + PADDING)
    y = PADDING + row * (TILE_SIZE + LABEL_H + PADDING)

    # 증강 적용
    augmented = aug(original)
    if not isinstance(augmented, Image.Image):
        augmented = transforms.ToPILImage()(augmented)
    augmented = augmented.resize((TILE_SIZE, TILE_SIZE))

    grid.paste(augmented, (x, y + LABEL_H))

    # 라벨
    draw.text((x + 5, y + 5), name, fill=(0, 0, 0))

output_path = "augmentation_demo.png"
grid.save(output_path)
print(f"저장 완료: {output_path}")
print(f"크기: {GRID_W}x{GRID_H}")
