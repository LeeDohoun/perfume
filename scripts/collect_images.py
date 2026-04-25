import os
import re
import time
import requests
import pandas as pd

from io import BytesIO
from PIL import Image

# ----------------------------
# 1. 설정
# ----------------------------
CSV_PATH = os.path.join("data", "raw", "final_perfume_data.csv")
CLEANED_CSV = os.path.join("data", "raw", "all_cleaned.csv")
SAVE_DIR = "perfume_images"
os.makedirs(os.path.dirname(CLEANED_CSV), exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# 2. CSV 읽기
# ----------------------------
df = pd.read_csv(CSV_PATH, encoding="latin1")

# 필요한 컬럼만 사용
df = df[["Name", "Brand", "Notes", "Image URL"]].copy()

# 결측치 제거
df = df.dropna(subset=["Name", "Notes", "Image URL"]).reset_index(drop=True)

# 문자열 정리
for col in ["Name", "Brand", "Notes", "Image URL"]:
    df[col] = df[col].astype(str).str.strip()

print("원본 데이터 수:", len(df))
print(df.head())
# ----------------------------
# 3. Notes -> 대표 향 계열 라벨 변환
# ----------------------------
LABEL_RULES = {
    "Citrus": [
        "bergamot", "lemon", "orange", "grapefruit", "lime", "mandarin", "yuzu", "neroli"
    ],
    "Floral": [
        "rose", "jasmine", "lily", "magnolia", "violet", "iris", "peony", "tuberose", "geranium"
    ],
    "Woody": [
        "sandalwood", "cedar", "vetiver", "oud", "patchouli", "guaiac", "cashmere wood"
    ],
    "Fresh": [
        "mint", "aquatic", "marine", "green", "tea", "lavender", "lemongrass", "basil"
    ],
    "Sweet": [
        "vanilla", "caramel", "chocolate", "honey", "tonka", "almond", "coconut"
    ],
    "Amber_Oriental": [
        "amber", "musk", "incense", "frankincense", "myrrh", "resin", "benzoin"
    ],
    "Spicy": [
        "pepper", "cardamom", "cinnamon", "clove", "nutmeg", "saffron", "ginger"
    ]
}

def notes_to_label(notes: str):
    text = notes.lower()

    score = {label: 0 for label in LABEL_RULES}
    for label, keywords in LABEL_RULES.items():
        for kw in keywords:
            if kw in text:
                score[label] += 1

    best_label = max(score, key=score.get)
    if score[best_label] == 0:
        return None   # 어떤 계열에도 안 걸리면 제거
    return best_label

df["label"] = df["Notes"].apply(notes_to_label)

# 라벨 없는 샘플 제거
df = df.dropna(subset=["label"]).reset_index(drop=True)

print("\n라벨링 후 데이터 수:", len(df))
print(df["label"].value_counts())
# ----------------------------
# 4. 파일명 안전하게 만들기
# ----------------------------
def safe_filename(text):
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:80]

# ----------------------------
# 5. 이미지 다운로드 + 검증
# ----------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

saved_rows = []

for idx, row in df.iterrows():
    url = row["Image URL"]
    name = row["Name"]
    brand = row["Brand"]
    notes = row["Notes"]
    label = row["label"]

    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()

        # 이미지 열기
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # 너무 작은 이미지 제거
        if img.width < 100 or img.height < 100:
            continue

        filename = f"{idx:05d}_{safe_filename(name)}.jpg"
        image_path = os.path.join(SAVE_DIR, filename)

        # 저장
        img.save(image_path, format="JPEG", quality=95)

        saved_rows.append({
            "image_path": image_path,
            "label": label,
            "name": name,
            "brand": brand,
            "notes": notes,
            "image_url": url
        })

        if idx % 100 == 0:
            print(f"{idx}개 처리 중...")

        time.sleep(0.1)  # 서버 부담 완화

    except Exception as e:
        print(f"[실패] {name}: {e}")
        continue

clean_df = pd.DataFrame(saved_rows)
print("\n다운로드 성공 수:", len(clean_df))
print(clean_df.head())
# ----------------------------
# 6. 샘플 수가 너무 적은 클래스 제거
# ----------------------------
label_counts = clean_df["label"].value_counts()
valid_labels = label_counts[label_counts >= 20].index   # 최소 20개 이상만 사용

clean_df = clean_df[clean_df["label"].isin(valid_labels)].reset_index(drop=True)

print("\n클래스 정리 후 데이터 수:", len(clean_df))
print(clean_df["label"].value_counts())
# ----------------------------
# 7. 정제 CSV 저장
# ----------------------------
clean_df.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")
print(f"\n정제 CSV 저장: {CLEANED_CSV}")
print("다음 단계: python scripts/prepare_data.py")
