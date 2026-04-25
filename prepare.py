"""
두 가지 분류 태스크를 위한 데이터 준비

  - note  : 향 계열 분류 (6클래스, 동점 제거, 단어 경계 매칭)
  - brand : 브랜드 분류  (15개+ 샘플 브랜드 → 35개 브랜드)

실행: python prepare.py
출력:
  data/note/train.csv, val.csv, test.csv
  data/brand/train.csv, val.csv, test.csv
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV   = os.path.join("data", "raw", "all_cleaned.csv")
SOURCE_CSV = os.path.join("data", "raw", "final_perfume_data.csv")
NOTE_DIR  = os.path.join("data", "note")
BRAND_DIR = os.path.join("data", "brand")

# ── 향 계열 규칙 (6클래스) ──────────────────────────────────────────────────
# Fresh 클래스 제거 (45개 → 너무 적음)
#   lavender  → Floral 이전 (꽃에 해당)
#   lemongrass → Citrus 이전 (시트러스 향 계열)
LABEL_RULES = {
    "Citrus": [
        "bergamot", "lemon", "orange", "grapefruit",
        "lime", "mandarin", "yuzu", "neroli", "lemongrass",
    ],
    "Floral": [
        "rose", "jasmine", "lily", "magnolia", "violet",
        "iris", "peony", "tuberose", "geranium", "lavender",
    ],
    "Woody": [
        "sandalwood", "cedar", "vetiver", "oud",
        "patchouli", "guaiac", "cashmere wood",
    ],
    "Sweet": [
        "vanilla", "caramel", "chocolate", "honey",
        "tonka", "almond", "coconut",
    ],
    "Amber_Oriental": [
        "amber", "musk", "incense", "frankincense",
        "myrrh", "resin", "benzoin",
    ],
    "Spicy": [
        "pepper", "cardamom", "cinnamon", "clove",
        "nutmeg", "saffron", "ginger",
    ],
}


def notes_to_label(notes: str):
    """
    노트를 콤마로 분리 → 각 노트 단위로 카테고리 매칭 → 동점이면 None 반환

    개선점 (기존 main.py 대비):
      1. 전체 문자열 검색 대신 노트별 검색 → 'lemon' 이 'lemongrass' 에 오매칭되는 문제 해결
      2. 동점 시 None 반환 → 불명확한 라벨 제거
    """
    individual = [n.strip().lower() for n in notes.split(",") if n.strip()]
    score = {label: 0 for label in LABEL_RULES}

    for label, keywords in LABEL_RULES.items():
        for note in individual:
            if any(kw in note for kw in keywords):
                score[label] += 1

    max_val = max(score.values())
    if max_val == 0:
        return None  # 어떤 계열에도 해당 없음

    winners = [l for l, v in score.items() if v == max_val]
    return winners[0] if len(winners) == 1 else None  # 동점 제거


def add_descriptions(df: pd.DataFrame):
    """원본 CSV의 Description을 image_url 기준으로 정제 데이터에 병합"""
    source = pd.read_csv(SOURCE_CSV, encoding="latin1")
    source = source[["Image URL", "Description"]].copy()
    source = source.rename(columns={
        "Image URL": "image_url",
        "Description": "description",
    })
    source["image_url"] = source["image_url"].astype(str).str.strip()
    source["description"] = source["description"].fillna("").astype(str).str.strip()

    df = df.copy()
    df["image_url"] = df["image_url"].astype(str).str.strip()
    df = df.merge(source, on="image_url", how="left")
    df["description"] = df["description"].fillna("")
    return df


def split_and_save(df: pd.DataFrame, out_dir: str):
    """Stratified 80 / 10 / 10 분할 후 data/{task}/ 에 저장"""
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
    optional_cols = ["name", "brand", "description", "image_url"]
    save_cols = ["image_path", "label"] + [c for c in optional_cols if c in df.columns]

    train_df, temp = train_test_split(
        df[save_cols],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp["label"],
        random_state=42,
    )
    for split_name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
        data.to_csv(os.path.join(out_dir, f"{split_name}.csv"), index=False, encoding="utf-8-sig")

    print(f"  저장 완료 ({out_dir}) → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


def prepare_note(df: pd.DataFrame):
    print("=" * 50)
    print("Task A : Note 계열 분류")
    print("=" * 50)

    note_df = df.copy()
    note_df["label"] = note_df["notes"].apply(notes_to_label)
    note_df = note_df.dropna(subset=["label"]).reset_index(drop=True)

    removed = len(df) - len(note_df)
    print(f"동점·무매칭 제거: {removed}개  →  잔여: {len(note_df)}개")

    # 20개 미만 클래스 제거
    cnt = note_df["label"].value_counts()
    valid = cnt[cnt >= 20].index
    note_df = note_df[note_df["label"].isin(valid)].reset_index(drop=True)

    print("\n클래스 분포:")
    print(cnt[cnt >= 20].to_string())
    print(f"\n최종 클래스 수: {note_df['label'].nunique()}개")
    split_and_save(note_df, NOTE_DIR)


def prepare_brand(df: pd.DataFrame, min_samples: int = 15):
    print("\n" + "=" * 50)
    print("Task B : Brand 분류")
    print("=" * 50)

    brand_df = df[["image_path", "brand", "name", "description", "image_url"]].copy()
    brand_df = brand_df.rename(columns={"brand": "label"})
    brand_df["label"] = brand_df["label"].str.strip()
    brand_df = brand_df.dropna(subset=["label"])

    cnt = brand_df["label"].value_counts()
    valid = cnt[cnt >= min_samples].index
    brand_df = brand_df[brand_df["label"].isin(valid)].reset_index(drop=True)

    print(f"기준: {min_samples}개+ 샘플 브랜드")
    print(f"브랜드 수: {brand_df['label'].nunique()}개  |  총 샘플: {len(brand_df)}개")
    print("\n브랜드별 샘플 수 (상위 15):")
    print(cnt[cnt >= min_samples].head(15).to_string())
    split_and_save(brand_df, BRAND_DIR)


if __name__ == "__main__":
    df = pd.read_csv(RAW_CSV)
    df = add_descriptions(df)
    prepare_note(df)
    prepare_brand(df, min_samples=15)
