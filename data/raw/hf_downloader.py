"""
hf_downloader.py
----------------
HuggingFace doevent/perfume 데이터셋 통합 스크립트.

캐시된 파일 사용:
  - images.zip  (835 MB, 이미 다운로드됨)
  - perfumes.csv (파이프 구분자, 26,319개)

사용법
------
  python data/raw/hf_downloader.py               # 압축 해제 + CSV 변환 + 병합
  python data/raw/hf_downloader.py --no_merge    # 병합 없이 hf_perfume.csv만 생성
  python data/raw/hf_downloader.py --skip_extract # 이미 압축 해제된 경우 건너뜀
"""

import argparse
import ast
import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
RAW_DIR    = Path(__file__).resolve().parent
IMAGE_DIR  = BASE_DIR / "perfume_images"
OUTPUT_CSV = RAW_DIR / "hf_perfume.csv"
ALL_CSV    = RAW_DIR / "all_cleaned.csv"

HF_CACHE = Path(os.environ.get("USERPROFILE", "~")).expanduser() / (
    ".cache/huggingface/hub/datasets--doevent--perfume/snapshots"
)

# ──────────────────────────────────────────────
# 캐시 경로 탐색
# ──────────────────────────────────────────────

def find_cached_file(filename: str) -> Path:
    """HuggingFace 캐시에서 파일을 찾습니다."""
    for snapshot_dir in HF_CACHE.iterdir() if HF_CACHE.exists() else []:
        candidate = snapshot_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{filename} 을(를) HuggingFace 캐시에서 찾을 수 없습니다.\n"
        "먼저 다음을 실행하세요:\n"
        "  pip install huggingface_hub\n"
        "  python -c \"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download('doevent/perfume', repo_type='dataset', filename='{filename}')\""
    )


# ──────────────────────────────────────────────
# 노트 기반 라벨 분류
# ──────────────────────────────────────────────

LABEL_KEYWORDS = {
    "Floral": [
        "rose", "jasmine", "lily", "violet", "iris", "peony", "magnolia",
        "tuberose", "gardenia", "ylang", "geranium", "neroli", "orange blossom",
        "mimosa", "narcissus", "cherry blossom", "heliotrope", "freesia",
        "caramel", "chocolate", "honey", "praline", "sugar", "candy",
        "marshmallow", "coconut", "almond", "hazelnut", "butterscotch",
        "toffee", "cream", "milk", "gourmand",
    ],
    "Woody": [
        "cedar", "sandalwood", "vetiver", "oakmoss", "patchouli", "agarwood",
        "birch", "pine", "fir", "guaiac", "teak", "bamboo", "driftwood",
        "woody", "wood",
        "pepper", "cinnamon", "cardamom", "clove", "nutmeg", "cumin",
        "ginger", "saffron", "chili", "paprika", "turmeric", "caraway",
        "coriander", "anise", "star anise",
    ],
    "Amber": [
        "amber", "vanilla", "benzoin", "labdanum", "tonka", "incense",
        "frankincense", "myrrh", "opoponax", "resin", "balsam", "copal",
        "castoreum", "musk", "oud",
    ],
    "Fresh": [
        "aquatic", "marine", "oceanic", "cucumber", "watermelon", "melon",
        "ozonic", "mint", "spearmint", "grass", "hay", "green", "herbal",
        "fig leaf", "tomato leaf", "basil", "lavender", "fougere",
        "bergamot", "lemon", "orange", "grapefruit", "lime", "mandarin",
        "yuzu", "tangerine", "kumquat", "pomelo", "petitgrain", "citrus",
    ],
}

# family → label (1순위)
FAMILY_PRIMARY = {
    "FLORAL":           "Floral",
    "WOODY":            "Woody",
    "AMBERY":           "Amber",
    "ORIENTAL":         "Amber",
    "CITRUS":           "Fresh",
    "AROMATIC FOUGERE": "Fresh",
    "CHYPRE":           "Woody",
    "LEATHER":          "Woody",
}

# subfamily → label (2순위)
SUBFAMILY_RESCUE = {
    "GOURMAND": "Floral",  # Sweet → Floral
    "SPICY":    "Woody",   # Spicy → Woody
}

# 기존 7클래스 → 4클래스 리매핑 (luckyscent 기존 데이터용)
LABEL_MAP_4 = {
    "Floral": "Floral", "Sweet": "Floral",
    "Woody": "Woody",   "Spicy": "Woody",
    "Fresh": "Fresh",   "Citrus": "Fresh",
    "Amber_Oriental": "Amber", "Amber": "Amber",
}


def classify_label(notes: str, family: str = "", subfamily: str = "") -> str:
    family_up    = family.upper()
    subfamily_up = subfamily.upper()

    # 1순위: GOURMAND/SPICY subfamily는 family보다 우선 (시각적으로도 구분됨)
    for key, label in SUBFAMILY_RESCUE.items():
        if key in subfamily_up:
            return label

    # 2순위: family 직접 매핑 (CITRUS family면 반드시 Citrus로 — FRUITY 서브패밀리에도 불구)
    for key, label in FAMILY_PRIMARY.items():
        if key in family_up:
            return label

    # 3순위: 재료 키워드 점수
    notes_lower = notes.lower()
    note_list   = [n.strip() for n in notes_lower.split(",")]
    scores = {label: 0 for label in LABEL_KEYWORDS}
    for label, keywords in LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw in notes_lower:
                scores[label] += 1
                for i, note in enumerate(note_list[:3]):
                    if kw in note:
                        scores[label] += (3 - i)

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    # 4순위: subfamily 추가 fallback
    fallback = {"AMBER": "Amber", "FRESH": "Fresh",
                "AQUATIC": "Fresh", "GREEN": "Fresh", "MUSK": "Amber"}
    for key, label in fallback.items():
        if key in family_up or key in subfamily_up:
            return label

    return "Unknown"


def parse_ingredients(raw) -> str:
    """'[\"Rose\", \"Lemon\"]' 형식의 문자열을 'Rose, Lemon'으로 변환합니다."""
    if not raw or pd.isna(raw):
        return ""
    try:
        items = ast.literal_eval(str(raw))
        if isinstance(items, list):
            return ", ".join(str(i).strip() for i in items if i)
    except Exception:
        pass
    # fallback: 따옴표/대괄호만 제거
    return str(raw).strip("[]").replace("'", "").replace('"', "")


# ──────────────────────────────────────────────
# 이미지 압축 해제
# ──────────────────────────────────────────────

def extract_images(zip_path: Path):
    IMAGE_DIR.mkdir(exist_ok=True)
    existing = set(os.listdir(IMAGE_DIR))

    print(f"[1단계] 이미지 압축 해제 중: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".jpg")]
        to_extract = [m for m in members if Path(m).name not in existing]
        print(f"  전체 {len(members):,}개 중 {len(to_extract):,}개 신규 해제")
        for i, member in enumerate(to_extract, 1):
            fname = Path(member).name
            data  = zf.read(member)
            (IMAGE_DIR / fname).write_bytes(data)
            if i % 2000 == 0:
                print(f"  [{i:,}/{len(to_extract):,}] 해제 중...")
    print(f"  완료: {IMAGE_DIR}")


# ──────────────────────────────────────────────
# CSV 변환
# ──────────────────────────────────────────────

def convert_csv(csv_path: Path) -> pd.DataFrame:
    print(f"[2단계] CSV 변환 중: {csv_path.name}")
    df = pd.read_csv(csv_path, sep="|", on_bad_lines="skip", encoding="utf-8")
    print(f"  원본 행 수: {len(df):,}")

    existing_images = set(os.listdir(IMAGE_DIR)) if IMAGE_DIR.exists() else set()

    rows = []
    skipped_no_img = 0
    skipped_unknown = 0

    for _, row in df.iterrows():
        name       = str(row.get("name_perfume", "")).strip()
        brand      = str(row.get("brand", "")).strip()
        family     = str(row.get("family", ""))
        subfamily  = str(row.get("subfamily", ""))
        image_name = str(row.get("image_name", "")).strip()

        notes = parse_ingredients(row.get("ingredients", ""))
        if not name or not notes:
            continue

        label = classify_label(notes, family, subfamily)
        if label == "Unknown":
            skipped_unknown += 1
            continue

        # 이미지 파일 확인
        if image_name not in existing_images:
            skipped_no_img += 1
            continue

        rows.append({
            "image_path": str(Path("perfume_images") / image_name),
            "label":      label,
            "name":       name,
            "brand":      brand,
            "notes":      notes,
            "image_url":  "",
        })

    result = pd.DataFrame(rows)
    buf = io.StringIO()
    print(f"  변환 결과: {len(result):,}개", file=buf)
    print(f"  제외: 이미지 없음={skipped_no_img:,}, 라벨 불명={skipped_unknown:,}", file=buf)
    print("  라벨 분포:", file=buf)
    print(result["label"].value_counts().to_string(), file=buf)
    sys.stdout.buffer.write(buf.getvalue().encode("utf-8", errors="replace"))
    return result


# ──────────────────────────────────────────────
# 병합
# ──────────────────────────────────────────────

def merge_with_all_cleaned(new_df: pd.DataFrame):
    buf = io.StringIO()
    if ALL_CSV.exists():
        old_df = pd.read_csv(ALL_CSV, encoding="utf-8-sig")
        # 기존 luckyscent 데이터의 7클래스 라벨 → 4클래스로 리매핑
        old_df["label"] = old_df["label"].map(LABEL_MAP_4).fillna(old_df["label"])
        existing_keys = set(
            zip(old_df["name"].str.strip().str.lower(),
                old_df["brand"].str.strip().str.lower())
        )
        mask = ~new_df.apply(
            lambda r: (r["name"].strip().lower(), r["brand"].strip().lower())
                      in existing_keys,
            axis=1,
        )
        added  = new_df[mask]
        merged = pd.concat([old_df, added], ignore_index=True)
        print(f"\n[3단계] 병합: 기존 {len(old_df):,} + 신규 {len(added):,} = {len(merged):,}", file=buf)
    else:
        merged = new_df
        print(f"\n[3단계] all_cleaned.csv 신규 생성: {len(merged):,}개", file=buf)

    merged.to_csv(ALL_CSV, index=False, encoding="utf-8-sig")
    print(f"  저장: {ALL_CSV}", file=buf)
    print("  최종 라벨 분포:", file=buf)
    print(merged["label"].value_counts().to_string(), file=buf)
    sys.stdout.buffer.write(buf.getvalue().encode("utf-8", errors="replace"))


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def run(no_merge: bool = False, skip_extract: bool = False):
    IMAGE_DIR.mkdir(exist_ok=True)

    zip_path = find_cached_file("images.zip")
    csv_path = find_cached_file("perfumes.csv")

    if not skip_extract:
        extract_images(zip_path)
    else:
        print("[건너뜀] 이미지 압축 해제 생략 (--skip_extract)")

    result = convert_csv(csv_path)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {OUTPUT_CSV}")

    if not no_merge:
        merge_with_all_cleaned(result)
        print("\n[완료] 이제 학습을 실행하세요:")
        print("  python perfume_classifier/main.py --mode full")
    else:
        print(f"\n[완료] {OUTPUT_CSV} 생성됨 (병합 생략)")
        print("  병합하려면: python data/raw/hf_downloader.py --skip_extract")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="doevent/perfume HuggingFace 데이터 통합")
    parser.add_argument("--no_merge",     action="store_true",
                        help="all_cleaned.csv 병합 생략")
    parser.add_argument("--skip_extract", action="store_true",
                        help="이미지 압축 해제 생략 (이미 완료된 경우)")
    args = parser.parse_args()
    run(no_merge=args.no_merge, skip_extract=args.skip_extract)
