"""
scraper.py
----------
luckyscent.com 향수 데이터 추가 수집기.

sitemap에서 전체 제품 URL을 가져와 이름/브랜드/노트/설명/이미지를 수집합니다.
노트 키워드 기반으로 4개 클래스 라벨을 자동 분류합니다.

사용법
------
  python data/raw/scraper.py                 # 전체 수집
  python data/raw/scraper.py --limit 20      # 20개 테스트
  python data/raw/scraper.py --resume        # 중단된 지점부터 이어서
  python data/raw/scraper.py --delay 2.0     # 요청 간격 조정 (기본 1.5초)
"""

import argparse
import os
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR     = BASE_DIR / "perfume_images"
RAW_DIR       = Path(__file__).resolve().parent
OUTPUT_CSV    = RAW_DIR / "scraped.csv"
PROGRESS_FILE = RAW_DIR / "scrape_progress.txt"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

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


def classify_label(notes: str) -> str:
    """노트 문자열에서 매칭 점수가 가장 높은 라벨을 반환합니다.
    동점 시 노트 앞쪽(주요 노트)에 나온 키워드를 우선합니다."""
    notes_lower = notes.lower()
    note_list = [n.strip() for n in notes_lower.split(",")]

    scores = {label: 0 for label in LABEL_KEYWORDS}
    for label, keywords in LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw in notes_lower:
                scores[label] += 1
                # 앞쪽 3개 노트에 있으면 가중치 추가
                for i, note in enumerate(note_list[:3]):
                    if kw in note:
                        scores[label] += (3 - i)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Unknown"


# ──────────────────────────────────────────────
# sitemap → 제품 URL 목록
# ──────────────────────────────────────────────

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def get_all_product_urls() -> list:
    urls = []
    for i in range(1, 22):
        sitemap_url = f"https://www.luckyscent.com/sitemap/products/{i}.xml"
        try:
            r = requests.get(sitemap_url, headers=HEADERS, timeout=10)
            root = ET.fromstring(r.content)
            for loc in root.findall(".//sm:loc", SITEMAP_NS):
                urls.append(loc.text.strip())
            print(f"  sitemap {i:2d}/21 → 누적 {len(urls):,}개")
            time.sleep(0.5)
        except Exception as e:
            print(f"  [경고] sitemap {i} 실패: {e}")
    return urls


# ──────────────────────────────────────────────
# 제품 페이지 파싱
# ──────────────────────────────────────────────

def scrape_product(url: str) -> dict:
    """제품 URL 하나에서 메타데이터를 추출합니다. 실패 시 None 반환."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")

        # 제품명
        h1 = soup.find("h1")
        name = h1.get_text(strip=True) if h1 else ""

        # 브랜드 — URL slug에서 추출이 가장 신뢰도 높음
        # 패턴: /products/{product-name}-by-{brand-name}
        slug = url.rstrip("/").split("/")[-1]
        if "-by-" in slug:
            brand_slug = slug.split("-by-", 1)[-1]
            brand = brand_slug.replace("-", " ").title()
        else:
            # slug에 -by- 없으면 HTML에서 파싱
            brand_tag = soup.find(
                "a",
                href=lambda h: h and h.startswith("/brands/") and len(h.strip("/")) > len("brands")
            )
            brand = brand_tag.get_text(strip=True) if brand_tag else ""

        # 노트 (필터 링크)
        note_tags = soup.find_all("a", href=lambda h: h and "f.l.notes=" in h)
        notes = ", ".join(a.get_text(strip=True) for a in note_tags)

        # 설명 (The Scoop 섹션)
        description = ""
        for tag in soup.find_all(["h2", "h3", "strong", "b"]):
            if "scoop" in tag.get_text(strip=True).lower():
                nxt = tag.find_next_sibling()
                if nxt:
                    description = nxt.get_text(separator=" ", strip=True)
                break

        # 이미지 (Shopify CDN 우선, 없으면 static.luckyscent)
        img_tag = soup.find("img", src=lambda s: s and "cdn.shopify.com" in s)
        if not img_tag:
            img_tag = soup.find("img", src=lambda s: s and "luckyscent.com" in s)
        image_url = ""
        if img_tag:
            src = img_tag.get("src", "")
            image_url = ("https:" + src) if src.startswith("//") else src

        if not name or not notes:
            return None

        return {
            "name":        name,
            "brand":       brand,
            "notes":       notes,
            "description": description,
            "image_url":   image_url,
            "label":       classify_label(notes),
            "source_url":  url,
        }
    except Exception as e:
        print(f"  [경고] 파싱 실패 {url}: {e}")
        return None


# ──────────────────────────────────────────────
# 이미지 다운로드
# ──────────────────────────────────────────────

def download_image(image_url: str, save_path: Path) -> bool:
    if not image_url:
        return False
    if save_path.exists():
        return True
    try:
        r = requests.get(image_url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            save_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def run_scraper(limit: int = 0, resume: bool = False, delay: float = 1.5):
    IMAGE_DIR.mkdir(exist_ok=True)

    # 이미 처리한 URL 로드
    done_urls: set = set()
    if resume and PROGRESS_FILE.exists():
        done_urls = set(PROGRESS_FILE.read_text(encoding="utf-8").splitlines())
        print(f"[Resume] 이전 처리 URL: {len(done_urls):,}개")

    # 기존 결과 로드
    rows = []
    if resume and OUTPUT_CSV.exists():
        rows = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig").to_dict("records")
        print(f"[Resume] 기존 수집 데이터: {len(rows):,}개")

    # 1단계: URL 수집
    print("\n[1단계] sitemap에서 제품 URL 수집 중...")
    all_urls = get_all_product_urls()
    targets = [u for u in all_urls if u not in done_urls]
    if limit:
        targets = targets[:limit]
    print(f"\n수집 대상: {len(targets):,}개 (전체 {len(all_urls):,}개 중)")

    # 2단계: 스크래핑
    print("\n[2단계] 제품 페이지 스크래핑 중...\n")
    for i, url in enumerate(targets, 1):
        data = scrape_product(url)

        if data:
            slug = url.rstrip("/").split("/")[-1][:60]
            img_path = IMAGE_DIR / f"scraped_{slug}.jpg"
            ok = download_image(data["image_url"], img_path)
            data["image_path"] = (
                str(Path("perfume_images") / f"scraped_{slug}.jpg") if ok else ""
            )
            rows.append(data)

        # 진행 상황 기록
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(url + "\n")

        # 50개마다 중간 저장
        if i % 50 == 0 or i == len(targets):
            df = pd.DataFrame(rows)
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            label_dist = df["label"].value_counts().to_dict() if len(df) else {}
            print(f"  [{i:4d}/{len(targets)}] 수집 {len(rows):,}개 | {label_dist}")

        time.sleep(delay + random.uniform(0.0, 0.5))

    # 최종 저장
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[완료] 총 {len(rows):,}개 수집")
    print(f"  CSV  : {OUTPUT_CSV}")
    print(f"  라벨 분포:")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="luckyscent 향수 데이터 수집기")
    parser.add_argument("--limit",  type=int,   default=0,
                        help="수집 개수 제한 (0=전체, 테스트용)")
    parser.add_argument("--resume", action="store_true",
                        help="중단된 지점부터 이어서 수집")
    parser.add_argument("--delay",  type=float, default=1.5,
                        help="요청 간 딜레이 초 (기본 1.5)")
    args = parser.parse_args()
    run_scraper(limit=args.limit, resume=args.resume, delay=args.delay)
