from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = DATA_ROOT / "All"
OUTPUT_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")

DIRECT_LABEL_MAP = {
    "Fresh": "Fresh",
    "Citrus": "Fresh",
    "Floral": "Floral",
    "Woody": "Woody",
    "Amber_Oriental": "Amber",
    "Spicy": "Amber",
}

# Sweet is not one of the four main fragrance-wheel families.
# Keep Amber as the default destination, and only move rows away from Amber
# when the note evidence is clearly stronger for another family.
SWEET_NOTE_KEYWORDS = {
    "Amber": (
        "amber",
        "ambergris",
        "benzoin",
        "caramel",
        "chocolate",
        "cinnamon",
        "coffee",
        "cocoa",
        "honey",
        "incense",
        "labdanum",
        "licorice",
        "marshmallow",
        "myrrh",
        "praline",
        "resin",
        "rum",
        "tobacco",
        "tonka",
        "vanilla",
    ),
    "Fresh": (
        "aldehyde",
        "apple",
        "bergamot",
        "blackcurrant",
        "citron",
        "citrus",
        "grapefruit",
        "green",
        "herb",
        "lemon",
        "lime",
        "mandarin",
        "marine",
        "mint",
        "orange",
        "ozonic",
        "pear",
        "petitgrain",
        "rhubarb",
        "water",
        "yuzu",
    ),
    "Floral": (
        "carnation",
        "freesia",
        "gardenia",
        "geranium",
        "heliotrope",
        "iris",
        "jasmine",
        "lavender",
        "lily",
        "magnolia",
        "mimosa",
        "orange blossom",
        "orris",
        "osmanthus",
        "peony",
        "rose",
        "tuberose",
        "violet",
        "ylang",
    ),
    "Woody": (
        "agarwood",
        "cashmeran",
        "cedar",
        "cedarwood",
        "cypress",
        "guaiac",
        "leather",
        "moss",
        "oakmoss",
        "oud",
        "patchouli",
        "pine",
        "sandalwood",
        "suede",
        "vetiver",
        "wood",
        "woody",
    ),
}

NON_AMBER_SWITCH_MARGIN = 2


def keyword_score(notes: str, family: str) -> int:
    lowered = notes.casefold()
    occupied_spans: list[tuple[int, int]] = []
    score = 0

    # Prefer longer phrases first so `cedarwood` does not also score as `cedar`
    # and `wood`, and `orange blossom` does not split across families.
    for keyword in sorted(SWEET_NOTE_KEYWORDS[family], key=len, reverse=True):
        pattern = rf"(?<![a-z]){re.escape(keyword.casefold())}(?![a-z])"
        for match in re.finditer(pattern, lowered):
            span = match.span()
            if any(start < span[1] and span[0] < end for start, end in occupied_spans):
                continue
            occupied_spans.append(span)
            score += 1

    return score


def remap_sweet(notes: str) -> tuple[str, dict[str, int]]:
    scores = {family: keyword_score(notes, family) for family in SWEET_NOTE_KEYWORDS}
    amber_score = scores["Amber"]
    alternative_families = ("Floral", "Woody", "Fresh")
    best_alternative = max(alternative_families, key=lambda family: scores[family])
    best_alternative_score = scores[best_alternative]

    if best_alternative_score >= amber_score + NON_AMBER_SWITCH_MARGIN:
        return best_alternative, scores

    if amber_score == 0 and best_alternative_score == 0:
        return "Amber", scores

    return "Amber", scores


def remap_label(original_label: str, notes: str) -> tuple[str, dict[str, int] | None]:
    if original_label == "Sweet":
        return remap_sweet(notes)

    return DIRECT_LABEL_MAP[original_label], None


def process_split(split: str) -> dict[str, object]:
    source_path = SOURCE_DIR / f"{split}.csv"
    output_path = OUTPUT_DIR / f"{split}.csv"

    original_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    sweet_routes: Counter[str] = Counter()
    sweet_examples: list[dict[str, object]] = []

    with source_path.open("r", encoding="utf-8", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {source_path}")

        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            original_label = row["label"]
            notes = row.get("notes", "") or ""
            mapped_label, sweet_scores = remap_label(original_label, notes)

            original_counts[original_label] += 1
            mapped_counts[mapped_label] += 1
            row["label"] = mapped_label
            writer.writerow(row)

            if original_label == "Sweet":
                sweet_routes[mapped_label] += 1
                if len(sweet_examples) < 20:
                    sweet_examples.append(
                        {
                            "name": row.get("name", ""),
                            "brand": row.get("brand", ""),
                            "mapped_label": mapped_label,
                            "scores": sweet_scores,
                            "notes": notes,
                        }
                    )

    return {
        "source": str(source_path),
        "output": str(output_path),
        "original_counts": dict(sorted(original_counts.items())),
        "mapped_counts": dict(sorted(mapped_counts.items())),
        "sweet_routes": dict(sorted(sweet_routes.items())),
        "sweet_examples": sweet_examples,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {split: process_split(split) for split in SPLITS}
    summary_path = OUTPUT_DIR / "label_remap_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved remapped CSV files to: {OUTPUT_DIR}")
    print(f"Saved summary report to: {summary_path}")


if __name__ == "__main__":
    main()
