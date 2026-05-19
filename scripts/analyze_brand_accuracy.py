"""
Audit the brand-classification evaluation setup.

This script does not require a saved model checkpoint. It checks whether the
reported brand accuracy is measured on a narrow/frequent-brand closed set,
whether exact duplicate samples leak across splits, and how much the selected
brand set covers the original data.
"""
import argparse
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")


def normalize_text(series):
    return series.fillna("").astype(str).str.strip()


def normalize_path(series):
    return (
        normalize_text(series)
        .str.replace("\\\\", "/", regex=False)
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def read_splits(brand_dir):
    splits = {}
    for split in SPLITS:
        df = pd.read_csv(brand_dir / f"{split}.csv")
        df["label"] = normalize_text(df["label"])
        df["image_path_norm"] = normalize_path(df["image_path"])
        df["image_url_norm"] = normalize_text(df.get("image_url", pd.Series([""] * len(df)))).str.lower()
        df["name_norm"] = normalize_text(df.get("name", pd.Series([""] * len(df)))).str.lower()
        splits[split] = df
    return splits


def exact_overlaps(splits, column):
    rows = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(splits[left][column]) & set(splits[right][column])
        rows.append((f"{left}-{right}", len(overlap), sorted(overlap)[:3]))
    return rows


def macro_recall_bounds(test_counts, error_count):
    """Best/worst possible macro recall given only total error count."""
    if error_count <= 0:
        return 1.0, 1.0
    per_error_loss = sorted((1.0 / n for n in test_counts), reverse=True)
    worst_loss = sum(per_error_loss[:error_count])
    best_loss = sum(sorted(1.0 / n for n in test_counts)[:error_count])
    class_count = len(test_counts)
    return max(0.0, 1.0 - worst_loss / class_count), max(0.0, 1.0 - best_loss / class_count)


def pct(value):
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Audit brand classification accuracy conditions.")
    parser.add_argument("--raw-csv", default="data/raw/all_cleaned.csv")
    parser.add_argument("--brand-dir", default="data/brand")
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--logged-accuracy", type=float, default=0.9512)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv)
    raw["brand"] = normalize_text(raw["brand"])
    raw = raw[raw["brand"] != ""].copy()
    raw_counts = raw["brand"].value_counts()
    selected_counts = raw_counts[raw_counts >= args.min_samples]
    rare_counts = raw_counts[raw_counts < args.min_samples]

    splits = read_splits(Path(args.brand_dir))
    all_split_rows = pd.concat(
        [df.assign(split=split) for split, df in splits.items()],
        ignore_index=True,
    )
    split_counts = (
        pd.DataFrame({split: df["label"].value_counts().sort_index() for split, df in splits.items()})
        .fillna(0)
        .astype(int)
    )
    split_counts["total"] = split_counts.sum(axis=1)
    test_counts = splits["test"]["label"].value_counts()
    test_n = len(splits["test"])
    correct = round(args.logged_accuracy * test_n)
    errors = test_n - correct
    macro_low, macro_high = macro_recall_bounds(test_counts.tolist(), errors)

    selected_coverage = selected_counts.sum() / len(raw)
    all_data_upper_bound = args.logged_accuracy * selected_coverage
    majority_baseline = test_counts.max() / test_n
    random_baseline = 1 / splits["test"]["label"].nunique()

    print("# Brand Accuracy Audit")
    print()
    print("## Data Coverage")
    print(f"- Raw rows: {len(raw):,}")
    print(f"- Raw brands: {raw_counts.shape[0]:,}")
    print(f"- Brands kept with >= {args.min_samples} samples: {selected_counts.shape[0]:,}")
    print(f"- Kept samples: {selected_counts.sum():,} ({pct(selected_coverage)})")
    print(f"- Excluded rare brands: {rare_counts.shape[0]:,}")
    print(f"- Excluded rare-brand samples: {rare_counts.sum():,} ({pct(rare_counts.sum() / len(raw))})")
    print()
    print("## Split Shape")
    for split, df in splits.items():
        counts = df["label"].value_counts()
        print(
            f"- {split}: {len(df):,} rows, {df['label'].nunique():,} classes, "
            f"class count range {counts.min()}-{counts.max()}"
        )
    print()
    print("## Logged Accuracy Context")
    print(f"- Logged test accuracy: {pct(args.logged_accuracy)}")
    print(f"- Test rows: {test_n:,}")
    print(f"- Implied correct / wrong: {correct:,} / {errors:,}")
    print(f"- Closed-set random baseline: {pct(random_baseline)}")
    print(f"- Test majority-class baseline: {pct(majority_baseline)}")
    print(f"- If rare brands are treated as unsupported, raw-data upper bound: {pct(all_data_upper_bound)}")
    print(f"- Possible macro recall range from {errors} errors: {pct(macro_low)} - {pct(macro_high)}")
    print()
    print("## Exact Split Leakage Checks")
    for column in ("image_path_norm", "image_url_norm", "name_norm"):
        print(f"- {column}:")
        for pair, count, examples in exact_overlaps(splits, column):
            suffix = f" examples={examples}" if examples else ""
            print(f"  - {pair}: {count}{suffix}")
    print()
    print("## Selected Brand Count Extremes")
    print("- Top 10 selected brands:")
    for label, count in selected_counts.head(10).items():
        print(f"  - {label}: {count}")
    print("- Bottom 10 selected brands:")
    for label, count in selected_counts.tail(10).items():
        print(f"  - {label}: {count}")
    print()
    print("## Test Count Distribution")
    for samples, brand_count in test_counts.value_counts().sort_index().items():
        print(f"- {brand_count} brands have {samples} test sample(s)")
    print()
    duplicates = all_split_rows[all_split_rows.duplicated("image_url_norm", keep=False)]
    print(f"## Duplicate Image URLs In Brand Splits: {len(duplicates)} rows")


if __name__ == "__main__":
    main()
