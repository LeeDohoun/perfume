"""데이터 분포 심층 분석"""
import pandas as pd

note_train = pd.read_csv("data/note/train.csv")
brand_train = pd.read_csv("data/brand/train.csv")
raw = pd.read_csv("data/raw/all_cleaned.csv")

print("=== Note 클래스별 분석 ===")
nc = note_train["label"].value_counts()
print(nc)
print(f"Min: {nc.min()}, Max: {nc.max()}, Ratio: {nc.max()/nc.min():.1f}x")

print("\n=== Brand 클래스별 분석 ===")
bc = brand_train["label"].value_counts()
print("Top 5:")
print(bc.head())
print("\nBottom 5:")
print(bc.tail())
print(f"\nMin: {bc.min()}, Max: {bc.max()}, Ratio: {bc.max()/bc.min():.1f}x")
print(f"15개 미만 클래스: {(bc < 15).sum()}개")
print(f"20개 미만 클래스: {(bc < 20).sum()}개")

print("\n=== 원본 전체 데이터 분석 ===")
print(f"원본 총 샘플: {len(raw)}")
brand_col = "brand"
print(f"원본 브랜드 수: {raw[brand_col].nunique()}")
all_brand_cnt = raw[brand_col].value_counts()
print(f"15개+ 브랜드: {(all_brand_cnt >= 15).sum()}개")
print(f"10개+ 브랜드: {(all_brand_cnt >= 10).sum()}개")
print(f"5개+ 브랜드:  {(all_brand_cnt >= 5).sum()}개")

used_note = 1258
used_brand = 815
print(f"\n=== 활용률 ===")
print(f"Note 태스크: {used_note}/{len(raw)} = {used_note/len(raw)*100:.1f}%")
print(f"Brand 태스크: {used_brand}/{len(raw)} = {used_brand/len(raw)*100:.1f}%")
