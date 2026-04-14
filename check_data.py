"""데이터 무결성 검증 스크립트"""
import pandas as pd
import os

print("=" * 60)
print("향수 이미지 분류 프로젝트 - 데이터 무결성 검증")
print("=" * 60)

for task in ["note", "brand"]:
    print(f"\n{'=' * 40}")
    print(f"Task: {task.upper()}")
    print(f"{'=' * 40}")
    for split in ["train", "val", "test"]:
        path = os.path.join("data", task, f"{split}.csv")
        df = pd.read_csv(path)
        
        # 이미지 존재 확인
        missing = sum(1 for p in df["image_path"] if not os.path.exists(p))
        
        label_counts = df["label"].value_counts()
        print(f"\n  [{split}] {len(df)}개 | 누락 이미지: {missing}개 | 라벨 수: {df['label'].nunique()}")
        for lbl, cnt in label_counts.items():
            print(f"    - {lbl}: {cnt}")

print("\n\n=== 이미지 폴더 확인 ===")
img_dir = "perfume_images"
img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
print(f"총 이미지 수: {len(img_files)}")
print(f"첫 5개: {img_files[:5]}")
print(f"마지막 5개: {img_files[-5:]}")
