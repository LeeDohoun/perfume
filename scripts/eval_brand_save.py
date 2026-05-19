"""
Brand EfficientNet-B0 평가 결과를 results/efficientnet/brand/ 에 저장.
Usage: python scripts/eval_brand_save.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from train_efficientnet import PerfumeDataset, build_model, get_transforms

CHECKPOINT = "checkpoints/brand/best_model.pth"
TEST_CSV   = "data/brand/test.csv"
OUT_DIR    = "results/efficientnet/brand"
DEVICE     = torch.device("cpu")

os.makedirs(OUT_DIR, exist_ok=True)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
num_classes = checkpoint["num_classes"]
idx2label   = checkpoint["idx2label"]
target_names = [idx2label[i] for i in range(len(idx2label))]

_, eval_tf = get_transforms()
dataset = PerfumeDataset(TEST_CSV, eval_tf)
loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

model = build_model(num_classes, freeze_backbone=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().to(DEVICE)

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(DEVICE))
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
report   = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
cm       = confusion_matrix(all_labels, all_preds)

# classification_report.txt
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Model        : EfficientNet-B0 brand classification (35 frequent brands)\n")
    f.write(f"Brands       : {num_classes}\n")
    f.write(f"Test samples : {len(dataset)}\n")
    f.write(f"Accuracy     : {accuracy:.4f}\n")
    f.write(f"Random Base  : {1/num_classes:.4f}\n\n")
    f.write(report)

# metrics.json
lines = [l for l in report.split("\n") if "macro avg" in l]
macro_parts = lines[0].split() if lines else []
macro_f1 = float(macro_parts[4]) if len(macro_parts) >= 5 else None
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump({"accuracy": accuracy, "macro_f1": macro_f1, "num_brands": num_classes,
               "test_samples": len(dataset), "random_baseline": 1/num_classes}, f, indent=2)

# confusion_matrix.png
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Brand Classification Confusion Matrix\nTest Accuracy: {accuracy*100:.2f}%")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

print(f"Saved to {OUT_DIR}")
print(f"Accuracy: {accuracy*100:.2f}%  |  Macro F1: {macro_f1}")
