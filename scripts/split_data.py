import json
import os
import random
import shutil

base = "C:/Users/EliuaK_Csy/Plants-Disease-Detection"
data_dir = os.path.join(base, "data")

# Load training annotations
ann_file = os.path.join(base, "data/temp/labels/AgriculturalDisease_train_annotations.json")

print("Loading training annotations...")
with open(ann_file, 'r') as f:
    annotations = json.load(f)

print(f"Total training samples: {len(annotations)}")

# Group by class for stratified split
from collections import defaultdict
class_items = defaultdict(list)
for ann in annotations:
    class_items[ann['disease_class']].append(ann)

# Split 90% train, 10% val
train_anns = []
val_anns = []

for class_id, items in class_items.items():
    random.shuffle(items)
    split_idx = int(len(items) * 0.9)
    train_anns.extend(items[:split_idx])
    val_anns.extend(items[split_idx:])
    print(f"Class {class_id}: {len(items)} -> train {split_idx}, val {len(items)-split_idx}")

# Save annotations
train_ann_file = os.path.join(base, "data/temp/labels/train_split.json")
val_ann_file = os.path.join(base, "data/temp/labels/val_split.json")

print(f"\nSaving train annotations ({len(train_anns)} items)...")
with open(train_ann_file, 'w') as f:
    json.dump(train_anns, f, indent=2)

print(f"Saving val annotations ({len(val_anns)} items)...")
with open(val_ann_file, 'w') as f:
    json.dump(val_anns, f, indent=2)

print("\nDone! Use these files for training and validation.")