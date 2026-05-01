import json
import os

base = "C:/Users/EliuaK_Csy/Plants-Disease-Detection"
ann_file = os.path.join(base, "data/temp/labels/AgriculturalDisease_train_annotations.json")

print(f"Loading annotation file: {ann_file}")

with open(ann_file, 'r') as f:
    data = json.load(f)

print(f"Annotation type: {type(data)}")

if isinstance(data, list):
    print(f"Total items: {len(data)}")
    print(f"\nFirst item:")
    print(json.dumps(data[0], indent=2))
    print(f"\nSecond item:")
    print(json.dumps(data[1], indent=2))
elif isinstance(data, dict):
    print(f"Keys: {data.keys()}")