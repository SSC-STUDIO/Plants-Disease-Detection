import zipfile
import os
import shutil

base = "C:/Users/EliuaK_Csy/Plants-Disease-Detection"
data_dir = os.path.join(base, "data")

# Extract testb as validation
testb_zip = os.path.join(data_dir, "ai_challenger_pdr2018_testb_20181023.zip")
val_dir = os.path.join(base, "data/temp/dataset/AgriculturalDisease_validationset")

print(f"Extracting testb as validation set...")

os.makedirs(val_dir, exist_ok=True)

with zipfile.ZipFile(testb_zip, 'r') as z:
    z.extractall(val_dir)

print(f"Extracted to {val_dir}")

# Create validation annotations (testb has similar structure)
# Find annotation file if exists
for root, dirs, files in os.walk(val_dir):
    for f in files:
        if 'annotation' in f.lower() and f.endswith('.json'):
            print(f"Found annotation: {os.path.join(root, f)}")

print("Done!")