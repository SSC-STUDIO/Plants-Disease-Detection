import os
import subprocess

base = "C:/Users/EliuaK_Csy/Plants-Disease-Detection"
val_file = os.path.join(base, "data/ai_challenger_pdr2018_validationset_20181023.zip")
dest_dir = os.path.join(base, "data/temp/dataset/AgriculturalDisease_validationset")

print(f"Extracting validation set using tarfile...")
print(f"Source: {val_file}")
print(f"Dest: {dest_dir}")

os.makedirs(dest_dir, exist_ok=True)

import tarfile
try:
    with tarfile.open(val_file, 'r:zip') as tar:
        tar.extractall(dest_dir)
    print("Extraction complete!")
except Exception as e:
    print(f"Error: {e}")

# List contents
if os.path.exists(dest_dir):
    contents = os.listdir(dest_dir)
    print(f"Extracted {len(contents)} items")