import zipfile
import os

base = "C:/Users/EliuaK_Csy/Plants-Disease-Detection"

# Extract validation set
val_zip = os.path.join(base, "data/ai_challenger_pdr2018_validationset_20181023.zip")
val_dir = os.path.join(base, "data/temp/dataset/AgriculturalDisease_validationset")

print(f"Extracting {val_zip}...")
os.makedirs(val_dir, exist_ok=True)

with zipfile.ZipFile(val_zip, 'r') as z:
    z.extractall(val_dir)

print(f"✓ Extracted to {val_dir}")

# Check extraction
files = os.listdir(val_dir)
print(f"Found {len(files)} items")