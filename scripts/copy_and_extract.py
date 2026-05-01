import os
import shutil
import zipfile

base = "C:/Users/EliuaK_Csy/Plants-Disease-Detection"
data_dir = os.path.join(base, "data")
temp_dir = os.path.join(base, "data/temp/locked")

# Create temp directory
os.makedirs(temp_dir, exist_ok=True)

val_file = os.path.join(data_dir, "ai_challenger_pdr2018_validationset_20181023.zip")
temp_file = os.path.join(temp_dir, "validationset_copy.zip")

print(f"Copying {val_file}...")
try:
    # Try to copy the file
    shutil.copy2(val_file, temp_file)
    print(f"Copied to {temp_file}")

    # Extract
    val_dir = os.path.join(base, "data/temp/dataset/AgriculturalDisease_validationset")
    os.makedirs(val_dir, exist_ok=True)

    print(f"Extracting to {val_dir}...")
    with zipfile.ZipFile(temp_file, 'r') as z:
        z.extractall(val_dir)

    print("✓ Extraction successful!")

    # Clean up temp copy
    os.remove(temp_file)

except Exception as e:
    print(f"Error: {e}")