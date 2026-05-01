import zipfile
import os

base = "C:/Users/EliuaK_Csy/Plants-Disease-Detection"
data_dir = os.path.join(base, "data")

files = [
    "ai_challenger_pdr2018_testa_20181023.zip",
    "ai_challenger_pdr2018_testb_20181023.zip"
]

for fname in files:
    fpath = os.path.join(data_dir, fname)
    print(f"Checking {fname}...")
    try:
        with zipfile.ZipFile(fpath, 'r') as z:
            # Test read
            bad_file = z.testzip()
            if bad_file:
                print(f"  X Corrupt file: {bad_file}")
            else:
                print(f"  OK ({len(z.namelist())} files)")
    except Exception as e:
        print(f"  X Error: {e}")