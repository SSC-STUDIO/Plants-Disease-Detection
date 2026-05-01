import os

data_dir = "C:/Users/EliuaK_Csy/Plants-Disease-Detection/data"
val_file = os.path.join(data_dir, "ai_challenger_pdr2018_validationset_20181023.zip")

print(f"File exists: {os.path.exists(val_file)}")
print(f"File size: {os.path.getsize(val_file)} bytes")

# Try to read first few bytes
try:
    with open(val_file, 'rb') as f:
        header = f.read(4)
        print(f"Header bytes: {header.hex()}")
        print(f"Valid ZIP: {header == b'PK\\x03\\x04'}")
except Exception as e:
    print(f"Error reading file: {e}")