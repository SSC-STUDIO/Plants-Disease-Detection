import os

base = "C:/Users/EliuaK_Csy/Plants-Disease-Detection"
val_file = os.path.join(base, "data/ai_challenger_pdr2018_validationset_20181023.zip")

print(f"Checking: {val_file}")
print(f"Exists: {os.path.exists(val_file)}")

if os.path.exists(val_file):
    print(f"Size: {os.path.getsize(val_file)} bytes")

    # Try to open and read
    try:
        with open(val_file, 'rb') as f:
            header = f.read(4)
            print(f"Header: {header}")
            print(f"Can read: True")
    except PermissionError as e:
        print(f"Permission denied: {e}")
    except Exception as e:
        print(f"Error: {e}")