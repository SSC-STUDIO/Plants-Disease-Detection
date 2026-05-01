"""
组织训练数据到正确的文件夹结构
根据标注文件将图像复制到 ./data/train/<class_id>/ 文件夹
"""
import os
import json
import shutil
from tqdm import tqdm
from collections import defaultdict

# 远程服务器基础路径
base = "C:/Users/EliuaK_Csy/Plants-Disease-Detection"
os.chdir(base)

# 路径配置
annotations_file = "data/temp/labels/train_split.json"
source_images_dir = "data/temp/dataset/AgriculturalDisease_trainingset/AgriculturalDisease_trainingset/images"
target_train_dir = "data/train"

print("=" * 60)
print("Training Data Organization Script")
print("=" * 60)

# 检查标注文件
if not os.path.exists(annotations_file):
    print(f"Error: Annotations file not found: {annotations_file}")
    exit(1)

# 检查源图像目录
if not os.path.exists(source_images_dir):
    print(f"Error: Source images directory not found: {source_images_dir}")
    exit(1)

# 加载标注
print(f"\nLoading annotations from: {annotations_file}")
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

print(f"Total samples: {len(annotations)}")

# 统计每个类别的样本数
class_counts = defaultdict(int)
for ann in annotations:
    class_counts[ann['disease_class']] += 1

print(f"\nClass distribution:")
for class_id in sorted(class_counts.keys()):
    print(f"  Class {class_id}: {class_counts[class_id]} samples")

# 创建目标目录
print(f"\nCreating target directories...")
for class_id in class_counts.keys():
    class_dir = os.path.join(target_train_dir, str(class_id))
    os.makedirs(class_dir, exist_ok=True)
    print(f"  Created: {class_dir}")

# 复制文件
print(f"\nCopying images to {target_train_dir}/...")
copied = 0
missing = 0
errors = []

for ann in tqdm(annotations, desc="Copying"):
    image_id = ann['image_id']
    class_id = ann['disease_class']

    # 源文件路径
    src_file = os.path.join(source_images_dir, image_id)

    # 目标文件路径
    dst_dir = os.path.join(target_train_dir, str(class_id))
    dst_file = os.path.join(dst_dir, image_id)

    if not os.path.exists(src_file):
        missing += 1
        errors.append(f"Missing: {src_file}")
        continue

    try:
        shutil.copy2(src_file, dst_file)
        copied += 1
    except Exception as e:
        errors.append(f"Error copying {src_file}: {e}")

# 输出统计
print("\n" + "=" * 60)
print("Data organization completed!")
print("=" * 60)
print(f"Copied: {copied}")
print(f"Missing: {missing}")
print(f"Errors: {len(errors)}")

if errors:
    print(f"\nFirst 10 errors:")
    for error in errors[:10]:
        print(f"  {error}")

# 验证结果
print(f"\nVerifying {target_train_dir} structure:")
for class_id in sorted(class_counts.keys()):
    class_dir = os.path.join(target_train_dir, str(class_id))
    if os.path.exists(class_dir):
        file_count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
        print(f"  Class {class_id}: {file_count} files")
    else:
        print(f"  Class {class_id}: Directory not found!")

print("\n" + "=" * 60)
print("Data is ready for training!")
print("=" * 60)
