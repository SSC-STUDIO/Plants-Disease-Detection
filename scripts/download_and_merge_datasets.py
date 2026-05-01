#!/usr/bin/env python3
"""
下载并合并多个公开植物病害数据集
支持的数据集:
1. PlantVillage (HuggingFace)
2. New Plant Diseases Dataset (Kaggle)
3. PlantWild / PlantSeg (HuggingFace / Google Drive)
4. FieldPlant (需要手动下载)
5. DiaMOS Plant (Zenodo)
6. Plant Leaf Disease Benchmark (Mendeley)
"""

import os
import sys
import json
import shutil
import zipfile
import tarfile
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config, paths


# ============ 数据集配置 ============
DATASET_CONFIGS = {
    "plantvillage": {
        "name": "PlantVillage",
        "source": "huggingface",
        "repo": "mohanty/PlantVillage",
        "config": "color",
        "expected_images": 54306,
        "license": "CC BY-SA 3.0",
    },
    "new_plant_diseases": {
        "name": "New Plant Diseases Dataset",
        "source": "kaggle",
        "dataset": "vipoooool/new-plant-diseases-dataset",
        "expected_images": 87000,
        "license": "CC BY",
    },
    "plantwild": {
        "name": "PlantWild",
        "source": "huggingface",
        "repo": "tianqiwei/PlantWild",
        "expected_images": 50000,
        "license": "CC BY-NC-ND 4.0",
    },
    "diamos": {
        "name": "DiaMOS Plant",
        "source": "zenodo",
        "url": "https://zenodo.org/record/5557313/files/Pear.zip",
        "expected_images": 3505,
        "license": "CC BY",
    },
}

DOWNLOAD_DIR = Path(paths.data_dir) / "raw_datasets"
MERGED_TRAIN_DIR = Path(paths.merged_train_dir)
MERGED_VAL_DIR = Path(paths.merged_val_dir)
MERGED_TEST_DIR = Path(paths.merged_test_dir)


def ensure_dirs():
    """创建必要的目录"""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_VAL_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_TEST_DIR.mkdir(parents=True, exist_ok=True)


def download_huggingface_dataset(name, repo, config_name=None):
    """从 HuggingFace 下载数据集"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets -q")
        from datasets import load_dataset

    target_dir = DOWNLOAD_DIR / name
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[{name}] Already downloaded at {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] Downloading from HuggingFace: {repo}")

    if config_name:
        ds = load_dataset(repo, config_name)
    else:
        ds = load_dataset(repo)

    # 保存为图像文件
    split_counts = {}
    for split_name, split_data in ds.items():
        split_dir = target_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        label_dirs = defaultdict(list)
        for i, item in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
            img = item["image"]
            label = item.get("label", item.get("disease", "unknown"))
            if isinstance(label, int):
                # 需要映射到名称
                label_names = split_data.features["label"].names
                label = label_names[label]

            label_dirs[label].append((i, img))

        for label, items in tqdm(label_dirs.items(), desc=f"Writing {split_name} classes"):
            label_dir = split_dir / sanitize_class_name(label)
            label_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in items:
                img_path = label_dir / f"{idx:06d}.jpg"
                img.save(img_path)

        split_counts[split_name] = len(split_data)

    print(f"[{name}] Downloaded: {split_counts}")
    return target_dir


def download_kaggle_dataset(name, dataset):
    """从 Kaggle 下载数据集"""
    target_dir = DOWNLOAD_DIR / name
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[{name}] Already downloaded at {target_dir}")
        return target_dir

    print(f"[{name}] Downloading from Kaggle: {dataset}")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 使用 kaggle API 下载
    zip_path = target_dir / "dataset.zip"
    cmd = f'kaggle datasets download -d {dataset} -p "{target_dir}"'
    ret = os.system(cmd)
    if ret != 0:
        print(f"ERROR: Failed to download {name}. Make sure kaggle API is configured.")
        return None

    # 解压
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)
    zip_path.unlink()

    print(f"[{name}] Downloaded and extracted to {target_dir}")
    return target_dir


def download_zenodo_dataset(name, url):
    """从 Zenodo 下载数据集"""
    import urllib.request

    target_dir = DOWNLOAD_DIR / name
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[{name}] Already downloaded at {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] Downloading from Zenodo: {url}")

    filename = Path(url).name
    file_path = target_dir / filename

    urllib.request.urlretrieve(url, file_path)

    # 解压
    if filename.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zf:
            zf.extractall(target_dir)
    elif filename.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(file_path, 'r:gz') as tf:
            tf.extractall(target_dir)

    file_path.unlink()
    print(f"[{name}] Downloaded and extracted to {target_dir}")
    return target_dir


def sanitize_class_name(name):
    """清理类别名称，使其适合作为目录名"""
    if not isinstance(name, str):
        name = str(name)
    # 替换非法字符
    name = name.replace("___", "_").replace(" ", "_")
    name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    name = name.strip("_")
    return name


def discover_images(root_dir):
    """发现目录中的所有图像文件，返回 (path, class_name) 列表"""
    root = Path(root_dir)
    images = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_exts:
            # 类别名是父目录名
            class_name = sanitize_class_name(path.parent.name)
            images.append((str(path), class_name))

    return images


def analyze_dataset_structure(dataset_dir, ds_name=""):
    """分析数据集目录结构，尝试识别 train/val/test 划分"""
    dataset_dir = Path(dataset_dir)

    # 常见的划分目录名
    split_names = ["train", "training", "val", "validation", "valid", "test", "testing"]
    found_splits = {}

    for split in split_names:
        split_dir = dataset_dir / split
        if split_dir.exists():
            key = "train" if split in ["train", "training"] else \
                  "val" if split in ["val", "validation", "valid"] else "test"
            found_splits[key] = split_dir

    if found_splits:
        return found_splits

    # 对于 ai_challenger，有 train 和 val 目录
    for split in split_names:
        split_dir = dataset_dir / split
        if split_dir.exists():
            key = "train" if split in ["train", "training"] else \
                  "val" if split in ["val", "validation", "valid"] else "test"
            found_splits[key] = split_dir

    if found_splits:
        return found_splits

    # 如果没有划分目录，检查是否有类别目录直接在根目录下
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if class_dirs:
        return {"train": dataset_dir}

    return {}


def merge_datasets(dataset_names, val_ratio=0.15, test_ratio=0.15, seed=42):
    """合并多个数据集到统一的目录结构"""
    import random
    random.seed(seed)

    # 收集所有图像和类别
    all_classes = set()
    split_images = defaultdict(list)  # split -> [(path, class_name, source)]

    for ds_name in dataset_names:
        ds_dir = DOWNLOAD_DIR / ds_name
        if not ds_dir.exists():
            print(f"WARNING: Dataset {ds_name} not found at {ds_dir}, skipping")
            continue

        print(f"\nAnalyzing {ds_name}...")
        structure = analyze_dataset_structure(ds_dir, ds_name)

        if not structure:
            print(f"WARNING: Could not determine structure of {ds_name}, skipping")
            continue

        for split, split_dir in structure.items():
            images = discover_images(split_dir)

            # SECURITY FIX: 过滤掉无标签的 test 目录（如 testa, testb）
            # 这些目录名不是有效的类别名
            valid_images = []
            for path, class_name in images:
                # 跳过 ai_challenger 的 testa/testb 等无标签测试集
                if ds_name == "ai_challenger" and split == "test" and class_name in ["testa", "testb"]:
                    continue
                # 跳过非数字类别名（对于 ai_challenger，类别应该是数字）
                if ds_name == "ai_challenger" and split in ["train", "val"]:
                    try:
                        int(class_name)
                    except ValueError:
                        continue
                valid_images.append((path, class_name))

            for path, class_name in valid_images:
                split_images[split].append((path, class_name, ds_name))
                all_classes.add(class_name)

            print(f"  {split}: {len(valid_images)} images")

    # 如果没有 val/test，从 train 中划分
    if "train" in split_images:
        train_items = split_images["train"]

        if "val" not in split_images or not split_images["val"]:
            n_val = int(len(train_items) * val_ratio)
            val_items = random.sample(train_items, min(n_val, len(train_items)))
            split_images["val"] = val_items
            train_set = set(id(item) for item in val_items)
            train_items = [item for item in train_items if id(item) not in train_set]
            split_images["train"] = train_items

        if "test" not in split_images or not split_images["test"]:
            n_test = int(len(train_items) * test_ratio)
            test_items = random.sample(train_items, min(n_test, len(train_items)))
            split_images["test"] = test_items
            test_set = set(id(item) for item in test_items)
            train_items = [item for item in train_items if id(item) not in test_set]
            split_images["train"] = train_items

    # 创建统一的类别映射
    sorted_classes = sorted(all_classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}

    print(f"\nTotal classes: {len(sorted_classes)}")
    for split in ["train", "val", "test"]:
        print(f"Total {split} images: {len(split_images.get(split, []))}")

    # 复制文件到合并目录
    print("\nMerging datasets...")
    for split, split_dir in [("train", MERGED_TRAIN_DIR), ("val", MERGED_VAL_DIR), ("test", MERGED_TEST_DIR)]:
        # 清理旧数据
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        items = split_images.get(split, [])
        if not items:
            continue

        # 并行复制
        def copy_item(args):
            path, class_name, source = args
            idx = class_to_idx[class_name]
            target_class_dir = split_dir / f"{idx:04d}"
            target_class_dir.mkdir(parents=True, exist_ok=True)

            src_path = Path(path)
            # 添加来源前缀避免重名
            target_name = f"{source}_{src_path.name}"
            target_path = target_class_dir / target_name

            # 处理重名
            counter = 1
            while target_path.exists():
                stem = src_path.stem
                suffix = src_path.suffix
                target_name = f"{source}_{stem}_{counter:03d}{suffix}"
                target_path = target_class_dir / target_name
                counter += 1

            shutil.copy2(src_path, target_path)
            return True

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(copy_item, items), total=len(items), desc=f"Copying {split}"))

    # 保存类别映射
    mapping_path = MERGED_TRAIN_DIR.parent / "class_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    # 保存类别统计
    stats = {}
    for split, split_dir in [("train", MERGED_TRAIN_DIR), ("val", MERGED_VAL_DIR), ("test", MERGED_TEST_DIR)]:
        if not split_dir.exists():
            continue
        class_counts = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.iterdir()))
                class_counts[class_dir.name] = count
        stats[split] = {
            "total_images": sum(class_counts.values()),
            "total_classes": len(class_counts),
            "class_counts": class_counts,
        }

    stats_path = MERGED_TRAIN_DIR.parent / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nMerge complete!")
    print(f"Class mapping saved to: {mapping_path}")
    print(f"Dataset stats saved to: {stats_path}")
    print(f"Total classes: {len(sorted_classes)}")

    return sorted_classes, class_to_idx


def update_config_num_classes(num_classes):
    """更新 config.py 中的 num_classes"""
    config_path = Path(__file__).parent.parent / "config.py"
    content = config_path.read_text(encoding='utf-8')

    # 替换 num_classes
    import re
    new_content = re.sub(
        r'num_classes:\s*int\s*=\s*\d+',
        f'num_classes: int = {num_classes}',
        content
    )

    if new_content != content:
        config_path.write_text(new_content, encoding='utf-8')
        print(f"Updated config.py: num_classes = {num_classes}")
    else:
        print(f"config.py already has num_classes = {num_classes} or pattern not found")


def main():
    parser = argparse.ArgumentParser(description="Download and merge plant disease datasets")
    parser.add_argument("--datasets", nargs="+", default=["plantvillage", "new_plant_diseases"],
                        help="Datasets to download")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, only merge existing")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    if not args.skip_download:
        for ds_name in args.datasets:
            cfg = DATASET_CONFIGS.get(ds_name)
            if not cfg:
                print(f"WARNING: Unknown dataset {ds_name}, skipping")
                continue

            source = cfg["source"]
            if source == "huggingface":
                download_huggingface_dataset(ds_name, cfg["repo"], cfg.get("config"))
            elif source == "kaggle":
                download_kaggle_dataset(ds_name, cfg["dataset"])
            elif source == "zenodo":
                download_zenodo_dataset(ds_name, cfg["url"])
            else:
                print(f"WARNING: Unknown source {source} for {ds_name}")

    # 合并
    sorted_classes, class_to_idx = merge_datasets(
        args.datasets,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # 更新配置
    update_config_num_classes(len(sorted_classes))


if __name__ == "__main__":
    main()
