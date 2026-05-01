#!/usr/bin/env python3
"""
重新平衡 train/val/test 划分，确保：
1. 每个类别在所有 split 中都有样本
2. val 和 test 从每个类别中按比例抽取
3. 删除样本数过少的类别（默认 < 3）
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config, paths


def collect_images(split_dir):
    """收集某个 split 下所有图像，按类别分组"""
    split_dir = Path(split_dir)
    classes = defaultdict(list)
    if not split_dir.exists():
        return classes
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.is_file():
                classes[class_dir.name].append(str(img_path))
    return classes


def rebalance(min_samples=3, val_per_class=5, test_per_class=5, seed=42):
    random.seed(seed)

    train_dir = Path(paths.merged_train_dir)
    val_dir = Path(paths.merged_val_dir)
    test_dir = Path(paths.merged_test_dir)
    temp_all_dir = train_dir.parent / "temp_all"

    # 收集当前所有数据（合并 train/val/test）
    all_images = defaultdict(list)
    for split_dir in [train_dir, val_dir, test_dir]:
        for cls, imgs in collect_images(split_dir).items():
            all_images[cls].extend(imgs)

    print(f"Total classes before filtering: {len(all_images)}")

    # 过滤样本数过少的类别
    valid_classes = {cls: imgs for cls, imgs in all_images.items() if len(imgs) >= min_samples}
    removed = set(all_images.keys()) - set(valid_classes.keys())
    if removed:
        print(f"Removed {len(removed)} classes with < {min_samples} total samples: {sorted(removed)[:10]}...")

    # 先把所有有效类别的图像复制到临时目录，避免清空后源文件丢失
    if temp_all_dir.exists():
        shutil.rmtree(temp_all_dir)
    temp_all_dir.mkdir(parents=True, exist_ok=True)

    print("\nConsolidating images to temp directory...")
    for cls, imgs in valid_classes.items():
        cls_dir = temp_all_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for src in imgs:
            src_path = Path(src)
            dst = cls_dir / src_path.name
            counter = 1
            while dst.exists():
                dst = cls_dir / f"{src_path.stem}_{counter:03d}{src_path.suffix}"
                counter += 1
            shutil.copy2(src, dst)

    # 重新从 temp_all_dir 收集
    valid_classes = collect_images(temp_all_dir)

    # 确保每个类别在 val/test 中至少有指定数量
    new_train = defaultdict(list)
    new_val = defaultdict(list)
    new_test = defaultdict(list)

    for cls, imgs in sorted(valid_classes.items()):
        random.shuffle(imgs)
        total = len(imgs)

        n_test = min(test_per_class, max(1, total // 10))
        n_val = min(val_per_class, max(1, total // 10))
        if total - n_test - n_val < 1:
            n_test = min(1, total - 1)
            n_val = min(1, total - n_test - 1)
            if n_val < 0:
                n_val = 0

        new_test[cls] = imgs[:n_test]
        new_val[cls] = imgs[n_test:n_test + n_val]
        new_train[cls] = imgs[n_test + n_val:]

    print(f"Total classes after filtering: {len(valid_classes)}")

    # 清空并重建目标目录
    for split_dir in [train_dir, val_dir, test_dir]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    # 复制文件到目标 split
    def copy_files(split_dict, split_dir):
        total = sum(len(imgs) for imgs in split_dict.values())
        copied = 0
        for cls, imgs in split_dict.items():
            class_dir = split_dir / cls
            class_dir.mkdir(parents=True, exist_ok=True)
            for src in imgs:
                src_path = Path(src)
                dst = class_dir / src_path.name
                counter = 1
                while dst.exists():
                    dst = class_dir / f"{src_path.stem}_{counter:03d}{src_path.suffix}"
                    counter += 1
                shutil.copy2(src, dst)
                copied += 1
        return copied

    print("\nRebuilding splits...")
    train_total = copy_files(new_train, train_dir)
    val_total = copy_files(new_val, val_dir)
    test_total = copy_files(new_test, test_dir)

    print(f"Train: {train_total} images, {len(new_train)} classes")
    print(f"Val:   {val_total} images, {len(new_val)} classes")
    print(f"Test:  {test_total} images, {len(new_test)} classes")

    # 保存统计
    stats = {}
    for split_name, split_dict in [("train", new_train), ("val", new_val), ("test", new_test)]:
        class_counts = {cls: len(imgs) for cls, imgs in split_dict.items()}
        stats[split_name] = {
            "total_images": sum(class_counts.values()),
            "total_classes": len(class_counts),
            "class_counts": class_counts,
        }

    stats_path = train_dir.parent / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")

    # 更新 class_mapping
    sorted_classes = sorted(valid_classes.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
    mapping_path = train_dir.parent / "class_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    print(f"Class mapping saved to {mapping_path}")
    print(f"Total classes: {len(sorted_classes)}")

    # 清理临时目录
    shutil.rmtree(temp_all_dir)
    print(f"Cleaned up temp directory")

    return len(sorted_classes)


def update_config_num_classes(num_classes):
    config_path = Path(__file__).parent.parent / "config.py"
    content = config_path.read_text(encoding='utf-8')
    import re
    new_content = re.sub(
        r'num_classes:\s*int\s*=\s*\d+',
        f'num_classes: int = {num_classes}',
        content
    )
    if new_content != content:
        config_path.write_text(new_content, encoding='utf-8')
        print(f"Updated config.py: num_classes = {num_classes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples", type=int, default=3, help="每个类别最少总样本数")
    parser.add_argument("--val-per-class", type=int, default=5, help="每个类别在 val 中最少样本数")
    parser.add_argument("--test-per-class", type=int, default=5, help="每个类别在 test 中最少样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_classes = rebalance(
        min_samples=args.min_samples,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed
    )
    update_config_num_classes(num_classes)


if __name__ == "__main__":
    main()
