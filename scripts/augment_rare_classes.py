#!/usr/bin/env python3
"""
对训练集中样本数少的类别进行离线数据增强，
使每个类别的训练样本数达到指定阈值。
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

from PIL import Image, ImageEnhance
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config, paths


def get_transforms():
    """定义增强变换函数列表"""
    def flip_h(img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def flip_v(img):
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    def rotate_90(img):
        return img.transpose(Image.ROTATE_90)

    def rotate_180(img):
        return img.transpose(Image.ROTATE_180)

    def rotate_270(img):
        return img.transpose(Image.ROTATE_270)

    def brightness_up(img):
        return ImageEnhance.Brightness(img).enhance(1.3)

    def brightness_down(img):
        return ImageEnhance.Brightness(img).enhance(0.7)

    def contrast_up(img):
        return ImageEnhance.Contrast(img).enhance(1.3)

    def contrast_down(img):
        return ImageEnhance.Contrast(img).enhance(0.7)

    def saturation_up(img):
        return ImageEnhance.Color(img).enhance(1.3)

    def saturation_down(img):
        return ImageEnhance.Color(img).enhance(0.7)

    def sharpness_up(img):
        return ImageEnhance.Sharpness(img).enhance(1.5)

    def scale_up(img):
        w, h = img.size
        return img.resize((int(w * 1.1), int(h * 1.1)), Image.BILINEAR).crop((w * 0.05, h * 0.05, w * 0.95 + w * 0.1, h * 0.95 + h * 0.1))

    def scale_down(img):
        w, h = img.size
        new_w, new_h = int(w * 0.9), int(h * 0.9)
        tmp = img.resize((new_w, new_h), Image.BILINEAR)
        result = Image.new(img.mode, (w, h))
        result.paste(tmp, ((w - new_w) // 2, (h - new_h) // 2))
        return result

    return [
        flip_h, flip_v, rotate_90, rotate_180, rotate_270,
        brightness_up, brightness_down, contrast_up, contrast_down,
        saturation_up, saturation_down, sharpness_up,
        scale_up, scale_down,
    ]


def augment_class(class_dir, target_count, transforms, seed=42):
    """对指定类别目录进行增强，直到达到目标数量"""
    random.seed(seed)
    images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
    images.sort()
    current = len(images)

    if current >= target_count:
        return 0

    needed = target_count - current
    generated = 0

    pbar = tqdm(total=needed, desc=f"Augmenting {class_dir.name}", leave=False)
    while generated < needed:
        for img_path in images:
            if generated >= needed:
                break
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    # 随机选择 1-3 个变换组合
                    n_transforms = random.randint(1, min(3, len(transforms)))
                    chosen = random.sample(transforms, n_transforms)
                    aug_img = img
                    for t in chosen:
                        aug_img = t(aug_img)

                    # 生成文件名
                    suffix = img_path.suffix
                    stem = img_path.stem
                    aug_name = f"{stem}_aug_{generated:04d}{suffix}"
                    aug_path = class_dir / aug_name
                    counter = 1
                    while aug_path.exists():
                        aug_name = f"{stem}_aug_{generated:04d}_{counter:03d}{suffix}"
                        aug_path = class_dir / aug_name
                        counter += 1

                    aug_img.save(aug_path, quality=95)
                    generated += 1
                    pbar.update(1)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    pbar.close()
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=500, help="每个类别的目标样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dir = Path(paths.merged_train_dir)
    stats = json.load(open('data/dataset_stats.json'))
    train_counts = stats['train']['class_counts']

    transforms = get_transforms()
    total_generated = 0

    rare_classes = [(cls, count) for cls, count in train_counts.items() if count < args.threshold]
    rare_classes.sort(key=lambda x: x[1])

    print(f"Found {len(rare_classes)} classes with < {args.threshold} samples")
    for cls, count in rare_classes:
        class_dir = train_dir / cls
        n = augment_class(class_dir, args.threshold, transforms, seed=args.seed)
        total_generated += n
        print(f"  Class {cls}: {count} -> {count + n} (+{n})")

    print(f"\nTotal augmented images: {total_generated}")

    # 更新统计
    new_stats = {}
    for split, split_dir in [('train', 'data/merged_train'), ('val', 'data/merged_val'), ('test', 'data/merged_test')]:
        class_counts = {}
        for class_dir in sorted(Path(split_dir).iterdir()):
            if class_dir.is_dir():
                class_counts[class_dir.name] = len(list(class_dir.iterdir()))
        new_stats[split] = {
            'total_images': sum(class_counts.values()),
            'total_classes': len(class_counts),
            'class_counts': class_counts,
        }

    with open('data/dataset_stats.json', 'w') as f:
        json.dump(new_stats, f, indent=2)

    print("Updated dataset_stats.json")
    print(f"Train: {new_stats['train']['total_images']} images, {new_stats['train']['total_classes']} classes")


if __name__ == "__main__":
    main()
