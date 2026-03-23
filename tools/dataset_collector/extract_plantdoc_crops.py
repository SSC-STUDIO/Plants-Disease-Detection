#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def datasets_root() -> Path:
    return repo_root().parents[2] / "Datasets"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract PlantDoc bounding boxes into classification crops")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=datasets_root() / "DataScience" / "PlantDoc-Object-Detection-Dataset(Linux)",
        help="PlantDoc dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=datasets_root() / "Processed" / "PlantDoc-Crops-Classification",
        help="Output directory for the crop-based classification dataset",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output dir before extraction")
    parser.add_argument("--padding-ratio", type=float, default=0.08, help="Crop padding ratio around each box")
    parser.add_argument("--min-width", type=int, default=64, help="Minimum crop width")
    parser.add_argument("--min-height", type=int, default=64, help="Minimum crop height")
    return parser.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool):
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    return normalized.strip("_") or "unknown"


def find_image_path(split_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = split_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        candidate_upper = split_dir / f"{stem}{ext.upper()}"
        if candidate_upper.exists():
            return candidate_upper
    raise FileNotFoundError(f"Missing image for stem: {stem}")


def clamp_box(xmin, ymin, xmax, ymax, width, height):
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)
    return xmin, ymin, xmax, ymax


def padded_box(xmin, ymin, xmax, ymax, width, height, padding_ratio):
    box_width = xmax - xmin
    box_height = ymax - ymin
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)
    return clamp_box(xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y, width, height)


def extract_split(split_name: str, split_dir: Path, output_dir: Path, padding_ratio: float, min_width: int, min_height: int):
    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    distribution = Counter()
    crop_count = 0
    skipped = 0

    for xml_path in sorted(split_dir.glob("*.xml")):
        try:
            image_path = find_image_path(split_dir, xml_path.stem)
            tree = ET.parse(xml_path)
        except Exception:
            skipped += 1
            continue

        root = tree.getroot()
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                img_width, img_height = image.size
                for index, obj in enumerate(root.findall(".//object")):
                    class_name = (obj.findtext("name") or "").strip()
                    bbox = obj.find("bndbox")
                    if not class_name or bbox is None:
                        continue

                    xmin = int(float(bbox.findtext("xmin", "0")))
                    ymin = int(float(bbox.findtext("ymin", "0")))
                    xmax = int(float(bbox.findtext("xmax", "0")))
                    ymax = int(float(bbox.findtext("ymax", "0")))
                    xmin, ymin, xmax, ymax = padded_box(xmin, ymin, xmax, ymax, img_width, img_height, padding_ratio)

                    if xmax - xmin < min_width or ymax - ymin < min_height:
                        skipped += 1
                        continue

                    crop = image.crop((xmin, ymin, xmax, ymax))
                    class_slug = slugify(class_name)
                    class_dir = split_output_dir / class_slug
                    class_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = f"{image_path.stem}_{index:02d}.jpg"
                    crop.save(class_dir / crop_name, quality=95)

                    distribution[class_slug] += 1
                    crop_count += 1
        except Exception:
            skipped += 1

    return {
        "split": split_name,
        "images_scanned": len(list(split_dir.glob('*.xml'))),
        "crops": crop_count,
        "classes": len(distribution),
        "class_distribution": dict(sorted(distribution.items())),
        "skipped": skipped,
    }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_clean_dir(output_dir, args.overwrite)

    train_summary = extract_split(
        split_name="train",
        split_dir=source_dir / "TRAIN",
        output_dir=output_dir,
        padding_ratio=args.padding_ratio,
        min_width=args.min_width,
        min_height=args.min_height,
    )
    test_summary = extract_split(
        split_name="test",
        split_dir=source_dir / "TEST",
        output_dir=output_dir,
        padding_ratio=args.padding_ratio,
        min_width=args.min_width,
        min_height=args.min_height,
    )

    all_classes = sorted(set(train_summary["class_distribution"]) | set(test_summary["class_distribution"]))
    labels = {
        str(index): {
            "id": index,
            "name": class_name,
        }
        for index, class_name in enumerate(all_classes)
    }

    manifest = {
        "dataset_name": output_dir.name,
        "dataset_type": "classification_crops_from_detection",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "config": {
            "padding_ratio": args.padding_ratio,
            "min_width": args.min_width,
            "min_height": args.min_height,
        },
        "splits": {
            "train": train_summary,
            "test": test_summary,
        },
        "labels": labels,
    }

    write_json(output_dir / "labels.json", labels)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
