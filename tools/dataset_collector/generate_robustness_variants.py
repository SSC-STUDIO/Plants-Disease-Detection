#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageEnhance, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate traceable robustness variants from a numeric classification dataset")
    parser.add_argument("--source-dir", type=Path, required=True, help="Source dataset root containing train/test/val splits")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset root")
    parser.add_argument("--target-gb", type=float, required=True, help="Stop after output reaches this logical size")
    parser.add_argument("--quality", type=int, default=98, help="JPEG quality for generated variants")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def split_and_label(path: Path, source_dir: Path) -> Tuple[str, str]:
    rel = path.relative_to(source_dir)
    if len(rel.parts) < 3:
        raise ValueError(f"Expected split/class/image path, got: {path}")
    return rel.parts[0], rel.parts[1]


def output_size_bytes(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def reset_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def variant_image(image: Image.Image, variant: str) -> Image.Image:
    image = image.convert("RGB")
    if variant == "jpeg_high":
        return image
    if variant == "jpeg_medium":
        return ImageEnhance.Contrast(image).enhance(1.08)
    if variant == "blur_light":
        return image.filter(ImageFilter.GaussianBlur(radius=0.7))
    if variant == "sharpen":
        return image.filter(ImageFilter.SHARPEN)
    if variant == "bright":
        return ImageEnhance.Brightness(image).enhance(1.16)
    if variant == "dark":
        return ImageEnhance.Brightness(image).enhance(0.86)
    raise ValueError(f"Unknown variant: {variant}")


def save_variant(src: Path, dst: Path, variant: str, quality: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        transformed = variant_image(image, variant)
        transformed.save(dst, format="JPEG", quality=quality, optimize=False, progressive=False)


def load_labels(source_dir: Path) -> Dict:
    labels_path = source_dir / "labels.json"
    if labels_path.exists():
        return json.loads(labels_path.read_text(encoding="utf-8"))
    labels = {}
    for class_dir in sorted((source_dir / "train").iterdir()):
        if class_dir.is_dir() and class_dir.name.isdigit():
            label_id = int(class_dir.name)
            labels[str(label_id)] = {"id": label_id, "name": class_dir.name, "directory": class_dir.name}
    return labels


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    random.seed(args.seed)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    reset_output(output_dir, args.overwrite)
    target_bytes = int(args.target_gb * (1024 ** 3))
    variants = ["jpeg_high", "jpeg_medium", "blur_light", "sharpen", "bright", "dark"]
    images = list(iter_images(source_dir))
    random.shuffle(images)

    distribution: Dict[str, Counter[str]] = {}
    generated = 0
    current_size = output_size_bytes(output_dir)
    round_index = 0

    while current_size < target_bytes:
        variant = variants[round_index % len(variants)]
        for src in images:
            if current_size >= target_bytes:
                break
            try:
                split, label = split_and_label(src, source_dir)
            except ValueError:
                continue
            dst = output_dir / split / label / f"{src.stem}__robust_{variant}_{round_index:02d}.jpg"
            if dst.exists():
                continue
            try:
                save_variant(src, dst, variant=variant, quality=args.quality)
            except Exception:
                continue
            generated += 1
            distribution.setdefault(split, Counter())[label] += 1
            current_size += dst.stat().st_size
            if generated % 5000 == 0:
                print(json.dumps({
                    "event": "progress",
                    "generated": generated,
                    "gb": round(current_size / (1024 ** 3), 3),
                    "variant": variant,
                }, ensure_ascii=False), flush=True)
        round_index += 1
        if round_index > len(variants) * 10:
            break

    labels = load_labels(source_dir)
    manifest = {
        "dataset_name": output_dir.name,
        "dataset_type": "classification_robustness_variants",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "target_gb": args.target_gb,
        "actual_gb": round(output_size_bytes(output_dir) / (1024 ** 3), 3),
        "license": "Derived from source dataset; follow source license",
        "citation": "Generated deterministic robustness variants for training and evaluation",
        "config": {
            "variants": variants,
            "quality": args.quality,
            "seed": args.seed,
        },
        "splits": {
            split: {
                "images": sum(counter.values()),
                "classes": len(counter),
                "class_distribution": dict(sorted(counter.items())),
            }
            for split, counter in sorted(distribution.items())
        },
        "labels": labels,
    }
    write_json(output_dir / "labels.json", labels)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
