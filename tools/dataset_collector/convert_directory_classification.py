#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert class-name image directories to numeric-label directories")
    parser.add_argument("--source-dir", type=Path, required=True, help="Source directory containing class folders")
    parser.add_argument("--output-dir", type=Path, required=True, help="Converted classification output directory")
    parser.add_argument("--split-name", default="train", help="Output split name")
    parser.add_argument("--url", default="", help="Source URL for the manifest")
    parser.add_argument("--license", default="", help="Source license for the manifest")
    parser.add_argument("--citation", default="", help="Source citation for the manifest")
    parser.add_argument("--dataset-name", default="", help="Dataset name for the manifest")
    parser.add_argument("--copy-mode", choices=["copy", "link"], default="link")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory before conversion")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "image"


def iter_images(class_dir: Path) -> Iterable[Path]:
    for dirpath, dirs, filenames in os.walk(class_dir):
        dirs[:] = sorted(name for name in dirs if name not in {".git", "__pycache__"})
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def class_dirs(source_dir: Path) -> List[Path]:
    return sorted(path for path in source_dir.iterdir() if path.is_dir() and path.name not in {".git", "__pycache__"})


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def convert(args) -> Dict:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    reset_output_dir(output_dir, args.overwrite)
    classes = class_dirs(source_dir)
    label_to_id = {path.name: index for index, path in enumerate(classes)}
    distribution: Counter[str] = Counter()

    for class_dir in classes:
        label_id = label_to_id[class_dir.name]
        target_class_dir = output_dir / args.split_name / str(label_id)
        for index, image_path in enumerate(iter_images(class_dir), start=1):
            rel_parts = image_path.relative_to(class_dir).parts
            rel_name = "__".join(safe_name(part) for part in rel_parts)
            target_path = target_class_dir / f"{index:06d}_{rel_name}"
            copy_or_link(image_path, target_path, args.copy_mode)
            distribution[class_dir.name] += 1
        print(f"[{class_dir.name}] converted {distribution[class_dir.name]} images")

    labels = {
        str(label_id): {
            "id": label_id,
            "name": class_name,
            "directory": str(label_id),
        }
        for class_name, label_id in sorted(label_to_id.items(), key=lambda item: item[1])
    }
    manifest = {
        "dataset_name": args.dataset_name or output_dir.name,
        "dataset_type": "classification",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "url": args.url,
        "license": args.license,
        "citation": args.citation,
        "splits": {
            args.split_name: {
                "images": sum(distribution.values()),
                "classes": len(distribution),
                "class_distribution": dict(sorted(distribution.items())),
            }
        },
        "labels": labels,
    }
    write_json(output_dir / "labels.json", labels)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main():
    args = parse_args()
    manifest = convert(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
