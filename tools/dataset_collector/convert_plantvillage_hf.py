#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "test": "test",
    "validation": "val",
    "val": "val",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def datasets_root() -> Path:
    return repo_root().parents[2] / "Datasets"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the Hugging Face mirror download of PlantVillage into a classification directory layout",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=datasets_root() / "DataScience" / "PlantVillage-HF",
        help="Directory created by huggingface_hub.snapshot_download for mohanty/PlantVillage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=datasets_root() / "Processed" / "PlantVillage-Color-Classification",
        help="Output directory for the converted classification dataset",
    )
    parser.add_argument(
        "--config",
        choices=["color", "grayscale", "segmented"],
        default="color",
        help="PlantVillage image variant to convert",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_split_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line.replace("\\", "/"))
    return rows


def class_name_from_member(member_name: str, config: str) -> str:
    parts = Path(member_name).parts
    expected_prefix = ("raw", config)
    if len(parts) < 4 or parts[0:2] != expected_prefix:
        raise ValueError(f"Unexpected PlantVillage member path: {member_name}")
    return parts[2]


def safe_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "image"


def numeric_label_dir(label_id: int, class_name: str) -> str:
    return str(label_id)


def split_source_rows(source_dir: Path, config: str) -> Dict[str, List[str]]:
    split_dir = source_dir / "splits"
    rows = {}
    for split_name in ("train", "test"):
        rows[split_name] = read_split_file(split_dir / f"{config}_{split_name}.txt")
    return rows


def collect_classes(split_rows: Dict[str, List[str]], config: str) -> List[str]:
    classes = {
        class_name_from_member(member_name, config)
        for rows in split_rows.values()
        for member_name in rows
    }
    return sorted(classes)


def zip_member_lookup(zip_file: zipfile.ZipFile) -> Dict[str, str]:
    lookup = {}
    for member_name in zip_file.namelist():
        normalized = member_name.replace("\\", "/")
        if Path(normalized).suffix.lower() in IMAGE_EXTENSIONS:
            lookup[normalized] = member_name
    return lookup


def extract_split(
    zip_file: zipfile.ZipFile,
    member_lookup: Dict[str, str],
    split_name: str,
    rows: Iterable[str],
    label_to_id: Dict[str, int],
    output_dir: Path,
    config: str,
) -> Tuple[Dict[str, int], List[str]]:
    split_output_dir = output_dir / SPLIT_ALIASES[split_name]
    distribution: Counter[str] = Counter()
    missing: List[str] = []

    for index, member_name in enumerate(rows, start=1):
        class_name = class_name_from_member(member_name, config)
        label_id = label_to_id[class_name]
        class_dir = split_output_dir / numeric_label_dir(label_id, class_name)
        class_dir.mkdir(parents=True, exist_ok=True)

        original_name = Path(member_name).name
        target_name = f"{index:06d}_{safe_filename(original_name)}"
        target_path = class_dir / target_name

        zip_member = member_lookup.get(member_name)
        if zip_member is None:
            missing.append(member_name)
            continue

        with zip_file.open(zip_member) as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)

        distribution[class_name] += 1
        if index % 5000 == 0:
            print(f"[{split_name}] converted {index} entries")

    return dict(sorted(distribution.items())), missing


def main():
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    zip_path = source_dir / "data.zip"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not zip_path.exists():
        raise FileNotFoundError(f"PlantVillage archive not found: {zip_path}")

    split_rows = split_source_rows(source_dir, args.config)
    classes = collect_classes(split_rows, args.config)
    label_to_id = {class_name: index for index, class_name in enumerate(classes)}
    labels = {
        str(label_id): {
            "id": label_id,
            "name": class_name,
            "directory": numeric_label_dir(label_id, class_name),
        }
        for class_name, label_id in sorted(label_to_id.items(), key=lambda item: item[1])
    }

    ensure_clean_dir(output_dir, args.overwrite)

    split_summaries = {}
    missing_members: List[str] = []
    with zipfile.ZipFile(zip_path) as zip_file:
        lookup = zip_member_lookup(zip_file)
        for split_name, rows in split_rows.items():
            distribution, missing = extract_split(
                zip_file=zip_file,
                member_lookup=lookup,
                split_name=split_name,
                rows=rows,
                label_to_id=label_to_id,
                output_dir=output_dir,
                config=args.config,
            )
            missing_members.extend(missing)
            split_summaries[SPLIT_ALIASES[split_name]] = {
                "source_split": split_name,
                "images": sum(distribution.values()),
                "classes": len(distribution),
                "class_distribution": distribution,
            }

    manifest = {
        "dataset_name": output_dir.name,
        "dataset_type": "classification",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "source_archive": str(zip_path),
        "output_dir": str(output_dir),
        "config": args.config,
        "url": "https://huggingface.co/datasets/mohanty/PlantVillage",
        "license": "CC BY-SA 3.0",
        "redistributable": True,
        "citation": "PlantVillage Dataset, Hughes and Salathe, 2015",
        "splits": split_summaries,
        "labels": labels,
        "missing_members": missing_members[:100],
        "missing_member_count": len(missing_members),
    }

    write_json(output_dir / "labels.json", labels)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
