#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def datasets_root() -> Path:
    return repo_root().parents[2] / "Datasets"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert AI Challenger PDR2018 archives into a classification layout")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=datasets_root() / "DataScience" / "Plant-Disese-Dataset",
        help="Directory containing the AI Challenger zip archives",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=datasets_root() / "Processed" / "AI-Challenger-PDR2018-Classification",
        help="Output directory for the converted classification dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool):
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_split(zip_path: Path, ann_member: str, image_prefix: str, split_name: str, output_dir: Path):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        annotations = json.loads(zf.read(ann_member).decode("utf-8"))
        counter = Counter()

        for index, row in enumerate(annotations, start=1):
            class_id = int(row["disease_class"])
            image_id = row["image_id"]
            class_name = f"class_{class_id:02d}"
            target_class_dir = split_dir / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            member_name = f"{image_prefix}{image_id}"
            target_path = target_class_dir / image_id

            with zf.open(member_name) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)

            counter[class_name] += 1
            if index % 5000 == 0:
                print(f"[{split_name}] converted {index} images")

    return {
        "split": split_name,
        "source_zip": str(zip_path),
        "images": sum(counter.values()),
        "classes": len(counter),
        "class_distribution": dict(sorted(counter.items())),
    }


def main():
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    train_zip = source_dir / "ai_challenger_pdr2018_trainingset_20181023.zip"
    val_zip = source_dir / "ai_challenger_pdr2018_validationset_20181023.zip"

    ensure_clean_dir(output_dir, args.overwrite)

    train_summary = convert_split(
        zip_path=train_zip,
        ann_member="AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json",
        image_prefix="AgriculturalDisease_trainingset/images/",
        split_name="train",
        output_dir=output_dir,
    )
    val_summary = convert_split(
        zip_path=val_zip,
        ann_member="AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json",
        image_prefix="AgriculturalDisease_validationset/images/",
        split_name="val",
        output_dir=output_dir,
    )

    label_mapping = {
        str(class_id): {
            "id": class_id,
            "name": f"class_{class_id:02d}",
        }
        for class_id in range(61)
    }

    manifest = {
        "dataset_name": output_dir.name,
        "dataset_type": "classification",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "splits": {
            "train": train_summary,
            "val": val_summary,
        },
        "labels": label_mapping,
        "notes": [
            "Class IDs are preserved from AI Challenger PDR2018 as numeric labels.",
            "TestA/TestB were not extracted because they are unlabeled.",
        ],
    }

    write_json(output_dir / "labels.json", label_mapping)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
