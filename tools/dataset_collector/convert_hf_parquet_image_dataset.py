#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import io
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pyarrow.parquet as pq
from PIL import Image


SPLIT_ALIASES = {
    "validation": "val",
    "valid": "val",
    "val": "val",
    "train": "train",
    "test": "test",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HF parquet image dataset files into numeric classification folders")
    parser.add_argument("--source-dir", type=Path, required=True, help="Dataset snapshot directory containing data/*.parquet")
    parser.add_argument("--output-dir", type=Path, required=True, help="Converted classification output directory")
    parser.add_argument("--dataset-name", default="", help="Dataset name for manifest")
    parser.add_argument("--url", default="", help="Source URL for manifest")
    parser.add_argument("--license", default="", help="Source license for manifest")
    parser.add_argument("--citation", default="", help="Source citation for manifest")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory before conversion")
    return parser.parse_args()


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "image"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parquet_files(source_dir: Path) -> List[Path]:
    return sorted((source_dir / "data").glob("*.parquet"))


def split_name_from_file(path: Path) -> str:
    raw = path.stem.split("-")[0]
    return SPLIT_ALIASES.get(raw, raw)


def load_label_names(readme_path: Path) -> Dict[int, str]:
    if not readme_path.exists():
        return {}

    labels: Dict[int, str] = {}
    in_names = False
    for line in readme_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped == "names:":
            in_names = True
            continue
        if in_names and stripped.startswith("- name:") and "label" not in stripped:
            break
        if in_names:
            match = re.match(r"'?(\d+)'?:\s*(.+)$", stripped)
            if match:
                labels[int(match.group(1))] = match.group(2).strip().strip("'\"")
    return labels


def image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw:
            return raw
        path = value.get("path")
        if path:
            return Path(path).read_bytes()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise ValueError(f"Unsupported image cell type: {type(value)!r}")


def row_dicts(table) -> Iterable[Dict[str, Any]]:
    columns = table.column_names
    arrays = {name: table[name].to_pylist() for name in columns}
    for index in range(table.num_rows):
        yield {name: arrays[name][index] for name in columns}


def convert(args) -> Dict[str, Any]:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = parquet_files(source_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {source_dir / 'data'}")

    reset_output_dir(output_dir, args.overwrite)
    label_names = load_label_names(source_dir / "README.md")
    distribution: Dict[str, Counter[str]] = {}
    split_counts: Counter[str] = Counter()
    skipped = 0

    for parquet_path in files:
        split = split_name_from_file(parquet_path)
        table = pq.read_table(parquet_path)
        for row_index, row in enumerate(row_dicts(table), start=1):
            try:
                label_id = int(row["label"])
                label_name = label_names.get(label_id, str(label_id))
                target_dir = output_dir / split / str(label_id)
                target_dir.mkdir(parents=True, exist_ok=True)

                raw = image_bytes(row["image"])
                suffix = ".jpg"
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        detected_format = (image.format or "").lower()
                        if detected_format in {"jpeg", "jpg"}:
                            suffix = ".jpg"
                        elif detected_format:
                            suffix = f".{detected_format}"
                except Exception:
                    pass

                split_counts[split] += 1
                filename = f"{split_counts[split]:08d}_{safe_name(label_name)}{suffix}"
                (target_dir / filename).write_bytes(raw)
                distribution.setdefault(split, Counter())[label_name] += 1
            except Exception:
                skipped += 1

        print(f"[{parquet_path.name}] converted {table.num_rows} rows")

    labels = {
        str(label_id): {
            "id": label_id,
            "name": label_names.get(label_id, str(label_id)),
            "directory": str(label_id),
        }
        for label_id in sorted(label_names or {int(path.name) for path in (output_dir / "train").iterdir() if path.is_dir()})
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
            split: {
                "images": sum(counter.values()),
                "classes": len(counter),
                "class_distribution": dict(sorted(counter.items())),
            }
            for split, counter in sorted(distribution.items())
        },
        "labels": labels,
        "skipped_rows": skipped,
    }
    write_json(output_dir / "labels.json", labels)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main():
    args = parse_args()
    print(json.dumps(convert(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
