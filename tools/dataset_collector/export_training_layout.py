#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SPLIT_MAP = {"train": "train", "test": "val"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an HF-ready metadata.csv image dataset into the numeric split layout expected by the trainer.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="HF-ready dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Numeric training dataset output directory")
    parser.add_argument("--metadata-csv", type=Path, default=None, help="Defaults to <input-dir>/metadata.csv")
    parser.add_argument(
        "--split-map",
        action="append",
        default=[],
        help="Map source split to target split, e.g. train=train or test=val. Defaults to train=train and test=val.",
    )
    parser.add_argument(
        "--stratified-val-ratio",
        type=float,
        default=None,
        help="Ignore source splits and create a deterministic label-stratified train/val split.",
    )
    parser.add_argument("--stratified-seed", type=int, default=888)
    parser.add_argument("--stratified-min-val-per-class", type=int, default=1)
    parser.add_argument("--copy-mode", choices=["copy", "link"], default="link")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory before exporting")
    parser.add_argument("--progress-interval", type=int, default=10000)
    return parser.parse_args()


def parse_split_map(values: List[str]) -> Dict[str, str]:
    if not values:
        return dict(DEFAULT_SPLIT_MAP)
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --split-map value {value!r}; expected source=target")
        source, target = value.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"Invalid --split-map value {value!r}; source and target are required")
        result[source] = target
    return result


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def load_labels(input_dir: Path) -> Dict[int, Dict[str, Any]]:
    labels_path = input_dir / "labels.json"
    if not labels_path.exists():
        return {}
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels: Dict[int, Dict[str, Any]] = {}
    for key, value in payload.items():
        label_id = int(value.get("id", key)) if isinstance(value, dict) else int(key)
        label_name = str(value.get("name", label_id)) if isinstance(value, dict) else str(value)
        labels[label_id] = {"id": label_id, "name": label_name, "directory": str(label_id)}
    return labels


def read_metadata(metadata_csv: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "file_name" not in fieldnames or "label" not in fieldnames:
            raise ValueError(f"metadata CSV must include file_name and label columns: {metadata_csv}")
        return [dict(row) for row in reader], fieldnames


def unique_target_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000000):
        candidate = path.with_name(f"{stem}_{index:06d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique target path for {path}")


def validate_stratified_ratio(value: Optional[float]) -> None:
    if value is None:
        return
    if value <= 0.0 or value >= 1.0:
        raise ValueError("--stratified-val-ratio must be between 0 and 1")


def assign_split_mapped_rows(rows: Iterable[Dict[str, str]], split_map: Dict[str, str]) -> Tuple[List[Dict[str, str]], Counter]:
    assigned: List[Dict[str, str]] = []
    skipped_counts: Counter = Counter()
    for row in rows:
        source_split = row.get("split", "")
        if source_split not in split_map:
            skipped_counts[f"split:{source_split}"] += 1
            continue
        output_row = dict(row)
        output_row["target_split"] = split_map[source_split]
        assigned.append(output_row)
    return assigned, skipped_counts


def assign_stratified_rows(
    rows: Iterable[Dict[str, str]],
    val_ratio: float,
    seed: int,
    min_val_per_class: int,
) -> Tuple[List[Dict[str, str]], Counter]:
    groups: DefaultDict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[int(row["label"])].append(dict(row))

    rng = random.Random(seed)
    assigned: List[Dict[str, str]] = []
    skipped_counts: Counter = Counter()
    for label_id in sorted(groups):
        group = sorted(groups[label_id], key=lambda item: item["file_name"])
        rng.shuffle(group)
        if len(group) <= 1:
            val_count = 0
        else:
            val_count = max(min_val_per_class, round(len(group) * val_ratio))
            val_count = min(val_count, len(group) - 1)

        for index, row in enumerate(group):
            row["target_split"] = "val" if index < val_count else "train"
            assigned.append(row)

        if val_count == 0:
            skipped_counts["classes_without_val"] += 1

    return assigned, skipped_counts


def export_rows(
    input_dir: Path,
    output_dir: Path,
    rows: Sequence[Dict[str, str]],
    labels: Dict[int, Dict[str, Any]],
    copy_mode: str,
    progress_interval: int,
) -> Tuple[List[Dict[str, Any]], Counter, Counter, Counter, Counter]:
    rows_out: List[Dict[str, Any]] = []
    split_counts: Counter = Counter()
    label_counts: Counter = Counter()
    source_counts: Counter = Counter()
    skipped_counts: Counter = Counter()

    for index, row in enumerate(rows, start=1):
        label_id = int(row["label"])
        label_name = row.get("label_name", labels.get(label_id, {}).get("name", str(label_id)))
        labels.setdefault(label_id, {"id": label_id, "name": label_name, "directory": str(label_id)})

        source_path = input_dir / row["file_name"]
        if not source_path.exists():
            skipped_counts["missing_file"] += 1
            continue

        target_split = row["target_split"]
        target_path = unique_target_path(output_dir / target_split / str(label_id) / source_path.name)
        copy_or_link(source_path, target_path, copy_mode)

        relative_target = str(target_path.relative_to(output_dir)).replace("\\", "/")
        output_row = {key: value for key, value in row.items() if key != "target_split"}
        output_row["file_name"] = relative_target
        output_row["source_file_name"] = row["file_name"]
        output_row["split"] = target_split
        rows_out.append(output_row)
        split_counts[target_split] += 1
        label_counts[label_id] += 1
        source_counts[row.get("source", "")] += 1

        if progress_interval and index % progress_interval == 0:
            print(f"processed={index} exported={len(rows_out)} skipped={sum(skipped_counts.values())}", flush=True)

    return rows_out, split_counts, label_counts, source_counts, skipped_counts


def write_metadata(path: Path, rows: List[Dict[str, Any]], source_fieldnames: List[str]) -> None:
    preferred = list(source_fieldnames)
    if "source_file_name" not in preferred:
        preferred.append("source_file_name")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=preferred, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_training_layout(args) -> Dict[str, Any]:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_csv = (args.metadata_csv or (input_dir / "metadata.csv")).resolve()
    split_map = parse_split_map(args.split_map)
    stratified_val_ratio = getattr(args, "stratified_val_ratio", None)
    stratified_seed = getattr(args, "stratified_seed", 888)
    stratified_min_val_per_class = getattr(args, "stratified_min_val_per_class", 1)
    validate_stratified_ratio(stratified_val_ratio)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if input_dir == output_dir:
        raise ValueError("--output-dir must be different from --input-dir")

    reset_output_dir(output_dir, args.overwrite)
    labels = load_labels(input_dir)
    metadata_rows, source_fieldnames = read_metadata(metadata_csv)

    if stratified_val_ratio is None:
        assigned_rows, assignment_skips = assign_split_mapped_rows(metadata_rows, split_map)
        split_strategy = {"type": "source_split_map", "split_map": split_map}
    else:
        assigned_rows, assignment_skips = assign_stratified_rows(
            metadata_rows,
            val_ratio=stratified_val_ratio,
            seed=stratified_seed,
            min_val_per_class=stratified_min_val_per_class,
        )
        split_strategy = {
            "type": "label_stratified",
            "val_ratio": stratified_val_ratio,
            "seed": stratified_seed,
            "min_val_per_class": stratified_min_val_per_class,
        }

    rows_out, split_counts, label_counts, source_counts, export_skips = export_rows(
        input_dir=input_dir,
        output_dir=output_dir,
        rows=assigned_rows,
        labels=labels,
        copy_mode=args.copy_mode,
        progress_interval=args.progress_interval,
    )
    skipped_counts = assignment_skips + export_skips

    labels_out = {
        str(label_id): labels[label_id]
        for label_id in sorted(labels)
        if label_counts.get(label_id, 0) > 0
    }
    manifest = {
        "dataset_name": output_dir.name,
        "dataset_type": "numeric_classification_layout",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "metadata_csv": str(metadata_csv),
        "output_dir": str(output_dir),
        "copy_mode": args.copy_mode,
        "split_strategy": split_strategy,
        "images": len(rows_out),
        "classes": len(labels_out),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
    }

    (output_dir / "labels.json").write_text(json.dumps(labels_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metadata(output_dir / "metadata.csv", rows_out, source_fieldnames)
    return manifest


def main() -> None:
    manifest = export_training_layout(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
