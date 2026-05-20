#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_COMPANION_FILES = ("labels.json", "provenance.json")


@dataclass(frozen=True)
class FilterOptions:
    input_dir: Path
    output_dir: Path
    metadata_csv: Optional[Path] = None
    copy_mode: str = "link"
    overwrite: bool = False
    min_file_size: int = 1024
    min_dimension: int = 96
    min_stddev: float = 3.0
    min_entropy: float = 1.0
    near_duplicate_hamming: int = 0
    near_bucket_bits: int = 16
    label_column: str = "label_name"
    max_images: Optional[int] = None
    progress_interval: int = 5000


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative_path: str
    metadata: Dict[str, str]
    label_key: str


@dataclass
class ImageMetrics:
    file_size: int = 0
    width: int = 0
    height: int = 0
    stddev: float = 0.0
    entropy: float = 0.0
    sha256: str = ""
    ahash: int = 0


def parse_args() -> FilterOptions:
    parser = argparse.ArgumentParser(
        description="Create an auditable filtered copy of an image-classification dataset export.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Input dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Filtered dataset output directory")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Metadata CSV with file_name and label columns. Defaults to <input-dir>/metadata.csv when present.",
    )
    parser.add_argument("--copy-mode", choices=["copy", "link"], default="link", help="Copy files or create hard links")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists")
    parser.add_argument("--min-file-size", type=int, default=1024, help="Reject files smaller than this many bytes")
    parser.add_argument("--min-dimension", type=int, default=96, help="Reject images whose shorter side is smaller")
    parser.add_argument("--min-stddev", type=float, default=3.0, help="Reject very low-contrast grayscale images")
    parser.add_argument("--min-entropy", type=float, default=1.0, help="Reject very low-information grayscale images")
    parser.add_argument(
        "--near-duplicate-hamming",
        type=int,
        default=0,
        help="aHash Hamming distance for near-duplicate rejection. 0 disables near-duplicate filtering.",
    )
    parser.add_argument(
        "--near-bucket-bits",
        type=int,
        default=16,
        help="Number of high-order aHash bits used for near-duplicate candidate buckets.",
    )
    parser.add_argument("--label-column", default="label_name", help="Metadata column used for same-label dedupe")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--progress-interval", type=int, default=5000, help="Print progress every N images")
    args = parser.parse_args()
    return FilterOptions(**vars(args))


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_relative(path_value: str) -> str:
    return str(Path(path_value)).replace("\\", "/")


def safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def ensure_output_dir(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    input_dir = safe_resolve(input_dir)
    output_dir = safe_resolve(output_dir)
    if input_dir == output_dir:
        raise ValueError("--output-dir must be different from --input-dir")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def iter_records_from_metadata(
    input_dir: Path,
    metadata_csv: Path,
    label_column: str,
    max_images: Optional[int],
) -> Tuple[List[ImageRecord], List[str]]:
    records: List[ImageRecord] = []
    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "file_name" not in fieldnames:
            raise ValueError(f"metadata CSV must include a file_name column: {metadata_csv}")
        for index, row in enumerate(reader):
            if max_images is not None and index >= max_images:
                break
            relative_path = normalize_relative(row["file_name"])
            label_key = row.get(label_column) or row.get("label") or Path(relative_path).parent.name
            records.append(
                ImageRecord(
                    path=input_dir / relative_path,
                    relative_path=relative_path,
                    metadata={key: value for key, value in row.items()},
                    label_key=str(label_key),
                ),
            )
    return records, fieldnames


def iter_records_from_directory(input_dir: Path, max_images: Optional[int]) -> Tuple[List[ImageRecord], List[str]]:
    records: List[ImageRecord] = []
    for path in sorted(input_dir.rglob("*")):
        if max_images is not None and len(records) >= max_images:
            break
        if not path.is_file() or not is_image_file(path):
            continue
        relative_path = normalize_relative(str(path.relative_to(input_dir)))
        label_key = path.parent.name
        records.append(
            ImageRecord(
                path=path,
                relative_path=relative_path,
                metadata={
                    "file_name": relative_path,
                    "split": path.parent.parent.name if path.parent.parent != input_dir else "",
                    "label": "",
                    "label_name": label_key,
                    "source": "",
                    "source_url": "",
                    "license": "",
                    "original_path": relative_path,
                },
                label_key=label_key,
            ),
        )
    return records, ["file_name", "split", "label", "label_name", "source", "source_url", "license", "original_path"]


def load_records(options: FilterOptions) -> Tuple[List[ImageRecord], List[str], Optional[Path]]:
    metadata_csv = options.metadata_csv
    default_metadata = options.input_dir / "metadata.csv"
    if metadata_csv is None and default_metadata.exists():
        metadata_csv = default_metadata

    if metadata_csv is not None:
        return (*iter_records_from_metadata(options.input_dir, metadata_csv, options.label_column, options.max_images), metadata_csv)
    records, fieldnames = iter_records_from_directory(options.input_dir, options.max_images)
    return records, fieldnames, None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_entropy(gray: Image.Image) -> float:
    histogram = gray.histogram()
    total = sum(histogram)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in histogram:
        if count:
            probability = count / total
            entropy -= probability * math.log2(probability)
    return entropy


def average_hash(gray: Image.Image) -> int:
    small = gray.resize((8, 8), Image.Resampling.LANCZOS)
    values = list(small.getdata())
    mean = sum(values) / len(values)
    result = 0
    for value in values:
        result = (result << 1) | int(value >= mean)
    return result


def analyze_image(path: Path) -> ImageMetrics:
    metrics = ImageMetrics(file_size=path.stat().st_size, sha256=file_sha256(path))
    with Image.open(path) as image:
        image.load()
        image = ImageOps.exif_transpose(image)
        metrics.width, metrics.height = image.size
        gray = ImageOps.grayscale(image)
        small = gray.resize((64, 64), Image.Resampling.LANCZOS)
        values = list(small.getdata())
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        metrics.stddev = math.sqrt(variance)
        metrics.entropy = image_entropy(small)
        metrics.ahash = average_hash(gray)
    return metrics


def quality_rejection_reason(metrics: ImageMetrics, options: FilterOptions) -> Optional[str]:
    if metrics.file_size < options.min_file_size:
        return "file_too_small"
    if min(metrics.width, metrics.height) < options.min_dimension:
        return "dimensions_too_small"
    if metrics.stddev < options.min_stddev:
        return "low_stddev"
    if metrics.entropy < options.min_entropy:
        return "low_entropy"
    return None


def build_hamming_masks(bits: int, max_distance: int) -> List[int]:
    if max_distance <= 0:
        return [0]
    positions = range(bits)
    masks = [0]
    for distance in range(1, max_distance + 1):
        for combo in combinations(positions, distance):
            mask = 0
            for bit_index in combo:
                mask |= 1 << bit_index
            masks.append(mask)
    return masks


def hash_prefix(value: int, bits: int) -> int:
    if bits <= 0:
        return 0
    return value >> (64 - bits)


def find_near_duplicate(
    label_key: str,
    image_hash: int,
    seen_hashes: DefaultDict[Tuple[str, int], List[Tuple[int, str]]],
    hamming_threshold: int,
    bucket_bits: int,
    prefix_masks: Sequence[int],
) -> Optional[str]:
    if hamming_threshold <= 0:
        return None
    prefix = hash_prefix(image_hash, bucket_bits)
    for mask in prefix_masks:
        candidate_prefix = prefix ^ mask
        for kept_hash, kept_relative_path in seen_hashes.get((label_key, candidate_prefix), []):
            if (image_hash ^ kept_hash).bit_count() <= hamming_threshold:
                return kept_relative_path
    return None


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


def copy_companion_files(input_dir: Path, output_dir: Path) -> None:
    for filename in DEFAULT_COMPANION_FILES:
        source = input_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_rejections(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "file_name",
        "label_key",
        "reason",
        "duplicate_of",
        "file_size",
        "width",
        "height",
        "stddev",
        "entropy",
        "sha256",
        "ahash",
        "source",
        "original_path",
    ]
    write_csv(path, rows, fieldnames)


def append_filter_card(output_dir: Path, report: Dict[str, Any]) -> None:
    section = [
        "",
        "## Quality Filter",
        "",
        f"- Filtered export generated: `{report['generated_at']}`",
        f"- Input images checked: `{report['input_images']}`",
        f"- Images kept: `{report['kept_images']}`",
        f"- Images rejected: `{report['rejected_images']}`",
        f"- Exact duplicates removed: `{report['rejection_counts'].get('exact_duplicate', 0)}`",
        f"- Near duplicates removed: `{report['rejection_counts'].get('near_duplicate', 0)}`",
        f"- Corrupt or unreadable files removed: `{report['rejection_counts'].get('invalid_image', 0)}`",
        "",
        "Filter artifacts:",
        "",
        "- `filter_report.json`: aggregate counts and thresholds.",
        "- `rejections.csv`: one row per excluded file with reason and duplicate source when applicable.",
        "- `metadata.csv`: filtered rows only.",
        "",
    ]
    for filename in ("README.md", "dataset_card.md"):
        path = output_dir / filename
        existing = path.read_text(encoding="utf-8") if path.exists() else "# Filtered Plant Disease Dataset\n"
        path.write_text(existing.rstrip() + "\n" + "\n".join(section), encoding="utf-8")


def build_rejection_row(
    record: ImageRecord,
    reason: str,
    metrics: Optional[ImageMetrics] = None,
    duplicate_of: str = "",
) -> Dict[str, Any]:
    metrics = metrics or ImageMetrics()
    return {
        "file_name": record.relative_path,
        "label_key": record.label_key,
        "reason": reason,
        "duplicate_of": duplicate_of,
        "file_size": metrics.file_size,
        "width": metrics.width,
        "height": metrics.height,
        "stddev": round(metrics.stddev, 4),
        "entropy": round(metrics.entropy, 4),
        "sha256": metrics.sha256,
        "ahash": f"{metrics.ahash:016x}" if metrics.ahash else "",
        "source": record.metadata.get("source", ""),
        "original_path": record.metadata.get("original_path", ""),
    }


def run_filter(options: FilterOptions) -> Dict[str, Any]:
    input_dir = safe_resolve(options.input_dir)
    output_dir = safe_resolve(options.output_dir)
    effective_options = FilterOptions(**{**asdict(options), "input_dir": input_dir, "output_dir": output_dir})

    ensure_output_dir(input_dir, output_dir, options.overwrite)
    records, metadata_fieldnames, metadata_csv = load_records(effective_options)
    copy_companion_files(input_dir, output_dir)

    seen_sha: Dict[str, str] = {}
    seen_hashes: DefaultDict[Tuple[str, int], List[Tuple[int, str]]] = defaultdict(list)
    prefix_masks = build_hamming_masks(options.near_bucket_bits, options.near_duplicate_hamming)
    kept_rows: List[Dict[str, Any]] = []
    kept_audit_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    rejection_counts: Counter = Counter()
    split_counts: Counter = Counter()
    source_counts: Counter = Counter()
    label_counts: Counter = Counter()

    for index, record in enumerate(records, start=1):
        if not record.path.exists():
            reason = "missing_file"
            rejected_rows.append(build_rejection_row(record, reason))
            rejection_counts[reason] += 1
            continue

        try:
            metrics = analyze_image(record.path)
        except (OSError, UnidentifiedImageError, ValueError):
            reason = "invalid_image"
            rejected_rows.append(build_rejection_row(record, reason))
            rejection_counts[reason] += 1
            continue

        reason = quality_rejection_reason(metrics, effective_options)
        duplicate_of = ""
        if reason is None and metrics.sha256 in seen_sha:
            reason = "exact_duplicate"
            duplicate_of = seen_sha[metrics.sha256]
        if reason is None:
            duplicate_of = find_near_duplicate(
                record.label_key,
                metrics.ahash,
                seen_hashes,
                effective_options.near_duplicate_hamming,
                effective_options.near_bucket_bits,
                prefix_masks,
            )
            if duplicate_of:
                reason = "near_duplicate"

        if reason is not None:
            rejected_rows.append(build_rejection_row(record, reason, metrics, duplicate_of))
            rejection_counts[reason] += 1
        else:
            seen_sha[metrics.sha256] = record.relative_path
            prefix = hash_prefix(metrics.ahash, effective_options.near_bucket_bits)
            seen_hashes[(record.label_key, prefix)].append((metrics.ahash, record.relative_path))
            copy_or_link(record.path, output_dir / record.relative_path, effective_options.copy_mode)
            kept_rows.append(record.metadata)
            kept_audit_rows.append(
                {
                    "file_name": record.relative_path,
                    "label_key": record.label_key,
                    "source": record.metadata.get("source", ""),
                    "sha256": metrics.sha256,
                    "ahash": f"{metrics.ahash:016x}",
                    "width": metrics.width,
                    "height": metrics.height,
                    "stddev": round(metrics.stddev, 4),
                    "entropy": round(metrics.entropy, 4),
                },
            )
            split_counts[record.metadata.get("split", "")] += 1
            source_counts[record.metadata.get("source", "")] += 1
            label_counts[record.metadata.get("label_name", record.label_key)] += 1

        if effective_options.progress_interval and index % effective_options.progress_interval == 0:
            print(
                f"checked={index} kept={len(kept_rows)} rejected={len(rejected_rows)} "
                f"exact_dup={rejection_counts.get('exact_duplicate', 0)} "
                f"near_dup={rejection_counts.get('near_duplicate', 0)}",
                flush=True,
            )

    if "file_name" not in metadata_fieldnames:
        metadata_fieldnames = ["file_name", *metadata_fieldnames]
    write_csv(output_dir / "metadata.csv", kept_rows, metadata_fieldnames)
    write_rejections(output_dir / "rejections.csv", rejected_rows)
    write_csv(
        output_dir / "kept_audit.csv",
        kept_audit_rows,
        ["file_name", "label_key", "source", "sha256", "ahash", "width", "height", "stddev", "entropy"],
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "metadata_csv": str(metadata_csv) if metadata_csv else None,
        "copy_mode": effective_options.copy_mode,
        "thresholds": {
            "min_file_size": effective_options.min_file_size,
            "min_dimension": effective_options.min_dimension,
            "min_stddev": effective_options.min_stddev,
            "min_entropy": effective_options.min_entropy,
            "near_duplicate_hamming": effective_options.near_duplicate_hamming,
            "near_bucket_bits": effective_options.near_bucket_bits,
            "label_column": effective_options.label_column,
        },
        "input_images": len(records),
        "kept_images": len(kept_rows),
        "rejected_images": len(rejected_rows),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "class_count": len(label_counts),
        "top_rejected_reasons": rejection_counts.most_common(),
    }
    (output_dir / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "export_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "images": len(kept_rows),
                "classes": len(label_counts),
                "sources": sorted(source for source in source_counts if source),
                "filtered_from": str(input_dir),
                "filter_report": "filter_report.json",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    append_filter_card(output_dir, report)
    return report


def main() -> None:
    report = run_filter(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
