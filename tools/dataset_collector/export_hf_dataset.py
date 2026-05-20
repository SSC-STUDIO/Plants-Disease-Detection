#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_REPO = "SSC-STUDIO/plant-disease-open-dataset"


def parse_args():
    parser = argparse.ArgumentParser(description="Export redistributable bundle sources into a Hugging Face dataset layout")
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Bundle directory containing manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for the HF-ready dataset")
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO, help="Target Hugging Face dataset repo id")
    parser.add_argument("--overwrite", action="store_true", help="Delete output dir before exporting")
    parser.add_argument("--copy-mode", choices=["copy", "link"], default="copy", help="Copy files or create hard links")
    return parser.parse_args()


def load_manifest(bundle_dir: Path) -> Dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for dirpath, dirs, filenames in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in {".git", "__pycache__"})
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if is_image(path):
                yield path


def image_class_name(image_path: Path, split_root: Path) -> Optional[str]:
    try:
        rel = image_path.relative_to(split_root)
    except ValueError:
        return None
    if len(rel.parts) < 2:
        return None
    return rel.parts[0]


def load_source_labels(source_root: Path) -> Tuple[Dict[str, str], List[str]]:
    labels_path = source_root / "labels.json"
    if not labels_path.exists():
        return {}, []

    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, []

    directory_to_name: Dict[str, str] = {}
    ordered_labels: List[Tuple[int, str]] = []

    for key, value in payload.items():
        if isinstance(value, dict):
            label_id = int(value.get("id", key)) if str(value.get("id", key)).isdigit() else len(ordered_labels)
            label_name = str(value.get("name") or value.get("label") or key)
            directory = str(value.get("directory") or value.get("dir") or value.get("id") or key)
        else:
            label_id = int(key) if str(key).isdigit() else len(ordered_labels)
            label_name = str(value)
            directory = str(key)

        directory_to_name[directory] = label_name
        directory_to_name[str(label_id)] = label_name
        directory_to_name[str(key)] = label_name
        ordered_labels.append((label_id, label_name))

    deduped_order: List[str] = []
    seen = set()
    for _, label_name in sorted(ordered_labels, key=lambda item: item[0]):
        if label_name not in seen:
            deduped_order.append(label_name)
            seen.add(label_name)

    return directory_to_name, deduped_order


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._") or "unknown"


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def export_source(
    source: Dict[str, Any],
    output_dir: Path,
    copy_mode: str,
    label_to_id: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], Counter]:
    rows: List[Dict[str, Any]] = []
    class_counts: Counter = Counter()
    source_root = Path(source["source_path"])
    directory_to_label, ordered_labels = load_source_labels(source_root)

    for label_name in ordered_labels:
        if label_name not in label_to_id:
            label_to_id[label_name] = len(label_to_id)

    for split_name, split_info in source.get("splits", {}).items():
        split_root = Path(split_info["path"])
        if not split_root.exists():
            continue

        for image_path in iter_images(split_root):
            class_dir_name = image_class_name(image_path, split_root)
            if class_dir_name is None:
                continue
            label_name = directory_to_label.get(class_dir_name, class_dir_name)
            if label_name not in label_to_id:
                label_to_id[label_name] = len(label_to_id)

            label_id = label_to_id[label_name]
            filename = f"{safe_name(source['name'])}_{safe_name(label_name)}_{image_path.stem}{image_path.suffix.lower()}"
            relative_target = Path("data") / split_name / f"{label_id:04d}_{safe_name(label_name)}" / filename
            target_path = output_dir / relative_target
            copy_or_link(image_path, target_path, copy_mode)

            try:
                original_relative = str(image_path.relative_to(source_root))
            except ValueError:
                original_relative = str(image_path)

            row = {
                "file_name": str(relative_target).replace("\\", "/"),
                "split": split_name,
                "label": label_id,
                "label_name": label_name,
                "source": source["name"],
                "source_url": source.get("url", ""),
                "license": source.get("license", ""),
                "original_path": original_relative.replace("\\", "/"),
            }
            rows.append(row)
            class_counts[(split_name, label_name)] += 1

    return rows, class_counts


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_name", "split", "label", "label_name", "source", "source_url", "license", "original_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_card(
    output_dir: Path,
    dataset_repo: str,
    manifest: Dict[str, Any],
    exported_sources: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
) -> None:
    split_counts = Counter(row["split"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    licenses = sorted({source.get("license", "") for source in exported_sources if source.get("license")})

    lines = [
        "---",
        "license: cc-by-sa-3.0",
        "task_categories:",
        "- image-classification",
        "pretty_name: Plant Disease Open Dataset",
        "tags:",
        "- agriculture",
        "- plant-disease",
        "- computer-vision",
        "- education",
        "---",
        "",
        "# Plant Disease Open Dataset",
        "",
        "This dataset export contains only sources marked `redistributable=true` in the project source manifest.",
        "Restricted or unclear-license data is intentionally excluded and remains documented in the bundle manifest.",
        "",
        f"- Target dataset repo: `{dataset_repo}`",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Source bundle: `{manifest.get('target_dir')}`",
        f"- Total images: `{len(rows)}`",
        f"- Classes: `{len(labels)}`",
        f"- Licenses: {', '.join(licenses) if licenses else 'See provenance.csv'}",
        "",
        "## Splits",
        "",
    ]

    for split, count in sorted(split_counts.items()):
        lines.append(f"- `{split}`: {count} images")

    lines.extend(["", "## Sources", ""])
    for source in exported_sources:
        lines.extend([
            f"### {source['name']}",
            f"- URL: {source.get('url', '')}",
            f"- License: {source.get('license', '')}",
            f"- Citation: {source.get('citation', '')}",
            f"- Images exported: {source_counts[source['name']]}",
            f"- Notes: {source.get('notes', '')}",
            "",
        ])

    lines.extend([
        "## Intended Use",
        "",
        "Use this dataset for plant-disease image classification education, baseline training, and reproducible experiments.",
        "It should not be used as the only evidence for agricultural decisions because field conditions, camera quality, geography, crop variety, and disease stage can shift model behavior.",
        "",
        "## Files",
        "",
        "- `metadata.csv`: file-level labels and source provenance.",
        "- `labels.json`: numeric label IDs and class names.",
        "- `provenance.json`: source-level license and citation metadata.",
        "",
    ])

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest = load_manifest(bundle_dir)
    reset_output_dir(output_dir, args.overwrite)

    redistributable_sources = [
        source for source in manifest.get("sources", [])
        if source.get("redistributable") and source.get("exists")
    ]
    if not redistributable_sources:
        raise ValueError("No redistributable sources found in bundle manifest")

    label_to_id: Dict[str, int] = {}
    all_rows: List[Dict[str, Any]] = []
    distribution: Dict[str, Dict[str, int]] = {}

    for source in redistributable_sources:
        rows, class_counts = export_source(source, output_dir=output_dir, copy_mode=args.copy_mode, label_to_id=label_to_id)
        all_rows.extend(rows)
        for (split, class_name), count in class_counts.items():
            distribution.setdefault(split, {})[class_name] = count

    labels = {
        str(label_id): {"id": label_id, "name": label_name}
        for label_name, label_id in sorted(label_to_id.items(), key=lambda item: item[1])
    }
    provenance = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_repo": args.dataset_repo,
        "source_bundle": manifest.get("target_dir"),
        "sources": redistributable_sources,
        "splits": distribution,
    }

    write_csv(output_dir / "metadata.csv", all_rows)
    write_json(output_dir / "labels.json", labels)
    write_json(output_dir / "provenance.json", provenance)
    write_dataset_card(output_dir, args.dataset_repo, manifest, redistributable_sources, labels, all_rows)

    summary = {
        "output_dir": str(output_dir),
        "dataset_repo": args.dataset_repo,
        "images": len(all_rows),
        "classes": len(labels),
        "sources": [source["name"] for source in redistributable_sources],
    }
    write_json(output_dir / "export_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
