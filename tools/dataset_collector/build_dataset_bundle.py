#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def datasets_root() -> Path:
    return repo_root().parents[2] / "Datasets"


def default_manifest_path() -> Path:
    return repo_root() / "data" / "sources.yaml"


def default_bundle_sources() -> List[Dict[str, Any]]:
    root = repo_root()
    data_root = root / "data"
    external_root = datasets_root() / "DataScience"

    return [
        {
            "name": "project-selfmade-train",
            "url": "https://github.com/SSC-STUDIO/Plants-Disease-Detection",
            "license": "Project-owned images; verify consent before public redistribution",
            "redistributable": False,
            "citation": "SSC-STUDIO Plants Disease Detection project data",
            "local_path": str(data_root / "train"),
            "task": "classification",
            "splits": {"train": "."},
            "class_mapping": "directory_name",
            "notes": "Project-local self-made training split",
        },
        {
            "name": "project-selfmade-val",
            "url": "https://github.com/SSC-STUDIO/Plants-Disease-Detection",
            "license": "Project-owned images; verify consent before public redistribution",
            "redistributable": False,
            "citation": "SSC-STUDIO Plants Disease Detection project data",
            "local_path": str(data_root / "val"),
            "task": "classification",
            "splits": {"val": "."},
            "class_mapping": "directory_name",
            "notes": "Project-local self-made validation split",
        },
        {
            "name": "project-selfmade-test",
            "url": "https://github.com/SSC-STUDIO/Plants-Disease-Detection",
            "license": "Project-owned images; verify consent before public redistribution",
            "redistributable": False,
            "citation": "SSC-STUDIO Plants Disease Detection project data",
            "local_path": str(data_root / "test"),
            "task": "unlabeled_or_private_test",
            "splits": {"test": "."},
            "class_mapping": "none",
            "notes": "Project-local test split",
        },
        {
            "name": "PlantVillage-Dataset",
            "url": "https://github.com/spMohanty/PlantVillage-Dataset",
            "license": "CC BY-SA 3.0",
            "redistributable": True,
            "citation": "PlantVillage Dataset, Hughes and Salathe, 2015",
            "local_path": str(external_root / "PlantVillage-Dataset"),
            "task": "classification",
            "splits": {"train": "."},
            "class_mapping": "directory_name",
            "notes": "Controlled leaf-classification dataset",
        },
        {
            "name": "PlantDoc-Object-Detection-Dataset-Linux",
            "url": "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset",
            "license": "Public GitHub dataset; verify upstream license before redistributing derived crops",
            "redistributable": False,
            "citation": "PlantDoc: A Dataset for Visual Plant Disease Detection",
            "local_path": str(external_root / "PlantDoc-Object-Detection-Dataset(Linux)"),
            "task": "detection",
            "splits": {"train": "TRAIN", "test": "TEST"},
            "class_mapping": "annotation_name",
            "notes": "In-the-wild detection dataset",
        },
        {
            "name": "AI-Challenger-PDR2018",
            "url": "https://challenger.ai/competition/pdr2018",
            "license": "Challenge dataset terms; research use only unless redistribution is explicitly permitted",
            "redistributable": False,
            "citation": "AI Challenger 2018 Plant Disease Recognition",
            "local_path": str(external_root / "Plant-Disese-Dataset"),
            "task": "classification_archive",
            "splits": {
                "train": "ai_challenger_pdr2018_trainingset_20181023.zip",
                "val": "ai_challenger_pdr2018_validationset_20181023.zip",
            },
            "class_mapping": "ai_challenger_numeric_id",
            "notes": "AI Challenger crop disease recognition archives",
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a reusable plant-disease dataset bundle from local assets")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=default_manifest_path(),
        help="YAML source manifest. Falls back to built-in defaults if missing.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=datasets_root() / "PlantDisease-Open-Bundle",
        help="Target bundle directory",
    )
    parser.add_argument(
        "--no-validate-images",
        action="store_true",
        help="Skip image open/verify checks for faster manifest generation",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip image SHA-256 hashing and duplicate counts for faster manifest generation",
    )
    return parser.parse_args()


def resolve_source_path(path_value: str, manifest_dir: Path) -> Path:
    raw = Path(os.path.expandvars(os.path.expanduser(path_value)))
    if raw.is_absolute():
        return raw

    candidates = [
        (manifest_dir / raw).resolve(),
        (repo_root() / raw).resolve(),
        (repo_root().parents[1] / raw).resolve(),
        (repo_root().parents[2] / raw).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_sources(source_manifest: Path) -> Dict[str, Any]:
    if source_manifest.exists():
        with open(source_manifest, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        payload.setdefault("bundle_name", "PlantDisease-Open-Bundle")
        payload.setdefault("sources", [])
        payload["_manifest_path"] = str(source_manifest.resolve())
        payload["_manifest_dir"] = source_manifest.resolve().parent
        return payload

    return {
        "bundle_name": "PlantDisease-Open-Bundle",
        "sources": default_bundle_sources(),
        "_manifest_path": None,
        "_manifest_dir": repo_root(),
    }


def is_junk_path(path: Path) -> bool:
    return any(part in {".git", "__pycache__"} for part in path.parts)


def iter_files(directory: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(directory):
        current_root = Path(root)
        dirs[:] = [name for name in dirs if name not in {".git", "__pycache__"}]
        if is_junk_path(current_root):
            continue
        for file_name in files:
            yield current_root / file_name


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def split_source_path(source_root: Path, split_path: str) -> Path:
    if split_path in ("", "."):
        return source_root
    return source_root / split_path


def class_name_for(path: Path, split_root: Path) -> Optional[str]:
    try:
        relative = path.relative_to(split_root)
    except ValueError:
        return None
    if len(relative.parts) < 2:
        return None
    return relative.parts[0]


def scan_split(split_root: Path, validate_images: bool, hash_images: bool) -> Dict[str, Any]:
    class_counts: Counter[str] = Counter()
    image_count = 0
    file_count = 0
    total_bytes = 0
    invalid_images: List[str] = []
    hashes: Counter[str] = Counter()

    if not split_root.exists():
        return {
            "path": str(split_root),
            "exists": False,
            "file_count": 0,
            "image_count": 0,
            "logical_size_bytes": 0,
            "logical_size_gb": 0,
            "class_count": 0,
            "class_distribution": {},
            "duplicate_images": 0,
            "invalid_images": [],
        }

    for file_path in iter_files(split_root):
        file_count += 1
        total_bytes += file_path.stat().st_size
        if not is_image_file(file_path):
            continue

        image_count += 1
        class_name = class_name_for(file_path, split_root)
        if class_name:
            class_counts[class_name] += 1

        if validate_images and not validate_image(file_path):
            invalid_images.append(str(file_path))
            continue

        if hash_images:
            hashes[file_sha256(file_path)] += 1

    duplicate_images = sum(count - 1 for count in hashes.values() if count > 1) if hash_images else None
    return {
        "path": str(split_root),
        "exists": True,
        "file_count": file_count,
        "image_count": image_count,
        "logical_size_bytes": total_bytes,
        "logical_size_gb": bytes_to_gb(total_bytes),
        "class_count": len(class_counts),
        "class_distribution": dict(sorted(class_counts.items())),
        "duplicate_images": duplicate_images,
        "duplicate_scan": "sha256" if hash_images else "skipped",
        "invalid_images": invalid_images[:100],
        "invalid_image_count": len(invalid_images),
    }


def ensure_junction(link_path: Path, target_path: Path):
    if link_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def bytes_to_gb(num_bytes: int) -> float:
    return round(num_bytes / (1024 ** 3), 3)


def source_summary(
    source: Dict[str, Any],
    manifest_dir: Path,
    sources_dir: Path,
    validate_images: bool,
    hash_images: bool,
) -> Dict[str, Any]:
    source_path = resolve_source_path(str(source["local_path"]), manifest_dir)
    link_path = sources_dir / source["name"]
    split_summaries = {}
    total_files = 0
    total_images = 0
    total_bytes = 0
    duplicate_images = 0
    invalid_image_count = 0

    if source_path.exists() and source_path.is_dir():
        ensure_junction(link_path, source_path)

    for split_name, split_path in (source.get("splits") or {}).items():
        actual_split_path = split_source_path(source_path, str(split_path))
        summary = scan_split(actual_split_path, validate_images=validate_images, hash_images=hash_images)
        split_summaries[split_name] = summary
        total_files += summary["file_count"]
        total_images += summary["image_count"]
        total_bytes += summary["logical_size_bytes"]
        duplicate_images += summary["duplicate_images"] or 0
        invalid_image_count += summary.get("invalid_image_count", 0)

    return {
        "name": source["name"],
        "url": source.get("url", ""),
        "license": source.get("license", ""),
        "redistributable": bool(source.get("redistributable", False)),
        "citation": source.get("citation", ""),
        "task": source.get("task", ""),
        "class_mapping": source.get("class_mapping", ""),
        "notes": source.get("notes", ""),
        "source_path": str(source_path),
        "linked_path": str(link_path) if source_path.exists() and source_path.is_dir() else None,
        "exists": source_path.exists(),
        "file_count": total_files,
        "image_count": total_images,
        "logical_size_bytes": total_bytes,
        "logical_size_gb": bytes_to_gb(total_bytes),
        "duplicate_images": duplicate_images,
        "invalid_image_count": invalid_image_count,
        "splits": split_summaries,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(target_dir: Path, manifest: dict):
    lines = [
        f"# {manifest['bundle_name']}",
        "",
        "This bundle is assembled from local project data and public plant-disease sources.",
        "It keeps provenance and redistribution flags explicit so public exports do not accidentally include restricted data.",
        "",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Bundle path: `{manifest['target_dir']}`",
        f"- Source manifest: `{manifest.get('source_manifest') or 'built-in defaults'}`",
        f"- Logical total size: `{manifest['logical_total_gb']} GB`",
        f"- Logical total files: `{manifest['logical_total_files']}`",
        f"- Total images: `{manifest['logical_total_images']}`",
        "",
        "## Included Sources",
        "",
    ]

    for source in manifest["sources"]:
        lines.extend([
            f"### {source['name']}",
            f"- Task type: `{source['task']}`",
            f"- Redistributable: `{source['redistributable']}`",
            f"- License: {source['license']}",
            f"- URL: {source['url']}",
            f"- Source path: `{source['source_path']}`",
            f"- Linked path: `{source['linked_path']}`",
            f"- Files / images: `{source['file_count']}` / `{source['image_count']}`",
            f"- Logical size: `{source['logical_size_gb']} GB`",
            f"- Duplicate images: `{source['duplicate_images']}`",
            f"- Invalid images: `{source['invalid_image_count']}`",
            f"- Notes: {source['notes']}",
            "",
        ])

    lines.extend([
        "## Public Export Policy",
        "",
        "- Export only sources where `redistributable` is `true`.",
        "- Keep restricted or unclear-license data as local research inputs with citation and download instructions.",
        "- Rebuild this bundle before each training release so dataset and model cards stay in sync.",
        "",
    ])

    (target_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    target_dir = args.target_dir.resolve()
    sources_dir = target_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    source_payload = load_sources(args.source_manifest)
    manifest_dir = Path(source_payload["_manifest_dir"])

    manifest_sources = []
    total_files = 0
    total_images = 0
    total_bytes = 0
    duplicate_images = 0
    invalid_image_count = 0

    for source in source_payload["sources"]:
        summary = source_summary(
            source,
            manifest_dir=manifest_dir,
            sources_dir=sources_dir,
            validate_images=not args.no_validate_images,
            hash_images=not args.no_hash,
        )
        manifest_sources.append(summary)
        total_files += summary["file_count"]
        total_images += summary["image_count"]
        total_bytes += summary["logical_size_bytes"]
        duplicate_images += summary["duplicate_images"]
        invalid_image_count += summary["invalid_image_count"]

    manifest = {
        "bundle_name": source_payload.get("bundle_name", "PlantDisease-Open-Bundle"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_dir": str(target_dir),
        "source_manifest": source_payload.get("_manifest_path"),
        "logical_total_files": total_files,
        "logical_total_images": total_images,
        "logical_total_bytes": total_bytes,
        "logical_total_gb": bytes_to_gb(total_bytes),
        "duplicate_images": duplicate_images,
        "invalid_image_count": invalid_image_count,
        "sources": manifest_sources,
    }

    write_json(target_dir / "manifest.json", manifest)
    write_readme(target_dir, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
