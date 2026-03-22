#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def datasets_root() -> Path:
    return repo_root().parents[2] / "Datasets"


def default_bundle_sources():
    root = repo_root()
    data_root = root / "data"
    external_root = datasets_root() / "DataScience"

    return [
        {
            "name": "project-selfmade-train",
            "path": data_root / "train",
            "task": "classification",
            "notes": "Project-local self-made training split",
        },
        {
            "name": "project-selfmade-val",
            "path": data_root / "val",
            "task": "classification",
            "notes": "Project-local self-made validation split",
        },
        {
            "name": "project-selfmade-test",
            "path": data_root / "test",
            "task": "classification",
            "notes": "Project-local self-made test split",
        },
        {
            "name": "PlantVillage-Dataset",
            "path": external_root / "PlantVillage-Dataset",
            "task": "classification",
            "notes": "Controlled leaf-classification dataset",
        },
        {
            "name": "PlantDoc-Object-Detection-Dataset-Linux",
            "path": external_root / "PlantDoc-Object-Detection-Dataset(Linux)",
            "task": "detection",
            "notes": "In-the-wild detection dataset",
        },
        {
            "name": "AI-Challenger-PDR2018",
            "path": external_root / "Plant-Disese-Dataset",
            "task": "classification_archive",
            "notes": "AI Challenger crop disease recognition archives",
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a reusable plant-disease dataset bundle from local assets")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=datasets_root() / "PlantDisease-10GB-Bundle",
        help="Target bundle directory",
    )
    return parser.parse_args()


def is_junk_path(path: Path) -> bool:
    return any(part == ".git" for part in path.parts)


def scan_directory(directory: Path):
    file_count = 0
    total_bytes = 0

    for root, dirs, files in os.walk(directory):
        current_root = Path(root)
        dirs[:] = [name for name in dirs if name != ".git"]
        if is_junk_path(current_root):
            continue

        for file_name in files:
            file_path = current_root / file_name
            file_count += 1
            total_bytes += file_path.stat().st_size

    return file_count, total_bytes


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


def write_readme(target_dir: Path, manifest: dict):
    lines = [
        "# Plant Disease 10GB Bundle",
        "",
        "This bundle is assembled from local project data plus public plant-disease datasets already present on disk.",
        "",
        f"- Generated: {manifest['generated_at']}",
        f"- Bundle path: `{manifest['target_dir']}`",
        f"- Logical total size: `{manifest['logical_total_gb']} GB`",
        f"- Logical total files: `{manifest['logical_total_files']}`",
        "",
        "## Included Sources",
        "",
    ]

    for source in manifest["sources"]:
        lines.extend([
            f"### {source['name']}",
            f"- Task type: `{source['task']}`",
            f"- Source path: `{source['source_path']}`",
            f"- Linked path: `{source['linked_path']}`",
            f"- File count: `{source['file_count']}`",
            f"- Logical size: `{source['logical_size_gb']} GB`",
            f"- Notes: {source['notes']}",
            "",
        ])

    lines.extend([
        "## Usage Notes",
        "",
        "- `project-selfmade-*` is the project's own dataset and should be the first choice for immediate fine-tuning.",
        "- `PlantVillage-Dataset` is clean and large, but visually controlled; use it for pretraining or class expansion, not as the only realism source.",
        "- `PlantDoc-Object-Detection-Dataset-Linux` is useful for harder real-world disease appearance and future detection work.",
        "- `AI-Challenger-PDR2018` is currently kept as archives for provenance and later extraction/conversion.",
        "",
    ])

    (target_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    target_dir = args.target_dir.resolve()
    sources_dir = target_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    manifest_sources = []
    total_files = 0
    total_bytes = 0

    for source in default_bundle_sources():
        source_path = Path(source["path"])
        if not source_path.exists():
            continue

        link_path = sources_dir / source["name"]
        if source_path.is_dir():
            ensure_junction(link_path, source_path)
            file_count, size_bytes = scan_directory(source_path)
        else:
            raise ValueError(f"Expected directory source, got file: {source_path}")

        total_files += file_count
        total_bytes += size_bytes
        manifest_sources.append(
            {
                "name": source["name"],
                "task": source["task"],
                "notes": source["notes"],
                "source_path": str(source_path),
                "linked_path": str(link_path),
                "file_count": file_count,
                "logical_size_bytes": size_bytes,
                "logical_size_gb": bytes_to_gb(size_bytes),
            }
        )

    manifest = {
        "bundle_name": "PlantDisease-10GB-Bundle",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_dir": str(target_dir),
        "logical_total_files": total_files,
        "logical_total_bytes": total_bytes,
        "logical_total_gb": bytes_to_gb(total_bytes),
        "sources": manifest_sources,
    }

    with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    write_readme(target_dir, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
