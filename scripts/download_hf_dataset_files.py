#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import fnmatch
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Download selected HF dataset files from repo metadata")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com")
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--target-local-gb", type=float, default=0.0, help="Stop once selected local files reach this size")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def matches(path: str, patterns: List[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def local_file_size(output_dir: Path, filename: str) -> int:
    path = output_dir / filename
    return path.stat().st_size if path.exists() and path.is_file() else 0


def selected_local_size(output_dir: Path, files: Iterable[str]) -> int:
    return sum(local_file_size(output_dir, filename) for filename in files)


def class_key(filename: str) -> str:
    parts = filename.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "data":
        return parts[1]
    return ""


def round_robin_by_class(file_infos: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    groups: Dict[str, List[Tuple[str, int]]] = {}
    for item in file_infos:
        groups.setdefault(class_key(item[0]), []).append(item)
    for group in groups.values():
        group.sort(key=lambda item: item[0])

    ordered: List[Tuple[str, int]] = []
    keys = sorted(groups)
    while keys:
        next_keys = []
        for key in keys:
            group = groups[key]
            if group:
                ordered.append(group.pop(0))
            if group:
                next_keys.append(key)
        keys = next_keys
    return ordered


def download_one(repo_id: str, filename: str, output_dir: Path, endpoint: str, retries: int) -> Tuple[str, bool, str]:
    from huggingface_hub import hf_hub_download

    for attempt in range(1, retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=output_dir,
                endpoint=endpoint,
            )
            return filename, True, ""
        except Exception as exc:
            if attempt == retries:
                return filename, False, repr(exc)
            time.sleep(min(2 ** attempt, 20))
    return filename, False, "unknown failure"


def main():
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import HfApi

    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=args.endpoint)
    info = api.dataset_info(args.repo_id, files_metadata=True)
    candidates: List[Tuple[str, int]] = []
    for sibling in info.siblings or []:
        filename = sibling.rfilename
        size = int(getattr(sibling, "size", 0) or 0)
        if matches(filename, args.allow_pattern):
            candidates.append((filename, size))

    candidates = round_robin_by_class(candidates)
    target_bytes = int(args.target_local_gb * (1024 ** 3)) if args.target_local_gb else 0

    selected: List[Tuple[str, int]] = []
    expected_bytes = 0
    for filename, size in candidates:
        selected.append((filename, size))
        expected_bytes += size
        if target_bytes and expected_bytes >= target_bytes:
            break

    selected_names = [filename for filename, _ in selected]
    current_bytes = selected_local_size(args.output_dir, selected_names)
    print(json.dumps({
        "event": "download_plan",
        "repo_id": args.repo_id,
        "endpoint": args.endpoint,
        "output_dir": str(args.output_dir.resolve()),
        "candidate_files": len(candidates),
        "selected_files": len(selected),
        "selected_expected_gb": round(expected_bytes / (1024 ** 3), 3),
        "selected_local_gb": round(current_bytes / (1024 ** 3), 3),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False), flush=True)

    if target_bytes and current_bytes >= target_bytes:
        print(json.dumps({"event": "target_already_met"}, ensure_ascii=False), flush=True)
        return

    pending = [
        filename
        for filename, _ in selected
        if local_file_size(args.output_dir, filename) == 0
    ]

    completed = 0
    failed: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(download_one, args.repo_id, filename, args.output_dir, args.endpoint, args.retries)
            for filename in pending
        ]
        for future in as_completed(futures):
            filename, ok, error = future.result()
            completed += 1
            if not ok:
                failed.append((filename, error))

            if completed % 250 == 0 or completed == len(pending):
                current_bytes = selected_local_size(args.output_dir, selected_names)
                print(json.dumps({
                    "event": "download_progress",
                    "completed": completed,
                    "pending": len(pending),
                    "failed": len(failed),
                    "selected_local_gb": round(current_bytes / (1024 ** 3), 3),
                }, ensure_ascii=False), flush=True)

    current_bytes = selected_local_size(args.output_dir, selected_names)
    print(json.dumps({
        "event": "download_complete",
        "repo_id": args.repo_id,
        "selected_local_gb": round(current_bytes / (1024 ** 3), 3),
        "failed": failed[:20],
        "failed_count": len(failed),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
