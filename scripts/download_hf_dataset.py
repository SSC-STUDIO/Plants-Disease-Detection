#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset snapshot with mirror support")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, for example mbsoft31/agri-foundation-v1")
    parser.add_argument("--output-dir", type=Path, required=True, help="Local directory for the snapshot")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com")
    parser.add_argument("--allow-pattern", action="append", default=[], help="Allowed file pattern. Can be repeated.")
    parser.add_argument("--ignore-pattern", action="append", default=[], help="Ignored file pattern. Can be repeated.")
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import snapshot_download

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "event": "download_start",
                "repo_id": args.repo_id,
                "repo_type": "dataset",
                "endpoint": args.endpoint,
                "output_dir": str(args.output_dir.resolve()),
                "allow_patterns": args.allow_pattern,
                "ignore_patterns": args.ignore_pattern,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=args.allow_pattern or None,
        ignore_patterns=args.ignore_pattern or None,
        max_workers=args.max_workers,
        endpoint=args.endpoint,
    )

    print(
        json.dumps(
            {
                "event": "download_complete",
                "repo_id": args.repo_id,
                "output_dir": str(Path(path).resolve()),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
