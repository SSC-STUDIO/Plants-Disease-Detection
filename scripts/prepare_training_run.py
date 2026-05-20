#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config
from dataset.stats import summarize_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Create a reproducible training run artifact folder")
    parser.add_argument("--output-dir", type=Path, default=None, help="Run artifact directory")
    parser.add_argument("--train-data", default="./data/train", help="Training data path for stats")
    parser.add_argument("--val-data", default="./data/val", help="Validation data path for stats")
    parser.add_argument("--model", default="convnextv2_base_384", help="Model name")
    parser.add_argument("--epochs", type=int, default=30, help="Planned epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Planned batch size")
    parser.add_argument("--seed", type=int, default=888, help="Random seed")
    return parser.parse_args()


def make_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: make_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("reports") / f"train_run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_command = [
        "python", "main.py", "train",
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--dataset-path", "./data",
        "--seed", str(args.seed),
        "--force-train",
        "--no-wandb",
    ]

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "planned_model": args.model,
        "planned_epochs": args.epochs,
        "planned_batch_size": args.batch_size,
        "planned_seed": args.seed,
        "train_command": train_command,
        "config": make_jsonable(config),
    }

    write_json(output_dir / "config.json", run_config)
    write_json(output_dir / "train_stats.json", summarize_dataset(args.train_data, cfg=config))
    if Path(args.val_data).exists():
        write_json(output_dir / "val_stats.json", summarize_dataset(args.val_data, cfg=config))
    write_json(
        output_dir / "metrics.json",
        {
            "status": "not_trained",
            "message": "Run training and evaluation, then replace this file with measured metrics.",
        },
    )

    readme = [
        "# Training Run",
        "",
        f"- Created: `{run_config['created_at']}`",
        f"- Git revision: `{run_config['git_revision']}`",
        f"- Model: `{args.model}`",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        "",
        "## Command",
        "",
        "```powershell",
        " ".join(train_command),
        "```",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
