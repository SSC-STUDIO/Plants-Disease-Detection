#!/usr/bin/env python
"""Portable helper for running a remote or headless training job.

The script does not know any hostnames or local machine paths. Run it after
copying the repository and dataset onto the target machine.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def default_dataset_path() -> str:
    root = os.environ.get("PLANT_DATA_ROOT")
    if root:
        return str(Path(root) / "PlantDisease-Open-Training-Filtered")
    return str(Path(".datasets") / "PlantDisease-Open-Training-Filtered")


def run_command(command, env) -> int:
    print("+ " + " ".join(str(part) for part in command))
    return subprocess.run(command, cwd=Path.cwd(), env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a portable plant disease training pipeline")
    parser.add_argument("--dataset-path", default=default_dataset_path(), help="Numeric dataset root with train/ and val/")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--model", default="convnextv2_base_384", help="Model architecture")
    parser.add_argument("--seed", type=int, default=888, help="Random seed")
    parser.add_argument("--offline", action="store_true", help="Disable online model/download lookups where supported")
    parser.add_argument("--force-train", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--no-prepare", action="store_true", help="Skip dataset preparation in main.py")
    parser.add_argument("--no-image-validation", action="store_true", help="Skip startup image validation")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not (dataset_path / "train").exists():
        print(f"ERROR: training split not found: {dataset_path / 'train'}")
        return 1

    env = os.environ.copy()
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    command = [
        sys.executable,
        "main.py",
        "train",
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--dataset-path",
        str(dataset_path),
        "--seed",
        str(args.seed),
        "--no-wandb",
    ]
    if args.force_train:
        command.append("--force-train")
    if args.no_prepare:
        command.append("--no-prepare")
    if args.no_image_validation:
        command.append("--no-image-validation")

    return run_command(command, env)


if __name__ == "__main__":
    raise SystemExit(main())
