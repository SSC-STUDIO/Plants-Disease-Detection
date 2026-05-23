#!/usr/bin/env python
"""Standalone Python inference client for the released plant disease model.

Examples:
    python clients/python/infer.py --input leaf.jpg --download --output prediction.json
    python clients/python/infer.py --input samples/ --download --topk 5 --output predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import DEFAULT_LABELS_PATH, DEFAULT_LABELS_URL, DEFAULT_MODEL_PATH, DEFAULT_MODEL_URL, load_labels
from config import DefaultConfigs
from libs.inference import InferenceManager
from utils.utils import get_image_extensions, is_image_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run plant disease inference from a Python client")
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Checkpoint path")
    parser.add_argument("--model-name", default="convnext_small", help="Model architecture")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="Label mapping JSON")
    parser.add_argument("--download", action="store_true", help="Download release checkpoint and labels if missing")
    parser.add_argument("--model-url", default=os.getenv("PLANT_DISEASE_MODEL_URL", DEFAULT_MODEL_URL))
    parser.add_argument("--labels-url", default=os.getenv("PLANT_DISEASE_LABELS_URL", DEFAULT_LABELS_URL))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Inference device")
    parser.add_argument("--topk", type=int, default=5, help="Number of predictions to return")
    parser.add_argument("--tta-views", type=int, choices=[1, 2, 3, 4], default=1, help="Test-time augmentation views")
    parser.add_argument("--batch-size", type=int, default=16, help="Directory inference batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Directory inference worker count")
    parser.add_argument("--save-probs", action="store_true", help="Include full probability vectors")
    return parser.parse_args()


def download_file(target_path: Path, url: str) -> None:
    if target_path.exists():
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, dir=str(target_path.parent)) as tmp:
            tmp_path = Path(tmp.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
    tmp_path.replace(target_path)


def iter_image_files(input_path: Path, cfg: DefaultConfigs) -> Iterable[Path]:
    image_extensions = get_image_extensions(cfg=cfg)
    if input_path.is_file():
        if is_image_file(str(input_path), image_extensions):
            yield input_path
        return

    for path in sorted(input_path.iterdir()):
        if path.is_file() and is_image_file(str(path), image_extensions):
            yield path


def format_topk(probabilities: np.ndarray, labels: Dict[int, str], topk: int) -> List[Dict[str, Any]]:
    limit = max(1, min(int(topk), len(probabilities)))
    indices = np.argsort(probabilities)[::-1][:limit]
    return [
        {
            "class_id": int(index),
            "label": labels.get(int(index), f"class_{int(index)}"),
            "score": float(probabilities[index]),
        }
        for index in indices
    ]


def predict_file(
    manager: InferenceManager,
    image_path: Path,
    labels: Dict[int, str],
    topk: int,
    tta_views: int,
    save_probs: bool,
) -> Dict[str, Any]:
    probabilities = manager.predict_single(str(image_path), tta_views=tta_views)
    topk_items = format_topk(probabilities, labels, topk)
    result: Dict[str, Any] = {
        "image": str(image_path),
        "top_prediction": topk_items[0],
        "topk": topk_items,
    }
    if save_probs:
        result["probabilities"] = probabilities.tolist()
    return result


def write_output(output_path: str, payload: List[Dict[str, Any]]) -> None:
    if not output_path:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)
    labels_path = Path(args.labels)

    if args.download:
        download_file(model_path, args.model_url)
        download_file(labels_path, args.labels_url)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Use --download or pass --model.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    cfg = DefaultConfigs()
    cfg.model_name = args.model_name
    labels = load_labels(labels_path)
    device = None if args.device == "auto" else args.device

    manager = InferenceManager(
        model_path=str(model_path),
        model_name=args.model_name,
        device=device,
        cfg=cfg,
        verify_model_integrity=False,
    )
    manager.load_model()

    image_files = list(iter_image_files(input_path, cfg))
    if not image_files:
        raise FileNotFoundError(f"No supported image files found in: {input_path}")

    results = [
        predict_file(manager, image_path, labels, args.topk, args.tta_views, args.save_probs)
        for image_path in image_files
    ]
    write_output(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
