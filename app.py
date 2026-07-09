"""Gradio web demo for plant disease classification."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import requests
from PIL import Image

from config import DefaultConfigs
from libs.inference import InferenceManager

DEFAULT_MODEL_URL = (
    "https://github.com/SSC-STUDIO/Plants-Disease-Detection/releases/download/"
    "convnext-small-filtered-v0.1/best_model.pth.tar"
)
DEFAULT_LABELS_URL = (
    "https://github.com/SSC-STUDIO/Plants-Disease-Detection/releases/download/"
    "convnext-small-filtered-v0.1/training_filtered_labels.json"
)
DEFAULT_MODEL_PATH = Path("checkpoints/best/convnext_small/0/best_model.pth.tar")
DEFAULT_LABELS_PATH = Path("reports/training_filtered_labels.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Plants Disease Detection web demo")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to model checkpoint")
    parser.add_argument("--model-name", default="convnext_small", help="Model architecture name")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="Optional label mapping JSON")
    parser.add_argument(
        "--download-url",
        default=os.getenv("PLANT_DISEASE_MODEL_URL", DEFAULT_MODEL_URL),
        help="Checkpoint URL used when --download is set",
    )
    parser.add_argument(
        "--labels-url",
        default=os.getenv("PLANT_DISEASE_LABELS_URL", DEFAULT_LABELS_URL),
        help="Label mapping URL used when --download is set",
    )
    parser.add_argument("--download", action="store_true", help="Download checkpoint if it is missing")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Inference device")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL")
    parser.add_argument("--server-name", default="127.0.0.1", help="Server bind address")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    return parser.parse_args()


def maybe_download_file(target_path: Path, url: str) -> None:
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


def load_labels(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, ValueError) as exc:
        logging.getLogger("app.load_labels").warning(
            "Labels file %s is corrupted or empty (%s); falling back to numeric class IDs",
            path,
            exc,
        )
        return {}

    if isinstance(payload, dict):
        if "labels" in payload and isinstance(payload["labels"], dict):
            payload = payload["labels"]
        labels = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                value = value.get("name", value.get("label", value.get("id", key)))
            labels[int(key)] = str(value)
        return labels
    if isinstance(payload, list):
        return {idx: str(value) for idx, value in enumerate(payload)}
    return {}


def build_demo(args: argparse.Namespace):
    import gradio as gr

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    if args.download:
        maybe_download_file(model_path, args.download_url)
        maybe_download_file(labels_path, args.labels_url)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Run training first or pass --download to fetch the release checkpoint."
        )

    cfg = DefaultConfigs()
    cfg.model_name = args.model_name
    device = None if args.device == "auto" else args.device
    labels = load_labels(labels_path)

    manager = InferenceManager(
        model_path=str(model_path),
        model_name=args.model_name,
        device=device,
        cfg=cfg,
        verify_model_integrity=False,
    )
    manager.load_model()

    def predict(image: Image.Image, tta_views: int, topk: int) -> Dict[str, float]:
        if image is None:
            return {}
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            image.convert("RGB").save(temp_path)
        try:
            probabilities = manager.predict_single(temp_path, tta_views=tta_views)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        limit = max(1, min(int(topk), len(probabilities)))
        indices = np.argsort(probabilities)[::-1][:limit]
        return {
            labels.get(int(idx), f"class_{int(idx)}"): float(probabilities[idx])
            for idx in indices
        }

    with gr.Blocks(title="Plants Disease Detection") as demo:
        gr.Markdown("# Plants Disease Detection")
        gr.Markdown("Upload a plant leaf image to inspect top disease-class predictions from the trained baseline.")
        with gr.Row():
            image = gr.Image(type="pil", label="Leaf image")
            output = gr.Label(num_top_classes=5, label="Predictions")
        with gr.Row():
            tta_views = gr.Slider(1, 4, value=1, step=1, label="TTA views")
            topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
        run = gr.Button("Predict", variant="primary")
        run.click(predict, inputs=[image, tta_views, topk], outputs=output)

    return demo


def main() -> None:
    args = parse_args()
    demo = build_demo(args)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
