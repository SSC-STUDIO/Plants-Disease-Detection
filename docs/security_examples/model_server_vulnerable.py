#!/usr/bin/env python
"""Intentionally vulnerable model download service for security lessons.

Do not run this service in production. It demonstrates three common issues:

1. debug mode enabled
2. unauthenticated model downloads
3. unsafe path handling

Use the root-level model_server.py for the safer reference implementation.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, send_file


app = Flask(__name__)
DEBUG = True
MODEL_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/models", methods=["GET"])
def list_models():
    models = []
    if MODEL_DIR.exists():
        for root, _, files in os.walk(MODEL_DIR):
            for filename in files:
                if filename.endswith((".pth", ".pt", ".pth.tar", ".onnx")):
                    path = Path(root) / filename
                    models.append({"name": filename, "path": str(path.relative_to(MODEL_DIR))})
    return jsonify({"models": models})


@app.route("/download/<path:model_path>", methods=["GET"])
def download_model(model_path):
    # Vulnerable on purpose: directly joins user input and does not authenticate.
    file_path = MODEL_DIR / model_path
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=DEBUG)
