#!/usr/bin/env python
"""Small authenticated service for downloading local model artifacts.

This service is optional and is intended for controlled local or internal use.
Set MODEL_API_KEY before exposing it beyond localhost.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file


ALLOWED_EXTENSIONS = (".pth", ".pt", ".pth.tar", ".onnx", ".json")

app = Flask(__name__)

API_KEY = os.environ.get("MODEL_API_KEY")
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "0") == "1"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent / "checkpoints")).resolve()


def require_auth(handler):
    """Require an X-API-Key header for model artifact endpoints."""

    @functools.wraps(handler)
    def decorated_function(*args, **kwargs):
        if not API_KEY:
            return jsonify({"error": "Server is missing MODEL_API_KEY"}), 503

        auth_header = request.headers.get("X-API-Key")
        if not auth_header:
            return jsonify({"error": "Unauthorized", "message": "Missing X-API-Key header."}), 401

        if auth_header != API_KEY:
            app.logger.warning("Invalid API key attempt from %s", request.remote_addr)
            return jsonify({"error": "Unauthorized", "message": "Invalid API key."}), 401

        return handler(*args, **kwargs)

    return decorated_function


def is_safe_model_path(model_path: str) -> bool:
    """Return True when model_path resolves inside MODEL_DIR."""
    try:
        target = (MODEL_DIR / model_path).resolve()
        target.relative_to(MODEL_DIR)
        return True
    except (OSError, ValueError):
        return False


def is_allowed_artifact(model_path: str) -> bool:
    return model_path.lower().endswith(ALLOWED_EXTENSIONS)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "model-download-service"})


@app.route("/models", methods=["GET"])
@require_auth
def list_models():
    models = []
    if MODEL_DIR.exists():
        for path in sorted(MODEL_DIR.rglob("*")):
            if path.is_file() and is_allowed_artifact(path.name):
                rel_path = path.relative_to(MODEL_DIR).as_posix()
                models.append({"name": path.name, "path": rel_path, "size": path.stat().st_size})
    return jsonify({"models": models})


@app.route("/download/<path:model_path>", methods=["GET"])
@require_auth
def download_model(model_path: str):
    if not is_safe_model_path(model_path):
        app.logger.warning("Path traversal attempt: %s from %s", model_path, request.remote_addr)
        abort(403, "Access denied: invalid path")

    if not is_allowed_artifact(model_path):
        app.logger.warning("Invalid file type download attempt: %s", model_path)
        abort(403, "Access denied: invalid file type")

    file_path = (MODEL_DIR / model_path).resolve()
    if not file_path.exists() or not file_path.is_file():
        abort(404, "Model file not found")

    app.logger.info("Model downloaded: %s by %s", model_path, request.remote_addr)
    return send_file(file_path, as_attachment=True, download_name=file_path.name)


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(403)
def forbidden(error):
    return jsonify({"error": "Forbidden"}), 403


@app.errorhandler(500)
def internal_error(error):
    app.logger.error("Server error: %s", error)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    if DEBUG_MODE:
        print("WARNING: Running in DEBUG mode. Set FLASK_DEBUG=0 for production.")
    if not API_KEY:
        print("WARNING: MODEL_API_KEY is not set. Authenticated endpoints will return 503.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=DEBUG_MODE)
