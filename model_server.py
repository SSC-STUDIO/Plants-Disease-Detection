#!/usr/bin/env python
"""Small authenticated service for downloading local model artifacts.

This service is optional and is intended for controlled local or internal use.
Set MODEL_API_KEY before exposing it beyond localhost.
"""

from __future__ import annotations

import functools
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file


ALLOWED_EXTENSIONS = (".pth", ".pt", ".pth.tar", ".onnx", ".json")

app = Flask(__name__)
_start_time = time.monotonic()

API_KEY = os.environ.get("MODEL_API_KEY")
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "0") == "1"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent / "checkpoints")).resolve()

# --- Rate limiting for authentication attempts ---
# Prevents brute-force attacks on the API key by tracking failed auth attempts
# per client IP within a rolling time window.
RATE_LIMIT_WINDOW = int(os.environ.get("MODEL_API_RATE_WINDOW", "60"))  # seconds
RATE_LIMIT_MAX_FAILURES = int(os.environ.get("MODEL_API_RATE_MAX_FAILURES", "10"))

# Map: client_ip -> deque of failure timestamps
_auth_failures: "defaultdict[str, deque[float]]" = defaultdict(deque)


def _record_auth_failure(client_ip: str) -> int:
    """Record a failed auth attempt and return the current failure count in the window."""
    now = time.monotonic()
    failures = _auth_failures[client_ip]
    # Evict timestamps outside the rolling window
    cutoff = now - RATE_LIMIT_WINDOW
    while failures and failures[0] < cutoff:
        failures.popleft()
    failures.append(now)
    return len(failures)


def _is_rate_limited(client_ip: str) -> bool:
    """Check whether the client IP has exceeded the failure threshold."""
    now = time.monotonic()
    failures = _auth_failures.get(client_ip)
    if not failures:
        return False
    cutoff = now - RATE_LIMIT_WINDOW
    while failures and failures[0] < cutoff:
        failures.popleft()
    # Garbage-collect: if all failures have expired, remove the IP key
    # entirely so the dict does not grow without bound (memory leak
    # protection — an attacker probing from many IPs could otherwise
    # exhaust server memory with empty deques).
    if not failures:
        del _auth_failures[client_ip]
        return False
    return len(failures) >= RATE_LIMIT_MAX_FAILURES


def require_auth(handler):
    """Require an X-API-Key header for model artifact endpoints.

    Includes IP-based rate limiting: after RATE_LIMIT_MAX_FAILURES failed
    authentication attempts within RATE_LIMIT_WINDOW seconds, subsequent
    requests from the same IP are rejected with 429 Too Many Requests until
    the oldest failure expires from the window.
    """

    @functools.wraps(handler)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr or "unknown"

        if _is_rate_limited(client_ip):
            app.logger.warning(
                "Rate limit exceeded for %s (%d failures in %ds window)",
                client_ip,
                RATE_LIMIT_MAX_FAILURES,
                RATE_LIMIT_WINDOW,
            )
            return jsonify({
                "error": "Too many failed authentication attempts",
                "message": f"Rate limit: max {RATE_LIMIT_MAX_FAILURES} failures per {RATE_LIMIT_WINDOW}s",
            }), 429

        if not API_KEY:
            return jsonify({"error": "Server is missing MODEL_API_KEY"}), 503

        auth_header = request.headers.get("X-API-Key")
        if not auth_header:
            _record_auth_failure(client_ip)
            return jsonify({"error": "Unauthorized", "message": "Missing X-API-Key header."}), 401

        if auth_header != API_KEY:
            _record_auth_failure(client_ip)
            app.logger.warning("Invalid API key attempt from %s", client_ip)
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
    """Return service status plus basic operational metrics.

    Exposes model count and available disk space on the partition holding
    MODEL_DIR so monitoring systems can detect resource exhaustion (e.g.
    disk-full during checkpoint saves) without an additional SSH probe.
    """
    model_count = 0
    total_size = 0
    if MODEL_DIR.exists():
        for path in MODEL_DIR.rglob("*"):
            if path.is_file() and is_allowed_artifact(path.name):
                model_count += 1
                try:
                    total_size += path.stat().st_size
                except OSError:
                    pass

    disk_free = None
    disk_total = None
    try:
        if hasattr(os, "statvfs"):
            # Unix: use statvfs for filesystem statistics
            stat = os.statvfs(MODEL_DIR)
            disk_free = stat.f_bavail * stat.f_frsize
            disk_total = stat.f_blocks * stat.f_frsize
        else:
            # Windows: use ctypes to call GetDiskFreeSpaceExW
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            available = ctypes.c_ulonglong(0)
            drive = str(MODEL_DIR.resolve().anchor)  # e.g. "C:\\"
            if ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(drive),
                ctypes.pointer(available),
                ctypes.pointer(total_bytes),
                ctypes.pointer(free_bytes),
            ):
                disk_free = free_bytes.value
                disk_total = total_bytes.value
    except (OSError, AttributeError):
        pass

    uptime = time.monotonic() - _start_time

    return jsonify({
        "status": "ok",
        "service": "model-download-service",
        "model_count": model_count,
        "model_dir_size": total_size,
        "disk_free_bytes": disk_free,
        "disk_total_bytes": disk_total,
        "uptime_seconds": round(uptime, 1),
    })


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


@app.errorhandler(429)
def too_many_requests(error):
    return jsonify({"error": "Too many failed authentication attempts"}), 429


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
    print(f"Rate limit: {RATE_LIMIT_MAX_FAILURES} failed auth attempts per {RATE_LIMIT_WINDOW}s per IP")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=DEBUG_MODE)
