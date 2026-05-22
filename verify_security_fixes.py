#!/usr/bin/env python
"""Verify the optional model artifact server security behavior."""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path


def load_model_server(model_dir: Path, api_key: str):
    os.environ["MODEL_API_KEY"] = api_key
    os.environ["MODEL_DIR"] = str(model_dir)
    os.environ["FLASK_DEBUG"] = "0"

    import model_server

    return importlib.reload(model_server)


def main() -> int:
    try:
        import flask  # noqa: F401
    except ImportError:
        print("Flask is not installed. Install demo/server dependencies first.")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        (model_dir / "demo.pth").write_text("fake model data", encoding="utf-8")

        server = load_model_server(model_dir, "test-key")
        app = server.app
        app.config["TESTING"] = True
        client = app.test_client()

        checks = [
            ("health endpoint", client.get("/health").status_code == 200),
            ("unauthenticated list is rejected", client.get("/models").status_code == 401),
            (
                "authenticated list works",
                client.get("/models", headers={"X-API-Key": "test-key"}).status_code == 200,
            ),
            (
                "path traversal is rejected",
                client.get("/download/../../../etc/passwd", headers={"X-API-Key": "test-key"}).status_code
                in {403, 404},
            ),
            (
                "valid model download works",
                client.get("/download/demo.pth", headers={"X-API-Key": "test-key"}).status_code == 200,
            ),
        ]

    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
