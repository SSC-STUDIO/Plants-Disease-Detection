import importlib

import pytest


flask = pytest.importorskip("flask")


def load_server(monkeypatch, model_dir, api_key="test-key"):
    monkeypatch.setenv("MODEL_API_KEY", api_key)
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("FLASK_DEBUG", "0")
    import model_server

    return importlib.reload(model_server)


def test_model_server_requires_configured_api_key(monkeypatch, temp_dir):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(temp_dir))
    import model_server

    server = importlib.reload(model_server)
    server.app.config["TESTING"] = True
    client = server.app.test_client()

    assert client.get("/models").status_code == 503


def test_model_server_auth_and_download(monkeypatch, temp_dir):
    (temp_dir / "demo.pth").write_text("fake model data", encoding="utf-8")
    server = load_server(monkeypatch, temp_dir)
    server.app.config["TESTING"] = True
    client = server.app.test_client()

    assert client.get("/health").status_code == 200
    assert client.get("/models").status_code == 401
    assert client.get("/models", headers={"X-API-Key": "test-key"}).status_code == 200

    response = client.get("/download/demo.pth", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.data == b"fake model data"


def test_model_server_rejects_traversal_and_invalid_extensions(monkeypatch, temp_dir):
    (temp_dir / "notes.txt").write_text("not a model", encoding="utf-8")
    server = load_server(monkeypatch, temp_dir)
    server.app.config["TESTING"] = True
    client = server.app.test_client()
    headers = {"X-API-Key": "test-key"}

    assert client.get("/download/../../../etc/passwd", headers=headers).status_code in {403, 404}
    assert client.get("/download/notes.txt", headers=headers).status_code == 403
