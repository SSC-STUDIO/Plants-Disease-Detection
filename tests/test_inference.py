import torch

from config import DefaultConfigs
from libs.inference import InferenceManager
from app import load_labels


def test_inference_load_model_syncs_num_classes_from_checkpoint(monkeypatch, temp_dir):
    checkpoint_path = temp_dir / "best_model.pth.tar"
    torch.save(
        {
            "state_dict": {
                "classifier.2.weight": torch.zeros(117, 768),
                "classifier.2.bias": torch.zeros(117),
            }
        },
        checkpoint_path,
    )

    captured = {}

    class DummyModel(torch.nn.Module):
        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict):
            return None

    def fake_get_net(model_name, num_classes, pretrained):
        captured["model_name"] = model_name
        captured["num_classes"] = num_classes
        captured["pretrained"] = pretrained
        return DummyModel()

    cfg = DefaultConfigs()
    cfg.num_classes = 61
    monkeypatch.setattr("libs.inference.get_net", fake_get_net)

    manager = InferenceManager(
        model_path=str(checkpoint_path),
        model_name="convnext_small",
        device="cpu",
        cfg=cfg,
        verify_model_integrity=False,
    )
    manager.load_model()

    assert captured["num_classes"] == 117
    assert cfg.num_classes == 117


def test_app_load_labels_supports_release_mapping(temp_dir):
    labels_path = temp_dir / "labels.json"
    labels_path.write_text(
        '{"0": {"id": 0, "name": "Apple___healthy"}, "1": {"id": 1, "name": "Tomato___Late_blight"}}',
        encoding="utf-8",
    )

    assert load_labels(labels_path) == {
        0: "Apple___healthy",
        1: "Tomato___Late_blight",
    }


def test_app_load_labels_returns_empty_on_corrupted_json(temp_dir):
    labels_path = temp_dir / "bad_labels.json"
    labels_path.write_text("{not valid json!!!", encoding="utf-8")

    assert load_labels(labels_path) == {}


def test_app_load_labels_returns_empty_on_empty_file(temp_dir):
    labels_path = temp_dir / "empty.json"
    labels_path.write_text("", encoding="utf-8")

    assert load_labels(labels_path) == {}


def test_app_load_labels_handles_list_format(temp_dir):
    labels_path = temp_dir / "labels_list.json"
    labels_path.write_text('["Apple___healthy", "Tomato___Late_blight"]', encoding="utf-8")

    assert load_labels(labels_path) == {
        0: "Apple___healthy",
        1: "Tomato___Late_blight",
    }
