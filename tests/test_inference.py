import torch

from config import DefaultConfigs
from libs.inference import InferenceManager


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
