"""Tests for model creation and architecture."""
import pytest
import torch

from models.model import MODEL_REGISTRY, get_available_models, get_net


class TestModelCreation:
    @pytest.mark.parametrize(
        "model_name",
        ["convnext_small", "densenet169", "efficientnet_b4"],
    )
    def test_model_creation_without_pretrained(self, model_name):
        model = get_net(model_name, num_classes=59, pretrained=False)
        assert model is not None
        assert hasattr(model, 'forward')

    def test_available_models_matches_registry(self):
        assert sorted(get_available_models()) == sorted(MODEL_REGISTRY.keys())

    def test_model_forward_pass(self):
        model = get_net('convnext_small', num_classes=59, pretrained=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 59)

    def test_model_output_finite(self):
        model = get_net('convnext_small', num_classes=59, pretrained=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            output = model(dummy_input)
        assert torch.isfinite(output).all()
