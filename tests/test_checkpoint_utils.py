"""Tests for checkpoint_utils shared helpers."""
from libs.checkpoint_utils import infer_model_name_from_path


class TestInferModelNameFromPath:
    """Verify the shared path-based model-name inference used by inference and evaluation."""

    def test_convnext_small_detected(self):
        path = "checkpoints/best/convnext_small/0/best_model.pth.tar"
        assert infer_model_name_from_path(path) == "convnext_small"

    def test_convnextv2_base_384_detected(self):
        path = "checkpoints/convnextv2_base_384/0/best_model.pth.tar"
        assert infer_model_name_from_path(path) == "convnextv2_base_384"

    def test_densenet169_detected(self):
        path = "checkpoints/best/densenet169/0/_latest_model.pth.tar"
        assert infer_model_name_from_path(path) == "densenet169"

    def test_windows_backslash_paths(self):
        path = r"checkpoints\best\convnext_small\0\best_model.pth.tar"
        assert infer_model_name_from_path(path) == "convnext_small"

    def test_unknown_architecture_returns_none(self):
        path = "checkpoints/best/resnet50/0/best_model.pth.tar"
        assert infer_model_name_from_path(path) is None

    def test_empty_path_returns_none(self):
        assert infer_model_name_from_path("") is None

    def test_no_directory_segment_returns_none(self):
        # The candidate must appear as a directory segment, not a bare filename.
        assert infer_model_name_from_path("convnext_small.pth.tar") is None
