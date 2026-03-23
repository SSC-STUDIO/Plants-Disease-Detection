"""Basic smoke test to verify test infrastructure."""
import pytest


def test_imports():
    """Test that core modules can be imported."""
    import config
    import models.model
    import dataset.dataloader
    import libs.training
    import libs.inference
    import libs.evaluation
    import utils.utils

    assert config is not None
    assert models.model is not None


def test_pytest_working():
    """Sanity check that pytest is working."""
    assert True
    assert 1 + 1 == 2
