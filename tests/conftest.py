"""Shared fixtures for pytest."""
import tempfile
from pathlib import Path

import pytest

from config import DefaultConfigs


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir):
    """Create a lightweight config for isolated tests."""
    cfg = DefaultConfigs()
    cfg.paths.base_dir = str(temp_dir).replace('\\', '/')
    cfg.num_workers = 1
    cfg.aug_num_workers = 1
    cfg.enable_image_validation = False
    cfg.use_ema = False
    cfg.use_amp = False
    cfg.device = 'cpu'
    return cfg
