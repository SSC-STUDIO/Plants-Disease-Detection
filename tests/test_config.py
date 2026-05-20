"""Tests for configuration validation."""
import pytest

from config import DefaultConfigs


class TestConfigValidation:
    """Test configuration validation and defaults."""

    def test_default_config_creation(self):
        cfg = DefaultConfigs()
        assert cfg is not None
        assert cfg.num_classes >= 1
        assert cfg.img_height == 384
        assert cfg.img_width == 384
        assert cfg.progressive_resizing is False

    def test_num_classes_syncs_from_train_directory(self, temp_dir):
        for class_id in [0, 1, 60]:
            (temp_dir / "train" / str(class_id)).mkdir(parents=True)

        cfg = DefaultConfigs()
        cfg.paths.merged_train_dir = str(temp_dir / "missing_merged").replace("\\", "/")
        cfg.paths.train_dir = str(temp_dir / "train").replace("\\", "/")
        cfg.train_data = cfg.paths.train_dir
        cfg.num_classes = 186

        detected = cfg.refresh_num_classes_from_data_dirs()

        assert detected == 61
        assert cfg.num_classes == 61

    def test_num_workers_auto(self):
        cfg = DefaultConfigs()
        cfg.num_workers = 'auto'
        cfg.__post_init__()
        assert isinstance(cfg.num_workers, int)
        assert cfg.num_workers > 0

    def test_device_validation(self):
        cfg = DefaultConfigs()
        cfg.device = 'invalid'
        with pytest.raises(ValueError, match="device must be"):
            cfg.__post_init__()

    def test_image_extensions_normalization(self):
        cfg = DefaultConfigs()
        cfg.image_extensions = ['jpg', '.png', 'JPEG']
        cfg.__post_init__()
        assert all(ext.startswith('.') for ext in cfg.image_extensions)
        assert all(ext.islower() for ext in cfg.image_extensions)

    def test_tta_views_range(self):
        cfg = DefaultConfigs()
        assert cfg.tta_views in [1, 2, 3, 4]
