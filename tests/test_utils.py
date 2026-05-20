"""Tests for utility functions."""
import torch

from config import DefaultConfigs
from utils.utils import build_transforms, get_loss_function, get_optimizer, get_scheduler, handle_datasets


class TestTransforms:
    def test_train_transforms(self):
        cfg = DefaultConfigs()
        transforms = build_transforms(train=True, test=False, use_data_aug=True, cfg=cfg)
        assert transforms is not None

    def test_val_transforms(self):
        cfg = DefaultConfigs()
        transforms = build_transforms(train=False, test=False, use_data_aug=False, cfg=cfg)
        assert transforms is not None

    def test_test_transforms(self):
        cfg = DefaultConfigs()
        transforms = build_transforms(train=False, test=True, use_data_aug=False, cfg=cfg)
        assert transforms is not None


class TestLossFunctions:
    def test_focal_loss_creation(self):
        cfg = DefaultConfigs()
        cfg.use_focal_loss = True
        loss_fn = get_loss_function(torch.device('cpu'), cfg=cfg)
        assert loss_fn is not None

    def test_crossentropy_loss_creation(self):
        cfg = DefaultConfigs()
        cfg.use_focal_loss = False
        cfg.label_smoothing = 0.0
        loss_fn = get_loss_function(torch.device('cpu'), cfg=cfg)
        assert loss_fn is not None


class TestOptimizers:
    def test_adamw_optimizer(self):
        cfg = DefaultConfigs()
        model = torch.nn.Linear(10, 5)
        optimizer = get_optimizer(model, name='adamw', cfg=cfg)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_sgd_optimizer(self):
        cfg = DefaultConfigs()
        model = torch.nn.Linear(10, 5)
        optimizer = get_optimizer(model, name='sgd', cfg=cfg)
        assert isinstance(optimizer, torch.optim.SGD)


class TestSchedulers:
    def test_cosine_scheduler(self):
        cfg = DefaultConfigs()
        cfg.scheduler = 'cosine'
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = get_scheduler(optimizer, num_epochs=2, steps_per_epoch=1, cfg=cfg)
        assert scheduler is not None

    def test_onecycle_scheduler(self):
        cfg = DefaultConfigs()
        cfg.scheduler = 'onecycle'
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = get_scheduler(optimizer, num_epochs=2, steps_per_epoch=2, cfg=cfg)
        assert scheduler is not None


class TestDatasetSelection:
    def test_custom_dataset_uses_labeled_test_split_as_validation(self, temp_dir):
        dataset_root = temp_dir / "external"
        (dataset_root / "train" / "0").mkdir(parents=True)
        (dataset_root / "test" / "0").mkdir(parents=True)
        (dataset_root / "train" / "0" / "sample.jpg").write_bytes(b"image")
        (dataset_root / "test" / "0" / "sample.jpg").write_bytes(b"image")

        cfg = DefaultConfigs()
        cfg.dataset_path = str(dataset_root)
        cfg.use_custom_dataset_path = True

        selected = handle_datasets(data_type="val", cfg=cfg)

        assert selected == str(dataset_root / "test")
