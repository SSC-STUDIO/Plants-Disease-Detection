"""Tests for utility functions."""
import torch

from config import DefaultConfigs
from utils.utils import build_transforms, get_loss_function, get_optimizer, get_scheduler


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
