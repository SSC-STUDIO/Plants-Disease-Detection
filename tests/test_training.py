"""Targeted tests for training helpers."""
import types
from unittest.mock import MagicMock

import pytest

from libs.training import Trainer


class TestProgressiveResizing:
    def test_progressive_size_disabled_returns_config_size(self, test_config):
        trainer = Trainer(test_config)
        trainer.config.progressive_resizing = False
        assert trainer._get_progressive_size(0) == (trainer.config.img_height, trainer.config.img_width)

    def test_progressive_size_uses_schedule(self, test_config):
        trainer = Trainer(test_config)
        trainer.config.progressive_resizing = True
        trainer.config.progressive_sizes = [224, 320, 380]
        trainer.config.progressive_epochs = 15
        assert trainer._get_progressive_size(0) == (224, 224)
        assert trainer._get_progressive_size(6) == (320, 320)
        assert trainer._get_progressive_size(14) == (380, 380)

    def test_progressive_size_after_transition_uses_end_size(self, test_config):
        trainer = Trainer(test_config)
        trainer.config.progressive_resizing = True
        trainer.config.progressive_end_size = 380
        trainer.config.progressive_epochs = 10
        assert trainer._get_progressive_size(10) == (380, 380)


class TestWandbIntegration:
    def test_init_wandb_skips_when_disabled(self, test_config, monkeypatch):
        trainer = Trainer(test_config)
        trainer.config.use_wandb = False

        monkeypatch.setattr('libs.training.importlib.import_module', lambda name: (_ for _ in ()).throw(AssertionError('should not import wandb')))

        trainer._init_wandb(epochs=5, start_epoch=0, force_train=False)

        assert trainer.wandb_enabled is False
        assert trainer.wandb_module is None
        assert trainer.wandb_run is None

    def test_init_wandb_gracefully_handles_import_failure(self, test_config, monkeypatch):
        trainer = Trainer(test_config)
        trainer.config.use_wandb = True

        def fail_import(name):
            raise ModuleNotFoundError('wandb missing')

        monkeypatch.setattr('libs.training.importlib.import_module', fail_import)

        trainer._init_wandb(epochs=5, start_epoch=0, force_train=False)

        assert trainer.wandb_enabled is False
        assert trainer.wandb_module is None

    def test_init_wandb_gracefully_handles_init_failure(self, test_config, monkeypatch):
        trainer = Trainer(test_config)
        trainer.config.use_wandb = True

        fake_wandb = types.SimpleNamespace(init=MagicMock(side_effect=RuntimeError('init failed')))
        monkeypatch.setattr('libs.training.importlib.import_module', lambda name: fake_wandb)

        trainer._init_wandb(epochs=5, start_epoch=2, force_train=True)

        assert trainer.wandb_enabled is False
        assert trainer.wandb_module is None

    def test_log_wandb_epoch_sends_core_metrics(self, test_config):
        trainer = Trainer(test_config)
        fake_wandb = types.SimpleNamespace(log=MagicMock())
        trainer.wandb_module = fake_wandb
        trainer.wandb_enabled = True

        trainer._log_wandb_epoch(
            epoch=3,
            train_loss=0.12,
            train_top1=91.5,
            train_top2=97.3,
            val_loss=0.2,
            val_top1=88.1,
            best_acc=88.1,
            lr=1e-4,
            current_img_size=(320, 320),
        )

        fake_wandb.log.assert_called_once()
        payload = fake_wandb.log.call_args[0][0]
        assert payload['epoch'] == 3
        assert payload['train/loss'] == 0.12
        assert payload['train/top1'] == 91.5
        assert payload['train/top2'] == 97.3
        assert payload['val/loss'] == 0.2
        assert payload['val/top1'] == 88.1
        assert payload['best_acc'] == 88.1
        assert payload['lr'] == 1e-4
        assert payload['image_size/height'] == 320
        assert payload['image_size/width'] == 320

    def test_log_wandb_epoch_disables_future_logging_after_failure(self, test_config):
        trainer = Trainer(test_config)
        fake_wandb = types.SimpleNamespace(log=MagicMock(side_effect=RuntimeError('log failed')))
        trainer.wandb_module = fake_wandb
        trainer.wandb_enabled = True

        trainer._log_wandb_epoch(
            epoch=1,
            train_loss=0.5,
            train_top1=80.0,
            train_top2=90.0,
            best_acc=80.0,
            lr=1e-3,
            current_img_size=(384, 384),
        )

        assert trainer.wandb_enabled is False

    def test_finish_wandb_ignores_finish_errors(self, test_config):
        trainer = Trainer(test_config)
        trainer.wandb_enabled = True
        trainer.wandb_run = types.SimpleNamespace(summary={})
        trainer.wandb_module = types.SimpleNamespace(finish=MagicMock(side_effect=RuntimeError('finish failed')))

        trainer._finish_wandb({'best_acc': 95.0})

        assert trainer.wandb_enabled is False
        assert trainer.wandb_module is None
        assert trainer.wandb_run is None

    def test_finish_wandb_updates_summary_before_finish(self, test_config):
        trainer = Trainer(test_config)
        finish_mock = MagicMock()
        trainer.wandb_run = types.SimpleNamespace(summary={})
        trainer.wandb_module = types.SimpleNamespace(finish=finish_mock)
        trainer.wandb_enabled = True

        trainer._finish_wandb({'best_acc': 96.4, 'epochs_trained': 4})

        assert trainer.wandb_run is None
        finish_mock.assert_called_once()

    def test_init_wandb_passes_mode_and_config(self, test_config, monkeypatch):
        trainer = Trainer(test_config)
        trainer.config.use_wandb = True
        trainer.config.wandb_project = 'demo-project'
        trainer.config.wandb_entity = 'demo-entity'
        trainer.config.wandb_run_name = 'demo-run'
        trainer.config.wandb_tags = ['baseline']
        trainer.config.wandb_mode = 'offline'

        init_mock = MagicMock(return_value=types.SimpleNamespace(summary={}))
        fake_wandb = types.SimpleNamespace(init=init_mock)
        monkeypatch.setattr('libs.training.importlib.import_module', lambda name: fake_wandb)

        trainer._init_wandb(epochs=7, start_epoch=1, force_train=True)

        assert trainer.wandb_enabled is True
        init_mock.assert_called_once()
        kwargs = init_mock.call_args.kwargs
        assert kwargs['project'] == 'demo-project'
        assert kwargs['entity'] == 'demo-entity'
        assert kwargs['name'] == 'demo-run'
        assert kwargs['tags'] == ['baseline']
        assert kwargs['mode'] == 'offline'
        assert kwargs['config']['planned_epochs'] == 7
        assert kwargs['config']['start_epoch'] == 1
        assert kwargs['config']['force_train'] is True
