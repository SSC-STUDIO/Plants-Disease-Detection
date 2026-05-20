"""CLI-focused tests for main.py helpers and parser."""
import json
from argparse import Namespace

from config import DefaultConfigs
from libs.checkpoint_utils import infer_num_classes_from_checkpoint
from libs.evaluation import (
    _infer_num_classes_from_labeled_dir,
    _sync_num_classes_for_evaluation,
)
from main import apply_train_overrides, export_config, setup_parser, train_pipeline


def test_export_config_returns_payload():
    payload = export_config()
    assert isinstance(payload, dict)
    assert payload["model_name"]
    assert payload["img_height"] > 0
    assert payload["img_width"] > 0


def test_export_config_writes_file(temp_dir):
    output_path = temp_dir / "config.json"
    payload = export_config(str(output_path))
    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["model_name"] == payload["model_name"]


def test_setup_parser_supports_export_command():
    parser = setup_parser()
    args = parser.parse_args([
        "export",
        "--model",
        "checkpoint.pth.tar",
        "--input-size",
        "320",
        "320",
        "--dynamic-axes",
        "--verify",
    ])
    assert args.command == "export"
    assert args.model == "checkpoint.pth.tar"
    assert args.input_size == [320, 320]
    assert args.dynamic_axes is True
    assert args.verify is True


def test_setup_parser_supports_evaluate_output_file():
    parser = setup_parser()
    args = parser.parse_args([
        "evaluate",
        "--model",
        "checkpoint.pth.tar",
        "--output",
        "reports/eval.json",
    ])
    assert args.command == "evaluate"
    assert args.model == "checkpoint.pth.tar"
    assert args.output == "reports/eval.json"


def test_train_parser_supports_wandb_arguments():
    parser = setup_parser()
    args = parser.parse_args([
        "train",
        "--wandb",
        "--wandb-project",
        "demo-project",
        "--wandb-entity",
        "demo-entity",
        "--wandb-run-name",
        "run-1",
        "--wandb-tags",
        "baseline",
        "debug",
        "--wandb-mode",
        "offline",
    ])
    assert args.command == "train"
    assert args.wandb is True
    assert args.wandb_project == "demo-project"
    assert args.wandb_entity == "demo-entity"
    assert args.wandb_run_name == "run-1"
    assert args.wandb_tags == ["baseline", "debug"]
    assert args.wandb_mode == "offline"


def test_train_parser_supports_no_image_validation():
    parser = setup_parser()
    args = parser.parse_args(["train", "--no-image-validation"])

    assert args.command == "train"
    assert args.no_image_validation is True


def test_apply_train_overrides_updates_wandb_settings(temp_dir):
    cfg = DefaultConfigs()
    cfg.paths.base_dir = str(temp_dir).replace('\\', '/')
    args = Namespace(
        epochs=None,
        model=None,
        batch_size=None,
        lr=None,
        dataset_path=None,
        enable_augmentation=False,
        disable_augmentation=False,
        merge_augmented=False,
        no_merge_augmented=False,
        optimizer=None,
        weight_decay=None,
        no_lookahead=False,
        scheduler=None,
        warmup_epochs=None,
        warmup_factor=None,
        no_mixup=False,
        mixup_alpha=None,
        cutmix_prob=None,
        no_random_erasing=False,
        enable_weighted_sampler=False,
        disable_weighted_sampler=False,
        weighted_sampler_power=None,
        tta_views=None,
        max_train_batches=None,
        max_val_batches=None,
        no_image_validation=False,
        no_early_stopping=False,
        patience=None,
        no_amp=False,
        gradient_clip_val=None,
        no_ema=False,
        ema_decay=None,
        wandb=True,
        no_wandb=False,
        wandb_project="demo-project",
        wandb_entity="team-alpha",
        wandb_run_name="exp-42",
        wandb_tags=["baseline", "nightly"],
        wandb_mode="offline",
        device=None,
        gpus=None,
        seed=None,
        label_smoothing=None,
        no_gradient_checkpointing=False,
    )

    apply_train_overrides(args, cfg=cfg)

    assert cfg.use_wandb is True
    assert cfg.wandb_project == "demo-project"
    assert cfg.wandb_entity == "team-alpha"
    assert cfg.wandb_run_name == "exp-42"
    assert cfg.wandb_tags == ["baseline", "nightly"]
    assert cfg.wandb_mode == "offline"


def test_apply_train_overrides_prefers_enable_when_wandb_flags_conflict(temp_dir):
    cfg = DefaultConfigs()
    cfg.paths.base_dir = str(temp_dir).replace('\\', '/')
    cfg.use_wandb = False
    args = Namespace(
        epochs=None,
        model=None,
        batch_size=None,
        lr=None,
        dataset_path=None,
        enable_augmentation=False,
        disable_augmentation=False,
        merge_augmented=False,
        no_merge_augmented=False,
        optimizer=None,
        weight_decay=None,
        no_lookahead=False,
        scheduler=None,
        warmup_epochs=None,
        warmup_factor=None,
        no_mixup=False,
        mixup_alpha=None,
        cutmix_prob=None,
        no_random_erasing=False,
        enable_weighted_sampler=False,
        disable_weighted_sampler=False,
        weighted_sampler_power=None,
        tta_views=None,
        max_train_batches=None,
        max_val_batches=None,
        no_image_validation=False,
        no_early_stopping=False,
        patience=None,
        no_amp=False,
        gradient_clip_val=None,
        no_ema=False,
        ema_decay=None,
        wandb=True,
        no_wandb=True,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_tags=None,
        wandb_mode=None,
        device=None,
        gpus=None,
        seed=None,
        label_smoothing=None,
        no_gradient_checkpointing=False,
    )

    apply_train_overrides(args, cfg=cfg)

    assert cfg.use_wandb is True


def test_apply_train_overrides_updates_debug_batch_limits(temp_dir):
    cfg = DefaultConfigs()
    cfg.paths.base_dir = str(temp_dir).replace('\\', '/')
    args = Namespace(
        epochs=None,
        model=None,
        batch_size=None,
        lr=None,
        dataset_path=None,
        enable_augmentation=False,
        disable_augmentation=False,
        merge_augmented=False,
        no_merge_augmented=False,
        optimizer=None,
        weight_decay=None,
        no_lookahead=False,
        scheduler=None,
        warmup_epochs=None,
        warmup_factor=None,
        no_mixup=False,
        mixup_alpha=None,
        cutmix_prob=None,
        no_random_erasing=False,
        enable_weighted_sampler=False,
        disable_weighted_sampler=False,
        weighted_sampler_power=None,
        tta_views=None,
        max_train_batches=2,
        max_val_batches=1,
        no_image_validation=True,
        no_early_stopping=False,
        patience=None,
        no_amp=False,
        gradient_clip_val=None,
        no_ema=False,
        ema_decay=None,
        wandb=False,
        no_wandb=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_tags=None,
        wandb_mode=None,
        device=None,
        gpus=None,
        seed=None,
        label_smoothing=None,
        no_gradient_checkpointing=False,
    )

    apply_train_overrides(args, cfg=cfg)

    assert cfg.max_train_batches == 2
    assert cfg.max_val_batches == 1
    assert cfg.enable_image_validation is False


def test_train_pipeline_no_prepare_skips_test_data_preparation(monkeypatch):
    calls = {"prepare_test": 0, "train": 0}

    def fake_prepare_test_data(*args, **kwargs):
        calls["prepare_test"] += 1
        return True

    def fake_load_train_function():
        def fake_train(cfg, force_train=False):
            calls["train"] += 1
            return {"ok": True}

        return fake_train

    monkeypatch.setattr("main.has_image_files", lambda path: False)
    monkeypatch.setattr("main.prepare_test_data", fake_prepare_test_data)
    monkeypatch.setattr("main.load_train_function", fake_load_train_function)
    monkeypatch.setattr("main.apply_train_overrides", lambda args, cfg=None: None)

    train_pipeline(
        Namespace(
            prepare=False,
            no_prepare=True,
            force_train=True,
            cleanup=False,
            dataset_path="custom-data",
        )
    )

    assert calls["prepare_test"] == 0
    assert calls["train"] == 1


def test_evaluation_syncs_num_classes_from_labeled_dir(temp_dir):
    for label in ("0", "116", "not-a-class"):
        (temp_dir / label).mkdir()

    cfg = DefaultConfigs()
    cfg.num_classes = 59

    assert _infer_num_classes_from_labeled_dir(str(temp_dir)) == 117
    _sync_num_classes_for_evaluation(
        model_path=None,
        data_dir=str(temp_dir),
        cfg=cfg,
        logger=__import__("logging").getLogger("test"),
    )

    assert cfg.num_classes == 117


def test_evaluation_infers_num_classes_from_checkpoint(temp_dir):
    import torch

    checkpoint_path = temp_dir / "model.pth.tar"
    torch.save(
        {
            "state_dict": {
                "classifier.2.weight": torch.zeros(117, 768),
                "classifier.2.bias": torch.zeros(117),
            }
        },
        checkpoint_path,
    )

    assert infer_num_classes_from_checkpoint(str(checkpoint_path)) == 117
