import csv
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config import config, DefaultConfigs
from dataset.dataloader import PlantDiseaseDataset, get_files
from libs.inference import InferenceManager
from models.model import get_net
from utils.utils import AverageMeter, accuracy, get_loss_function, handle_datasets


def _setup_logger(cfg: Optional[DefaultConfigs] = None) -> logging.Logger:
    cfg = cfg or config
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(cfg.paths.log_dir, "evaluation.log"), encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("Evaluation")
    logger.setLevel(logging.INFO)
    return logger


def _resolve_device(device: Optional[str]) -> torch.device:
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _auto_model_path(cfg: DefaultConfigs) -> str:
    best_model_path = os.path.join(cfg.best_weights, cfg.model_name, "0", "best_model.pth.tar")
    if os.path.exists(best_model_path):
        return best_model_path
    latest_model_path = os.path.join(cfg.weights, cfg.model_name, "0", "_latest_model.pth.tar")
    if os.path.exists(latest_model_path):
        return latest_model_path
    raise FileNotFoundError("Could not find any model weights. Please provide --model.")


def _infer_model_name(model_path: str) -> Optional[str]:
    path_norm = model_path.replace("\\", "/")
    candidates = [
        "densenet169",
        "efficientnet_b4",
        "efficientnetv2_s",
        "convnext_small",
        "convnextv2_base_384",
        "swin_transformer",
        "hybrid_model",
        "ensemble_model",
    ]
    for name in candidates:
        if f"/{name}/" in path_norm:
            return name
    return None


def _load_model(
    model_path: str,
    device: torch.device,
    model_name: Optional[str],
    cfg: DefaultConfigs,
) -> torch.nn.Module:
    model_name = model_name or cfg.model_name
    model = get_net(
        model_name=model_name,
        num_classes=cfg.num_classes,
        pretrained=False,
    )
    # SECURITY FIX: Added weights_only=True to prevent arbitrary code execution
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def _write_confusion_matrix(csv_path: str, matrix: np.ndarray, labels: List[int]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class"] + labels)
        for idx, row in enumerate(matrix):
            writer.writerow([labels[idx]] + row.tolist())


def evaluate_model(
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    device: Optional[str] = None,
    topk: int = 2,
    tta_views: Optional[int] = None,
    output_dir: Optional[str] = None,
    save_confusion: bool = True,
    save_report: bool = True,
    cfg: Optional[DefaultConfigs] = None,
) -> Dict[str, Any]:
    """评估模型并输出报告"""
    cfg = cfg or config
    logger = _setup_logger(cfg)

    model_path = model_path or _auto_model_path(cfg)
    inferred_name = _infer_model_name(model_path)
    model_name = model_name or inferred_name or cfg.model_name
    eval_device = _resolve_device(device)
    output_dir = output_dir or cfg.paths.report_dir

    os.makedirs(output_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"eval_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    if data_dir is None:
        data_dir = handle_datasets(data_type="val", cfg=cfg)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Evaluation data directory not found: {data_dir}")

    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Model architecture: {model_name}")
    logger.info(f"Evaluation data: {data_dir}")
    logger.info(f"Device: {eval_device}")
    logger.info(f"TTA views: {tta_views or cfg.tta_views}")

    model = _load_model(model_path, eval_device, model_name=model_name, cfg=cfg)
    tta_helper = InferenceManager(model_path=None, device=str(eval_device), model_name=model_name, cfg=cfg)
    tta_helper.model = model
    tta_helper.device = eval_device

    eval_files = get_files(data_dir, mode="val")
    eval_dataset = PlantDiseaseDataset(
        eval_files,
        sampling_threshold=cfg.sampling_threshold,
        sample_size=cfg.sample_size,
        seed=cfg.seed,
        img_width=cfg.img_width,
        img_height=cfg.img_height,
        use_data_aug=False,
        train=False,
        test=False,
        enable_sampling=False,
        validate_images=cfg.enable_image_validation,
        validation_workers=cfg.image_validation_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.val_batch_size if batch_size is None else batch_size,
        shuffle=False,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=eval_device.type == "cuda",
        collate_fn=lambda batch: (torch.stack([x[0] for x in batch], 0), [x[1] for x in batch]),
    )

    criterion = get_loss_function(eval_device, cfg=cfg)
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    topk_meter = AverageMeter()

    all_targets: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs = inputs.to(eval_device)
            targets_tensor = torch.tensor(targets).to(eval_device)

            probabilities = tta_helper._predict_probabilities(inputs, tta_views=tta_views)
            outputs = torch.log(probabilities.clamp_min(1e-8))
            loss = criterion(outputs, targets_tensor)

            prec1, preck = accuracy(outputs, targets_tensor, topk=(1, max(1, topk)))
            loss_meter.update(loss.item(), inputs.size(0))
            top1_meter.update(prec1.item(), inputs.size(0))
            topk_meter.update(preck.item(), inputs.size(0))

            preds = torch.argmax(probabilities, dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets)

    summary = {
        "model_path": model_path,
        "data_dir": data_dir,
        "device": str(eval_device),
        "tta_views": int(tta_views or cfg.tta_views),
        "samples": len(all_targets),
        "loss": float(loss_meter.avg),
        "top1": float(top1_meter.avg),
        "topk": float(topk_meter.avg),
    }

    summary_path = os.path.join(run_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_report:
        report = classification_report(
            all_targets,
            all_preds,
            output_dict=True,
            zero_division=0,
        )
        report_path = os.path.join(run_dir, "classification_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    if save_confusion:
        labels = list(range(cfg.num_classes))
        matrix = confusion_matrix(all_targets, all_preds, labels=labels)
        cm_path = os.path.join(run_dir, "confusion_matrix.csv")
        _write_confusion_matrix(cm_path, matrix, labels)

    logger.info(f"Evaluation summary saved to: {summary_path}")
    return summary
