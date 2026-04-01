#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import time
import glob
import json
import platform
from dataclasses import asdict
from typing import Dict, Any, List, Optional, Tuple

# 将项目根目录添加到路径
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

# 导入配置
from config import config, paths

# 确保处理器的级别正确设置
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.setLevel(logging.INFO)

logger = logging.getLogger('Main')


def dedupe_paths(paths_list):
    """保持顺序去重路径列表。"""
    unique_paths = []
    seen = set()

    for path in paths_list:
        key = os.path.normcase(os.path.abspath(normalize_path(path)))
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)

    return unique_paths


def load_predict_function():
    """按需加载推理函数，避免非推理命令被重依赖阻塞。"""
    try:
        from libs.inference import predict
    except ModuleNotFoundError as exc:
        logger.error(
            "Inference dependencies are unavailable while loading predict pipeline: %s",
            exc,
        )
        raise

    return predict


def load_train_function():
    """按需加载训练函数，缺依赖时给出可读错误而不是抛栈。"""
    try:
        from libs.training import train_model
    except ModuleNotFoundError as exc:
        logger.error(
            "Training dependencies are unavailable while loading train pipeline: %s",
            exc,
        )
        raise

    return train_model

def normalize_path(path: Optional[str]) -> Optional[str]:
    """轻量级路径规范化，避免在轻量命令中加载重依赖。"""
    if path is None:
        return None
    normalized = os.path.normpath(path)
    return normalized.replace('\\', '/')


def get_image_extensions_local(extensions: Optional[Tuple[str, ...]] = None) -> Tuple[str, ...]:
    """获取规范化后的图像扩展名列表（小写，带点）。"""
    exts = extensions or getattr(config, "image_extensions", ())
    normalized: List[str] = []
    for ext in exts:
        if not ext:
            continue
        ext = ext.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in normalized:
            normalized.append(ext)
    return tuple(normalized)


def get_image_glob_patterns_local(extensions: Optional[Tuple[str, ...]] = None) -> Tuple[str, ...]:
    """根据扩展名生成 glob 模式（同时包含大小写）。"""
    exts = get_image_extensions_local(extensions)
    patterns: List[str] = []
    for ext in exts:
        patterns.append(f"*{ext}")
        patterns.append(f"*{ext.upper()}")
    return tuple(dict.fromkeys(patterns))


IMAGE_EXTENSIONS = get_image_extensions_local()
COPY_PATTERNS = get_image_glob_patterns_local(IMAGE_EXTENSIONS)


def get_project_version() -> str:
    """读取 pyproject.toml 中的版本信息。"""
    pyproject_path = os.path.join(project_dir, "pyproject.toml")
    if not os.path.exists(pyproject_path):
        return "unknown"
    try:
        import tomllib  # Python 3.11+
    except Exception:
        return "unknown"
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("version", "unknown")
    except Exception:
        return "unknown"


def get_version_payload() -> Dict[str, Any]:
    """收集项目与运行环境版本信息。"""
    payload: Dict[str, Any] = {
        "project": "plants-disease-detection",
        "version": get_project_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        payload.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        })
    except Exception:
        payload.update({
            "torch": None,
            "cuda_available": False,
            "cuda_version": None,
        })
    return payload


def print_version_info(simple: bool = False) -> None:
    """输出版本信息到 stdout。"""
    if simple:
        version = get_project_version()
        print(f"plants-disease-detection {version}")
        return
    payload = get_version_payload()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def export_config(output_path: Optional[str] = None, pretty: bool = True) -> Dict[str, Any]:
    """导出当前配置到 JSON 文件（或 stdout）。"""
    payload = asdict(config)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2 if pretty else None)
    return payload


def list_image_files(directory: str) -> List[str]:
    """列出目录中的图像文件（按扩展名过滤）。"""
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS)]


def has_image_files(path: str) -> bool:
    """判断路径是否包含至少一张图像。"""
    if not os.path.exists(path):
        return False
    if os.path.isfile(path):
        return path.lower().endswith(IMAGE_EXTENSIONS)
    return len(list_image_files(path)) > 0


def get_dataset_search_roots(custom_dataset_path: Optional[str]) -> List[str]:
    """根据自定义路径与默认数据目录构造搜索根目录列表。"""
    roots: List[str] = []
    if custom_dataset_path:
        custom_root = custom_dataset_path
        if os.path.isfile(custom_root):
            custom_root = os.path.dirname(custom_root)
        roots.append(normalize_path(custom_root))
    roots.append(normalize_path(paths.data_dir))
    return dedupe_paths([root for root in roots if root])


def find_test_archives(search_roots: List[str], cfg=None) -> List[str]:
    """在给定根目录中查找测试集压缩包文件。"""
    cfg = cfg or config
    test_files: List[str] = []
    for root in search_roots:
        for ext in cfg.supported_dataset_formats:
            test_files.extend(glob.glob(os.path.join(root, f"*test*{ext}")))
            test_files.extend(glob.glob(os.path.join(root, f"*TEST*{ext}")))
            test_files.extend(glob.glob(os.path.join(root, "**", f"*test*{ext}"), recursive=True))
            test_files.extend(glob.glob(os.path.join(root, "**", f"*TEST*{ext}"), recursive=True))
    return dedupe_paths(test_files)


def find_test_archives_by_pattern(search_roots: List[str], pattern: Optional[str]) -> List[str]:
    """使用显式模式查找测试集压缩包。"""
    if not pattern:
        return []
    test_files: List[str] = []
    for root in search_roots:
        test_files.extend(glob.glob(os.path.join(root, pattern)))
        test_files.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return dedupe_paths(test_files)


def collect_test_image_dirs(extract_to: str) -> List[str]:
    """从解压目录中收集可能的测试图像目录。"""
    potential_image_dirs: List[str] = []
    potential_image_dirs_abs: List[str] = []

    def add_dir(path: str) -> None:
        if os.path.exists(path) and os.path.isdir(path):
            abs_path = os.path.abspath(path)
            if abs_path not in potential_image_dirs_abs:
                potential_image_dirs.append(path)
                potential_image_dirs_abs.append(abs_path)

    # 1. 标准 images 目录
    add_dir(normalize_path(os.path.join(extract_to, "images")))

    # 2. 可能的测试子目录
    test_subdirs = glob.glob(os.path.join(extract_to, "AgriculturalDisease_test*"))
    for subdir in test_subdirs:
        add_dir(os.path.join(subdir, "images"))

    # 3. 递归搜索包含图像的目录
    for root, _, files in os.walk(extract_to):
        root_abs = os.path.abspath(root)
        if any(root_abs == dir_abs or root_abs.startswith(dir_abs + os.sep) for dir_abs in potential_image_dirs_abs):
            continue
        if os.path.basename(root) == "labels":
            continue
        if any(file.lower().endswith(IMAGE_EXTENSIONS) for file in files):
            add_dir(root)

    return potential_image_dirs


def copy_test_images(data_prep, source_dirs: List[str], dest_dir: str) -> int:
    """复制测试图像到目标目录并返回总数量。"""
    copied_count = 0
    for image_dir in source_dirs:
        count = 0
        for pattern in COPY_PATTERNS:
            count += data_prep.copy_files_to_folder(image_dir, dest_dir, file_pattern=pattern)
        copied_count += count
        logger.info(f"Copied {count} image files from {image_dir} to {dest_dir}")
    return copied_count


def prepare_test_data(custom_dataset_path: Optional[str] = None, cfg=None) -> bool:
    """准备测试数据（解压与复制到测试目录）。"""
    from dataset.data_prep import DataPreparation

    cfg = cfg or config
    data_prep = DataPreparation(config_obj=cfg)
    cfg.merge_test_datasets = True
    logger.info("Test dataset merging enabled")

    search_roots = get_dataset_search_roots(custom_dataset_path)
    test_files = find_test_archives(search_roots, cfg=cfg)
    if not test_files:
        test_files = find_test_archives_by_pattern(search_roots, cfg.test_name_pattern)

    if not test_files:
        logger.error("Could not find any test dataset files")
        return False

    logger.info(f"Found the following test dataset files: {test_files}")

    extract_to = normalize_path(os.path.join(paths.temp_dataset_dir, "AgriculturalDisease_testset"))
    os.makedirs(extract_to, exist_ok=True)

    for test_file in test_files:
        logger.info(f"Extracting test dataset file: {test_file}")
        if data_prep.extract_zip_file(test_file, extract_to):
            logger.info(f"Successfully extracted {test_file}")
        else:
            logger.error(f"Failed to extract {test_file}")

    os.makedirs(paths.test_images_dir, exist_ok=True)
    potential_image_dirs = collect_test_image_dirs(extract_to)

    if not potential_image_dirs:
        logger.error("No test image directory found in the extracted directory")
        return False

    logger.info(f"Found the following test image directories: {potential_image_dirs}")
    copied_count = copy_test_images(data_prep, potential_image_dirs, paths.test_images_dir)

    if copied_count > 0:
        logger.info(f"Successfully prepared test data: copied {copied_count} image files in total")
        logger.info("Cleaning up temporary test data files...")
        data_prep.cleanup_temp_files(force=True)
        return True

    logger.warning("Could not find any test image files")
    return False


def apply_train_overrides(args: argparse.Namespace, cfg=None) -> None:
    """根据命令行参数覆盖训练配置。"""
    cfg = cfg or config
    def set_if_not_none(arg_name: str, config_attr: str, label: str) -> None:
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None and value != "":
                setattr(cfg, config_attr, value)
                logger.info(f"Setting {label} to {value}")

    def set_if_true(arg_name: str, config_attr: str, label: str) -> None:
        if getattr(args, arg_name, False):
            setattr(cfg, config_attr, False)
            logger.info(label)

    def resolve_toggle(enable_attr: str, disable_attr: str, config_attr: str, enable_msg: str, disable_msg: str) -> None:
        enable_flag = getattr(args, enable_attr, False)
        disable_flag = getattr(args, disable_attr, False)
        if enable_flag and disable_flag:
            logger.warning(f"Both --{enable_attr.replace('_', '-')} and --{disable_attr.replace('_', '-')} specified, using --{enable_attr.replace('_', '-')}")
        if enable_flag:
            setattr(cfg, config_attr, True)
            logger.info(enable_msg)
        elif disable_flag:
            setattr(cfg, config_attr, False)
            logger.info(disable_msg)

    set_if_not_none('epochs', 'epoch', 'epochs')
    set_if_not_none('model', 'model_name', 'model')
    set_if_not_none('batch_size', 'train_batch_size', 'batch size')
    set_if_not_none('lr', 'lr', 'learning rate')

    if getattr(args, 'dataset_path', None):
        cfg.dataset_path = args.dataset_path
        cfg.use_custom_dataset_path = True
        logger.info(f"Using custom dataset path: {cfg.dataset_path}")

    resolve_toggle(
        'enable_augmentation',
        'disable_augmentation',
        'use_data_aug',
        'Data augmentation enabled for training',
        'Data augmentation explicitly disabled for training',
    )

    if hasattr(args, 'merge_augmented') and hasattr(args, 'no_merge_augmented'):
        if args.merge_augmented and args.no_merge_augmented:
            logger.warning("Both --merge-augmented and --no-merge-augmented specified, using --merge-augmented")
            cfg.merge_augmented_data = True
        elif args.merge_augmented:
            cfg.merge_augmented_data = True
            logger.info("Will merge augmented data with original training data")
        elif args.no_merge_augmented:
            cfg.merge_augmented_data = False
            logger.info("Will not merge augmented data with original training data")

    set_if_not_none('optimizer', 'optimizer', 'optimizer')
    set_if_not_none('weight_decay', 'weight_decay', 'weight decay')
    set_if_true('no_lookahead', 'use_lookahead', 'Disabling Lookahead optimizer wrapper')

    set_if_not_none('scheduler', 'scheduler', 'LR scheduler')
    set_if_not_none('warmup_epochs', 'warmup_epochs', 'warmup epochs')
    set_if_not_none('warmup_factor', 'warmup_factor', 'warmup factor')

    set_if_true('no_mixup', 'use_mixup', 'Disabling Mixup data augmentation')
    set_if_not_none('mixup_alpha', 'mixup_alpha', 'Mixup alpha')
    set_if_not_none('cutmix_prob', 'cutmix_prob', 'CutMix probability')
    set_if_true('no_random_erasing', 'use_random_erasing', 'Disabling random erasing augmentation')
    resolve_toggle(
        'enable_weighted_sampler',
        'disable_weighted_sampler',
        'use_weighted_sampler',
        'Weighted sampler enabled for training',
        'Weighted sampler disabled for training',
    )
    set_if_not_none('weighted_sampler_power', 'weighted_sampler_power', 'weighted sampler power')
    set_if_not_none('tta_views', 'tta_views', 'TTA views')

    set_if_true('no_early_stopping', 'use_early_stopping', 'Disabling early stopping')
    set_if_not_none('patience', 'early_stopping_patience', 'early stopping patience')

    set_if_true('no_amp', 'use_amp', 'Disabling automatic mixed precision training')
    set_if_not_none('gradient_clip_val', 'gradient_clip_val', 'gradient clipping value')

    set_if_true('no_ema', 'use_ema', 'Disabling Exponential Moving Average (EMA)')
    set_if_not_none('ema_decay', 'ema_decay', 'EMA decay rate')

    resolve_toggle(
        'wandb',
        'no_wandb',
        'use_wandb',
        'Weights & Biases experiment tracking enabled',
        'Weights & Biases experiment tracking disabled',
    )
    set_if_not_none('wandb_project', 'wandb_project', 'wandb project')
    set_if_not_none('wandb_entity', 'wandb_entity', 'wandb entity')
    set_if_not_none('wandb_run_name', 'wandb_run_name', 'wandb run name')
    set_if_not_none('wandb_mode', 'wandb_mode', 'wandb mode')
    if hasattr(args, 'wandb_tags') and args.wandb_tags is not None:
        cfg.wandb_tags = [tag for tag in args.wandb_tags if tag]
        logger.info(f"Setting wandb tags to {cfg.wandb_tags}")

    set_if_not_none('device', 'device', 'device')
    set_if_not_none('gpus', 'gpus', 'GPUs')

    set_if_not_none('seed', 'seed', 'random seed')
    set_if_not_none('label_smoothing', 'label_smoothing', 'label smoothing')
    set_if_true('no_gradient_checkpointing', 'use_gradient_checkpointing', 'Disabling gradient checkpointing')

def add_train_arguments(train_parser: argparse.ArgumentParser) -> None:
    """添加训练相关的参数到解析器
    
    参数:
        train_parser: 训练命令的参数解析器
    """
    # 基本训练参数
    train_parser.add_argument('--epochs', type=int, 
                             help='Number of epochs (default: from config)')
    train_parser.add_argument('--model', type=str, 
                             help='Model name (default: from config)')
    train_parser.add_argument('--batch-size', type=int, 
                             help='Batch size (default: from config)')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    
    # 数据准备标志
    train_parser.add_argument('--prepare', action='store_true', 
                             help='Run data preparation before training')
    train_parser.add_argument('--no-prepare', action='store_true', 
                             help='Skip data preparation before training')
    train_parser.add_argument('--force-train', action='store_true', 
                             help='Force retraining even if model is already trained')
    train_parser.add_argument('--merge-augmented', action='store_true', 
                             help='Merge augmented data with original training data')
    train_parser.add_argument('--no-merge-augmented', action='store_true', 
                             help='Do not merge augmented data with original training data')
    train_parser.add_argument('--dataset-path', type=str, 
                             help='Custom path to dataset files or directory')
    train_parser.add_argument('--cleanup', action='store_true', 
                             help='Clean up temporary files after processing and training')
    train_parser.add_argument('--force-cleanup', action='store_true', 
                             help='Force cleanup without asking for confirmation')
    
    # 优化器选项
    train_parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd', 'ranger'],
                             help='Optimizer selection (default: from config)')
    train_parser.add_argument('--weight-decay', type=float,
                             help='Weight decay for optimizer (default: from config)')
    train_parser.add_argument('--no-lookahead', action='store_true',
                             help='Disable Lookahead optimizer wrapper')
    
    # 学习率调度器选项
    train_parser.add_argument('--scheduler', type=str, choices=['step', 'cosine', 'onecycle'],
                             help='Learning rate scheduler (default: from config)')
    train_parser.add_argument('--warmup-epochs', type=int,
                             help='Number of warmup epochs (default: from config)')
    train_parser.add_argument('--warmup-factor', type=float,
                             help='Warmup factor (default: from config)')
    
    # Mixup 和 CutMix 选项
    train_parser.add_argument('--no-mixup', action='store_true',
                             help='Disable Mixup data augmentation')
    train_parser.add_argument('--mixup-alpha', type=float,
                             help='Mixup alpha parameter (default: from config)')
    train_parser.add_argument('--cutmix-prob', type=float,
                             help='CutMix probability (default: from config)')
    train_parser.add_argument('--no-random-erasing', action='store_true',
                             help='Disable random erasing augmentation')
    train_parser.add_argument('--enable-weighted-sampler', action='store_true',
                             help='Enable weighted random sampling for class balancing')
    train_parser.add_argument('--disable-weighted-sampler', action='store_true',
                             help='Disable weighted random sampling for class balancing')
    train_parser.add_argument('--weighted-sampler-power', type=float,
                             help='Exponent applied to inverse class frequency for weighted sampling')
    train_parser.add_argument('--disable-augmentation', action='store_true',
                             help='Disable all data augmentation (overrides configured default)')
    train_parser.add_argument('--enable-augmentation', action='store_true',
                             help='Enable all data augmentation (overrides configured default)')
    
    # 早停参数
    train_parser.add_argument('--no-early-stopping', action='store_true',
                             help='Disable early stopping')
    train_parser.add_argument('--patience', type=int,
                             help='Early stopping patience (default: from config)')
    
    # 混合精度选项
    train_parser.add_argument('--no-amp', action='store_true',
                             help='Disable automatic mixed precision training')
    
    # 梯度裁剪选项
    train_parser.add_argument('--gradient-clip-val', type=float,
                             help='Gradient clipping value (default: from config)')
    
    # EMA 选项
    train_parser.add_argument('--no-ema', action='store_true',
                             help='Disable Exponential Moving Average (EMA)')
    train_parser.add_argument('--ema-decay', type=float,
                             help='EMA decay rate (default: from config)')

    # Weights & Biases 选项
    train_parser.add_argument('--wandb', action='store_true',
                             help='Enable Weights & Biases experiment tracking')
    train_parser.add_argument('--no-wandb', action='store_true',
                             help='Disable Weights & Biases experiment tracking')
    train_parser.add_argument('--wandb-project', type=str,
                             help='Weights & Biases project name (default: from config)')
    train_parser.add_argument('--wandb-entity', type=str,
                             help='Weights & Biases entity/team name (default: from config)')
    train_parser.add_argument('--wandb-run-name', type=str,
                             help='Weights & Biases run name (default: auto-generated)')
    train_parser.add_argument('--wandb-tags', nargs='*',
                             help='Weights & Biases tags for the run')
    train_parser.add_argument('--wandb-mode', type=str, choices=['online', 'offline', 'disabled'],
                             help='Weights & Biases mode (default: from config)')

    # 设备选项
    train_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                             help='Device to use for training (default: from config)')
    train_parser.add_argument('--gpus', type=str,
                             help='GPU device IDs to use, comma separated (e.g., "0,1")')
    
    # 高级选项
    train_parser.add_argument('--seed', type=int,
                             help='Random seed for reproducibility (default: from config)')
    train_parser.add_argument('--label-smoothing', type=float,
                             help='Label smoothing coefficient (default: from config)')
    train_parser.add_argument('--no-gradient-checkpointing', action='store_true',
                             help='Disable gradient checkpointing')

def setup_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器
    
    返回:
        配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--version', action='store_true',
                        help='Show project version and exit')
    
    # 为不同的命令创建子解析器
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # 数据准备命令
    prep_parser = subparsers.add_parser('prepare', 
                                        help='Prepare datasets (extract, process, augment)')
    prep_parser.add_argument('--extract', action='store_true', 
                            help='Extract dataset archives')
    prep_parser.add_argument('--process', action='store_true', 
                            help='Process images from temp directory into training directory')
    prep_parser.add_argument('--augment', action='store_true', 
                            help='Perform data augmentation')
    prep_parser.add_argument('--status', action='store_true', 
                            help='Check data preparation status')
    prep_parser.add_argument('--all', action='store_true', 
                            help='Perform all data preparation steps')
    prep_parser.add_argument('--merge', choices=['train', 'test', 'val', 'all'], 
                            help='Merge datasets')
    prep_parser.add_argument('--dataset-path', type=str, 
                            help='Custom path to dataset files or directory')
    prep_parser.add_argument('--merge-augmented', action='store_true', 
                            help='Merge augmented data with original data')
    prep_parser.add_argument('--no-merge-augmented', action='store_true', 
                            help='Do not merge augmented data with original data')
    prep_parser.add_argument('--cleanup', action='store_true', 
                            help='Clean up temporary files after processing')
    prep_parser.add_argument('--force-cleanup', action='store_true', 
                            help='Force cleanup without asking for confirmation')
    prep_parser.add_argument('--disable-augmentation', action='store_true',
                            help='Disable data augmentation (overrides configured default)')
    prep_parser.add_argument('--enable-augmentation', action='store_true',
                            help='Enable data augmentation (overrides configured default)')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', 
                                         help='Train a model')
    # 使用专用函数添加训练参数
    add_train_arguments(train_parser)
    
    # 推理命令
    infer_parser = subparsers.add_parser('predict', 
                                         help='Run inference with a trained model')
    infer_parser.add_argument('--model', type=str, 
                             help=f'Path to model weights file (default: auto-detect best model)')
    infer_parser.add_argument('--model-name', type=str,
                             help='Model architecture name (override configured default)')
    infer_parser.add_argument('--input', type=str, 
                             help=f'Path to image folder or single image (default: {paths.test_images_dir})')
    infer_parser.add_argument('--output', type=str, 
                             help=f'Output JSON file path (default: {paths.prediction_file})')
    infer_parser.add_argument('--batch-size', type=int,
                             help='Batch size for inference on directories')
    infer_parser.add_argument('--num-workers', type=int,
                             help='Number of workers for inference dataloader')
    infer_parser.add_argument('--topk', type=int, default=3,
                             help='Top-k predictions to include')
    infer_parser.add_argument('--save-probs', action='store_true',
                             help='Include full probability vector in output (only for output-format=full)')
    infer_parser.add_argument('--output-format', choices=['submit', 'full'], default='submit',
                             help='Output format: submit (image_id + class) or full (with confidence/topk)')
    infer_parser.add_argument('--confidence-threshold', type=float,
                             help='Mark predictions below this confidence as low_confidence')
    infer_parser.add_argument('--tta-views', type=int, choices=[1, 2, 3, 4],
                             help='Number of test-time augmentation views to average')
    infer_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                             help='Device for inference (default: auto)')

    # 评估命令
    eval_parser = subparsers.add_parser('evaluate',
                                        help='Evaluate a trained model on labeled data')
    eval_parser.add_argument('--model', type=str,
                             help='Path to model weights file (default: auto-detect best model)')
    eval_parser.add_argument('--model-name', type=str,
                             help='Model architecture name (override configured default)')
    eval_parser.add_argument('--data', type=str,
                             help='Path to labeled dataset directory (default: auto-detect val set)')
    eval_parser.add_argument('--batch-size', type=int,
                             help='Batch size for evaluation')
    eval_parser.add_argument('--num-workers', type=int,
                             help='Number of workers for evaluation dataloader')
    eval_parser.add_argument('--topk', type=int, default=2,
                             help='Top-k accuracy to report')
    eval_parser.add_argument('--tta-views', type=int, choices=[1, 2, 3, 4],
                             help='Number of test-time augmentation views to average during evaluation')
    eval_parser.add_argument('--output-dir', type=str,
                             help=f'Output directory for evaluation reports (default: {paths.report_dir})')
    eval_parser.add_argument('--no-confusion', action='store_true',
                             help='Skip confusion matrix output')
    eval_parser.add_argument('--no-report', action='store_true',
                             help='Skip classification report output')
    eval_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                             help='Device for evaluation (default: auto)')

    # 数据集统计命令
    stats_parser = subparsers.add_parser('stats',
                                         help='Summarize dataset statistics')
    stats_parser.add_argument('--data', type=str,
                              help='Path to dataset directory (default: auto-detect train set)')
    stats_parser.add_argument('--top', type=int, default=10,
                              help='Top-N classes to include in summary')
    stats_parser.add_argument('--output', type=str,
                              help='Output JSON file path for dataset stats')

    # 版本信息命令
    version_parser = subparsers.add_parser('version', help='Show detailed version info')
    version_parser.add_argument('--simple', action='store_true',
                                help='Only show project name and version')

    # 配置导出命令
    config_parser = subparsers.add_parser('config', help='Export current config to JSON')
    config_parser.add_argument('--output', type=str,
                               help='Output JSON file path (default: stdout)')
    config_parser.add_argument('--compact', action='store_true',
                               help='Disable pretty JSON formatting')

    # 模型列表命令
    models_parser = subparsers.add_parser('models', help='List available model architectures')
    models_parser.add_argument('--json', action='store_true',
                               help='Output as JSON list')

    # 模型导出命令
    export_parser = subparsers.add_parser('export', help='Export trained model to ONNX format')
    export_parser.add_argument('--model', type=str, required=True,
                               help='Path to model weights file')
    export_parser.add_argument('--model-name', type=str,
                               help='Model architecture name (override configured default)')
    export_parser.add_argument('--output', type=str,
                               help='Output ONNX file path (default: model_name.onnx)')
    export_parser.add_argument('--input-size', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'),
                               help=f'Input image size (default: {config.img_height} {config.img_width})')
    export_parser.add_argument('--opset', type=int, default=11,
                               help='ONNX opset version (default: 11)')
    export_parser.add_argument('--dynamic-axes', action='store_true',
                               help='Enable dynamic batch size in exported model')
    export_parser.add_argument('--verify', action='store_true',
                               help='Verify exported model with sample inference')

    return parser

def prepare_data(args: argparse.Namespace, cfg=None) -> Dict[str, Any]:
    """运行数据准备流程
    
    参数:
        args: 命令行参数
        
    返回:
        数据准备结果字典
    """
    logger.info("Starting data preparation")
    cfg = cfg or config

    # 延迟导入重依赖，避免轻量命令启动慢
    from dataset.data_prep import setup_data
    
    # 确定要执行的操作
    extract = args.extract if hasattr(args, 'extract') else False
    process = args.process if hasattr(args, 'process') else False
    augment = args.augment if hasattr(args, 'augment') else False
    status = args.status if hasattr(args, 'status') else False
    merge = args.merge if hasattr(args, 'merge') else None
    
    # 如果指定了--all，执行所有操作
    if hasattr(args, 'all') and args.all:
        extract = process = augment = status = True
        merge = "all"
    
    # 获取其他参数
    custom_dataset_path = getattr(args, 'dataset_path', None)
    
    # 处理merge_augmented和no_merge_augmented
    merge_augmented = None
    if hasattr(args, 'merge_augmented') and args.merge_augmented:
        merge_augmented = True
        if hasattr(args, 'no_merge_augmented') and args.no_merge_augmented:
            logger.warning("Both --merge-augmented and --no-merge-augmented specified, using --merge-augmented")
    elif hasattr(args, 'no_merge_augmented') and args.no_merge_augmented:
        merge_augmented = False
    
    # 处理数据增强选项
    if hasattr(args, 'enable_augmentation') and args.enable_augmentation:
        if hasattr(args, 'disable_augmentation') and args.disable_augmentation:
            logger.warning("Both --enable-augmentation and --disable-augmentation specified, using --enable-augmentation")
        cfg.use_data_aug = True
        logger.info("Data augmentation enabled for training")
    elif hasattr(args, 'disable_augmentation') and args.disable_augmentation:
        cfg.use_data_aug = False
        logger.info("Data augmentation explicitly disabled for training")
    else:
        state = "enabled" if cfg.use_data_aug else "disabled"
        logger.info(f"Data augmentation is {state} for training (cfg.use_data_aug={cfg.use_data_aug})")
    
    # 获取清理临时文件的参数
    cleanup_temp = getattr(args, 'cleanup', False)
    force_cleanup = getattr(args, 'force_cleanup', False)
    
    # 运行数据准备
    result = setup_data(
        extract=extract,
        process=process, 
        augment=augment,
        status=status,
        merge=merge,
        cleanup_temp=cleanup_temp,
        custom_dataset_path=custom_dataset_path,
        merge_augmented=merge_augmented,
        force_cleanup=force_cleanup,
        config_obj=cfg,
    )
    
    return result

def train_pipeline(args: argparse.Namespace) -> None:
    """训练模型流程
    
    参数:
        args: 命令行参数
    """
    start_time = time.time()
    
    # 配置日志
    logger.info("Starting training pipeline")
    
    # 首先检查是否需要数据准备
    should_prepare = getattr(args, 'prepare', False)
    if getattr(args, 'no_prepare', False):
        if should_prepare:
            logger.warning("Both --prepare and --no-prepare specified, using --no-prepare")
        should_prepare = False

    if should_prepare:
        logger.info("Running data preparation before training")
        # Create prepare data arguments
        prepare_args = argparse.Namespace(
            extract=True,
            process=True,
            augment=True,
            status=True,
            merge="all",
            dataset_path=getattr(args, 'dataset_path', None),
            merge_augmented=getattr(args, 'merge_augmented', None),
            no_merge_augmented=getattr(args, 'no_merge_augmented', None),
            cleanup=True  # 启用清理临时文件
        )
        prepare_data(prepare_args, cfg=config)
    else:
        # 检查测试数据是否存在，不存在时仅处理测试数据
        test_images_path = normalize_path(paths.test_images_dir)
        if not has_image_files(test_images_path):
            logger.warning(f"Test image directory does not exist or is empty: {test_images_path}")
            logger.info("Preparing test data only")
            prepare_test_data(custom_dataset_path=getattr(args, 'dataset_path', None), cfg=config)
    
    # 根据命令行参数覆盖配置
    apply_train_overrides(args, cfg=config)
        
    # 训练模型
    try:
        train_model = load_train_function()
    except ModuleNotFoundError:
        return

    train_model(config, force_train=getattr(args, 'force_train', False))
    
    # 训练完成后，如果参数中指定了清理临时文件，则执行清理
    if getattr(args, 'cleanup', False):
        logger.info("Training completed, cleaning up temporary files...")
        from dataset.data_prep import DataPreparation
        data_prep = DataPreparation()
        # 使用force_cleanup参数决定是否强制清理
        force_cleanup = getattr(args, 'force_cleanup', False)
        data_prep.cleanup_temp_files(force=force_cleanup)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Training pipeline completed in {elapsed_time:.2f} seconds")

def run_inference(args, cfg=None) -> None:
    """执行模型推理
    
    参数:
        args: 命令行参数
    """
    logger.info("Starting inference")
    cfg = cfg or config
    
    # 设置默认模型路径
    if not hasattr(args, 'model') or not args.model:
        # 自动查找最佳模型文件
        best_model_path = os.path.join(cfg.best_weights, cfg.model_name, "0", "best_model.pth.tar")
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(cfg.weights, cfg.model_name, "0", "_latest_model.pth.tar")
        
        if not os.path.exists(best_model_path):
            logger.error(f"Could not find model file. Please specify --model or ensure model exists at: {best_model_path}")
            return
        
        model_path = best_model_path
        logger.info(f"Using auto-detected model: {model_path}")
    else:
        model_path = args.model
    
    # 验证模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # 设置默认输入路径
    if not hasattr(args, 'input') or not args.input:
        input_path = paths.test_images_dir
        logger.info(f"Using default input path: {input_path}")
    else:
        input_path = args.input

    def normalize_for_compare(path: str) -> str:
        return os.path.normcase(os.path.abspath(normalize_path(path)))

    default_test_paths = {
        normalize_for_compare(paths.test_images_dir),
        normalize_for_compare(paths.test_dir),
    }
    is_default_test_input = normalize_for_compare(input_path) in default_test_paths

    is_existing_dir = os.path.isdir(input_path)
    existing_images = list_image_files(input_path) if is_existing_dir else []
    input_missing_or_empty = (not os.path.exists(input_path)) or (is_existing_dir and len(existing_images) == 0)
        
    # 获取输入路径，可以是单个图像或目录
    if input_missing_or_empty:
        if not os.path.exists(input_path):
            logger.error(f"Input path not found: {input_path}")
        else:
            logger.warning(f"Input directory is empty: {input_path}")
        
        # 检查是否是测试目录
        if is_default_test_input:
            logger.info("Test data preparation needed")
            prepared = prepare_test_data(
                custom_dataset_path=getattr(args, 'dataset_path', None) if cfg.use_custom_dataset_path else None,
                cfg=cfg,
            )
            if prepared:
                logger.info(f"Successfully prepared test data: {input_path}")

            prepared_images = list_image_files(input_path) if os.path.isdir(input_path) else []
            if (not os.path.exists(input_path)) or (os.path.isdir(input_path) and not prepared_images):
                logger.error(f"Even after attempting to prepare test data, no valid input images found at: {input_path}")
                return
        else:
            logger.error("Input path is missing/empty and is not a default test directory")
            return
    
    try:
        predict = load_predict_function()
    except ModuleNotFoundError:
        return

    # 设置输出文件路径
    output_path = args.output if args.output else paths.prediction_file
    
    device = None if getattr(args, 'device', None) in (None, 'auto') else args.device

    # 检查输入是文件还是目录
    model_name = getattr(args, 'model_name', None)

    if os.path.isfile(input_path):
        logger.info(f"Running inference on single image: {input_path}")
        results = predict(
            model_path,
            input_path,
            output_path,
            is_dir=False,
            device=device,
            model_name=model_name,
            topk=getattr(args, 'topk', 3),
            save_probs=getattr(args, 'save_probs', False),
            output_format=getattr(args, 'output_format', 'submit'),
            confidence_threshold=getattr(args, 'confidence_threshold', None),
            tta_views=getattr(args, 'tta_views', None),
            cfg=cfg,
        )
    else:
        logger.info(f"Running inference on directory: {input_path}")
        results = predict(
            model_path,
            input_path,
            output_path,
            is_dir=True,
            device=device,
            model_name=model_name,
            batch_size=getattr(args, 'batch_size', None),
            num_workers=getattr(args, 'num_workers', None),
            topk=getattr(args, 'topk', 3),
            save_probs=getattr(args, 'save_probs', False),
            output_format=getattr(args, 'output_format', 'submit'),
            confidence_threshold=getattr(args, 'confidence_threshold', None),
            tta_views=getattr(args, 'tta_views', None),
            cfg=cfg,
        )
    
    # 保存预测结果
    if results:
        logger.info(f"Inference completed successfully, results saved to: {output_path}")
    else:
        logger.error("Inference failed or no results generated")

def run_evaluation(args) -> None:
    """执行模型评估"""
    from libs.evaluation import evaluate_model

    device = None if getattr(args, 'device', None) in (None, 'auto') else args.device
    try:
        summary = evaluate_model(
            model_path=getattr(args, 'model', None),
            data_dir=getattr(args, 'data', None),
            batch_size=getattr(args, 'batch_size', None),
            num_workers=getattr(args, 'num_workers', None),
            device=device,
            topk=getattr(args, 'topk', 2),
            tta_views=getattr(args, 'tta_views', None),
            output_dir=getattr(args, 'output_dir', None),
            save_confusion=not getattr(args, 'no_confusion', False),
            save_report=not getattr(args, 'no_report', False),
            cfg=config,
        )
        logger.info(f"Evaluation summary: {summary}")
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}")

def run_stats(args) -> None:
    """执行数据集统计"""
    from dataset.stats import summarize_dataset
    from utils.utils import handle_datasets
    cfg = config
    data_path = getattr(args, 'data', None) or handle_datasets(data_type="train", cfg=cfg)
    try:
        summary = summarize_dataset(
            data_path,
            output_file=getattr(args, 'output', None),
            top_n=getattr(args, 'top', 10),
            cfg=cfg,
        )
        logger.info(f"Dataset stats: {summary}")
    except Exception as exc:
        logger.error(f"Stats failed: {exc}")

def export_model(args) -> None:
    """导出模型为 ONNX 格式"""
    try:
        import torch
        import onnx
    except ImportError as e:
        logger.error("ONNX export requires 'onnx' package. Install with: pip install onnx onnxruntime")
        logger.error(f"Import error: {e}")
        return

    from models.model import get_net
    from utils.utils import setup_device

    cfg = config

    # 获取模型路径
    model_path = args.model
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    # 获取模型名称
    model_name = getattr(args, 'model_name', None) or cfg.model_name

    # 获取输入尺寸
    if hasattr(args, 'input_size') and args.input_size:
        img_height, img_width = args.input_size
    else:
        img_height, img_width = cfg.img_height, cfg.img_width

    # 获取输出路径
    output_path = getattr(args, 'output', None)
    if not output_path:
        output_path = f"{model_name}.onnx"

    logger.info(f"Exporting model: {model_name}")
    logger.info(f"Model weights: {model_path}")
    logger.info(f"Input size: {img_height}x{img_width}")
    logger.info(f"Output path: {output_path}")

    # 设置设备
    device = setup_device(cfg)

    # 创建模型
    try:
        model = get_net(model_name, num_classes=cfg.num_classes, pretrained=False)
        model = model.to(device)

        # 加载权重 - SECURITY FIX: Added weights_only=True to prevent arbitrary code execution
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 创建示例输入
    dummy_input = torch.randn(1, 3, img_height, img_width, device=device)

    # 设置动态轴
    dynamic_axes = None
    if getattr(args, 'dynamic_axes', False):
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    # 导出模型
    try:
        logger.info("Exporting to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=getattr(args, 'opset', 11),
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        logger.info(f"Model exported successfully to: {output_path}")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return

    # 验证导出的模型
    if getattr(args, 'verify', False):
        try:
            logger.info("Verifying exported model...")
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model is valid")

            # 尝试使用 onnxruntime 进行推理测试
            try:
                import onnxruntime as ort
                ort_session = ort.InferenceSession(output_path)

                # 创建测试输入
                test_input = torch.randn(1, 3, img_height, img_width).numpy()
                ort_inputs = {ort_session.get_inputs()[0].name: test_input}
                ort_outputs = ort_session.run(None, ort_inputs)

                logger.info(f"ONNX Runtime inference test passed. Output shape: {ort_outputs[0].shape}")
            except ImportError:
                logger.warning("onnxruntime not installed, skipping inference test")
            except Exception as e:
                logger.warning(f"ONNX Runtime inference test failed: {e}")

        except Exception as e:
            logger.error(f"Model verification failed: {e}")

def main():
    """主函数"""
    args = setup_parser().parse_args()
    cfg = config

    if getattr(args, "version", False):
        print_version_info(simple=True)
        return
    
    # 处理未指定命令的情况
    if args.command is None:
        logger.info("No command specified. Running all operations in sequence.")
        
        # 1. 数据准备
        logger.info("Step 1: Data preparation")
        
        # 检查是否需要准备数据
        need_data_preparation = True
        
        # 检查训练数据是否已存在
        train_path = normalize_path(paths.train_dir)
        if os.path.exists(train_path):
            train_files = sum(len(glob.glob(os.path.join(d, "*.*"))) 
                           for d in glob.glob(os.path.join(train_path, "*")) if os.path.isdir(d))
            if train_files > cfg.min_files_threshold:
                logger.info(f"Training data already exists ({train_files} files), skipping training data preparation")
                need_data_preparation = False
        
        # 检查测试数据是否已存在
        test_path = normalize_path(paths.test_images_dir)
        need_test_data = True
        if os.path.exists(test_path) and len(glob.glob(os.path.join(test_path, "*.*"))) > 0:
            need_test_data = False
            logger.info(f"Test data already exists, skipping test data preparation")
        
        # 根据检查结果运行不同的数据准备步骤
        if need_data_preparation:
            logger.info("Preparing full training and test data")
            prepare_args = argparse.Namespace(
                extract=True,
                process=True,
                augment=True,
                status=True,
                merge="all",
                cleanup=True,  # 启用清理临时文件
                dataset_path=None,
                merge_augmented=True,
                no_merge_augmented=False
            )
            prepare_data(prepare_args, cfg=cfg)
        elif need_test_data:
            logger.info("Preparing test data only")
            prepare_test_data(cfg=cfg)
        else:
            logger.info("All data already exists, skipping data preparation")
        
        # 2. 训练模型
        logger.info("Step 2: Model training")
        
        # 创建训练模型的参数对象
        train_args = argparse.Namespace(
            epochs=None,                    # 使用配置文件中的默认轮次
            model=None,                     # 使用配置文件中的默认模型
            batch_size=None,
            lr=None,
            prepare=False,                  # 我们已经在步骤1中准备了数据
            no_prepare=True,
            dataset_path=None,
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
            no_early_stopping=False,
            patience=None,
            no_amp=False,
            gradient_clip_val=None,
            no_ema=False,
            ema_decay=None,
            device=None,
            gpus=None,
            seed=None,
            label_smoothing=None,
            no_gradient_checkpointing=False,
            cleanup=True,                   # 在训练后清理临时文件
            force_cleanup=True,             # 强制执行清理，不询问用户
            force_train=False               # 不强制重新训练
        )
        
        # 运行训练流程
        train_pipeline(train_args)
        
        # 3. 模型推理
        logger.info("Step 3: Model inference")
        # 查找最佳模型文件
        best_model_path = os.path.join(cfg.best_weights, cfg.model_name, "0", "best_model.pth.tar")
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(cfg.weights, cfg.model_name, "0", "_latest_model.pth.tar")
        
        predict_args = argparse.Namespace(
            model=best_model_path,
            input=paths.test_images_dir,
            output=paths.prediction_file,
            merge=True,  # 确保测试集合并设置正确
            output_format="submit",
            topk=3,
            save_probs=False,
            confidence_threshold=None
        )
        run_inference(predict_args, cfg=cfg)
        
        logger.info("All operations completed successfully.")
        return
    
    logger.info(f"Running command: {args.command}")
    
    if args.command == "prepare":
        prepare_data(args, cfg=cfg)
    elif args.command == "train":
        train_pipeline(args)
    elif args.command == "predict":
        # 确保测试集合并设置正确
        if not hasattr(args, 'merge') or not args.merge:
            # 默认启用测试集合并
            logger.info("Test dataset merging enabled")
            cfg.merge_test_datasets = True
        run_inference(args, cfg=cfg)
    elif args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "stats":
        run_stats(args)
    elif args.command == "version":
        print_version_info(simple=getattr(args, "simple", False))
    elif args.command == "config":
        payload = export_config(
            output_path=getattr(args, "output", None),
            pretty=not getattr(args, "compact", False),
        )
        if not getattr(args, "output", None):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "models":
        from models.model import get_available_models

        models = sorted(get_available_models())
        if getattr(args, "json", False):
            print(json.dumps(models, ensure_ascii=False, indent=2))
        else:
            print("\n".join(models))
    elif args.command == "export":
        export_model(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
