import os
import random
import importlib
import torch
import numpy as np
import warnings
import logging
from collections import Counter
from config import config, paths
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset.dataloader import *    
from timeit import default_timer as timer
from models.model import *
from utils.utils import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Any
from utils.utils import ModelEmaV2
from libs.training_helpers import (
    validate_batch, apply_augmentation,
    forward_backward_amp, forward_backward_standard,
    cleanup_memory, format_error_message
)
from libs.training_checkpoint import (
    load_training_state, load_model_weights, setup_model,
    setup_optimizer_state, log_epoch_results
)

class Trainer:
    """训练管理器类，封装所有训练功能"""
    
    def __init__(self, config, logger=None):
        """初始化训练器
        
        参数:
            config: 配置对象
            logger: 可选的日志记录器实例
        """
        self.config = config
        
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 初始化环境和设备
        self.setup_environment()
        self.device = self.get_device()
        self.create_directories()
        
        # 初始化性能监控
        self.memory_tracker = MemoryTracker(track_cuda=self.device.type == 'cuda')
        self.performance_metrics = PerformanceMetrics()
        self.wandb_module = None
        self.wandb_run = None
        self.wandb_enabled = False
        
    def _setup_logger(self):
        """设置并返回训练日志记录器"""
        log_path = self.config.paths.training_log
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        
        # 确保训练日志记录器正确设置
        logger = logging.getLogger('Training')
        
        # 删除旧的handler以避免重复
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # 添加新的handler
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False
        
        # 设置日志级别 - SECURITY FIX: 从环境变量读取，生产环境默认为INFO
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        if log_level.upper() == 'DEBUG':
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        return logger
        
    def setup_environment(self):
        """设置随机种子和CUDA环境"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpus
        if self.config.cuda_alloc_conf:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.config.cuda_alloc_conf
        else:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if self.config.cuda_launch_blocking:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        else:
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
        torch.backends.cudnn.benchmark = True
        warnings.filterwarnings('ignore')

    def get_device(self):
        """获取训练设备"""
        if self.config.device == "cuda":
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
            else:
                self.logger.warning('CUDA requested but not available, falling back to CPU')
                device = torch.device('cpu')
                self.logger.info('Using CPU')
        elif self.config.device == "cpu":
            device = torch.device('cpu')
            self.logger.info('Using CPU (as requested)')
        else:
            # Auto mode
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
            else:
                device = torch.device('cpu')
                self.logger.info('Using CPU (GPU is not available)')
        return device

    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.submit, 
            self.config.weights, 
            self.config.best_weights, 
            self.config.logs,
            self.config.test_data
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            if directory in [self.config.best_weights, self.config.weights]:
                os.makedirs(os.path.join(directory, self.config.model_name, "0"), exist_ok=True)

    def train_epoch(self, model, train_dataloader, 
                   criterion, optimizer, epoch, 
                   log=None, scaler=None, 
                   model_ema=None, scheduler=None):
        """训练单个轮次
        
        参数:
            model: 模型
            train_dataloader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
            epoch: 当前轮次
            log: 日志记录器，如果为None则创建
            scaler: 用于混合精度的梯度缩放器
            model_ema: EMA模型
            
        返回:
            训练损失和准确率
        """
        if log is None:
            log = init_logger(f'train_epoch_{epoch}.log', cfg=self.config)
            
        train_losses = AverageMeter()
        train_top1 = AverageMeter()
        train_top2 = AverageMeter()
        
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')

        error_files = []
        batch_times = []
        error_count = 0
        consecutive_errors = 0
        max_errors = 50
        
        import gc
        cleanup_memory(self.device)
        
        for iter, batch in enumerate(progress_bar):
            try:
                if consecutive_errors >= max_errors:
                    log.write(f"Too many consecutive errors ({consecutive_errors}), stopping training\n")
                    self.logger.error(f"Too many consecutive errors ({consecutive_errors}), stopping training")
                    break
                    
                batch_start = timer()
                
                # Validate batch
                input_tensor, target, is_valid = validate_batch(batch, iter, log, self.device)
                if not is_valid:
                    consecutive_errors += 1
                    continue
                
                # Apply augmentation
                input_tensor, target_a, target_b, lam, use_mixup = apply_augmentation(
                    input_tensor, target, self.config, iter, log
                )
                
                # Forward and backward pass
                if self.config.use_amp and scaler is not None:
                    loss, output = forward_backward_amp(
                        model, input_tensor, target, target_a, target_b, lam,
                        criterion, optimizer, scaler, scheduler, self.config, use_mixup
                    )
                else:
                    loss, output = forward_backward_standard(
                        model, input_tensor, target, target_a, target_b, lam,
                        criterion, optimizer, scheduler, self.config, use_mixup
                    )
                
                # Update EMA
                if model_ema is not None:
                    try:
                        update_ema(model_ema, model, iter)
                    except Exception as e:
                        log.write(f"Error updating EMA in iteration {iter}: {str(e)}\n")
                
                # Calculate metrics
                from utils.utils import accuracy
                acc_target = target_a if use_mixup else target
                precision1_train, precision2_train = accuracy(output, acc_target, topk=(1, 2))
                    
                train_losses.update(loss.item(), input_tensor.size(0))
                train_top1.update(precision1_train.item(), input_tensor.size(0))
                train_top2.update(precision2_train.item(), input_tensor.size(0))
                
                batch_time = timer() - batch_start
                batch_times.append(batch_time)
                consecutive_errors = 0
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{train_losses.avg:.3f}',
                    'top1': f'{train_top1.avg:.3f}',
                    'top2': f'{train_top2.avg:.3f}',
                    'batch_time': f'{np.mean(batch_times[-100:]):.3f}s'
                })
                
                # Monitor memory
                if iter % 10 == 0:
                    self.memory_tracker.update()
                    if self.memory_tracker.should_warn():
                        self.logger.warning(self.memory_tracker.get_warning())
                
                if iter % 100 == 0:
                    cleanup_memory(self.device)
                
                del output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                consecutive_errors += 1
                error_count += 1
                msg, files = format_error_message(e, batch, iter)
                error_files.extend(files)
                log.write(msg + "\n")
                continue
                
        if error_files:
            log.write(f"Found {len(error_files)} problematic files in this epoch.\n")
        
        if error_count > 0:
            log.write(f"Total errors in this epoch: {error_count}\n")
        
        cleanup_memory(self.device)
        
        avg_batch_time = float(np.mean(batch_times)) if batch_times else 0.0
        self.performance_metrics.update_epoch_metrics(
            epoch=epoch,
            loss=train_losses.avg,
            top1=train_top1.avg,
            top2=train_top2.avg,
            batch_time=avg_batch_time,
            memory_usage=self.memory_tracker.get_current_usage()
        )
        
        return train_losses.avg, train_top1.avg, train_top2.avg

    def _get_val_loader(self):
        """基于配置设置创建并返回验证数据加载器
        
        返回基于合并数据集的适当数据加载器（如果有），
        否则返回最佳可用数据集。
        """
        # 使用handle_datasets函数获取适当的验证数据路径
        val_path = handle_datasets(data_type="val", cfg=self.config)
        
        self.logger.info(f"Using validation data from: {val_path}")
        
        # 检查数据集是否存在
        if not os.path.exists(val_path):
            self.logger.error(f"Validation data path does not exist: {val_path}")
            if os.path.exists(self.config.val_data):
                self.logger.info(f"Falling back to default validation data: {self.config.val_data}")
                val_path = self.config.val_data
            else:
                self.logger.warning(f"Cannot find validation data at {val_path} or {self.config.val_data}")
                return None
        
        # 检查目录中是否有图像文件
        image_files = 0
        for root, _, files in os.walk(val_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files += 1
        
        if image_files == 0:
            self.logger.warning(f"No image files found in validation path: {val_path}")
            return None
        
        # 获取数据集文件列表
        val_files = get_files(val_path, mode="val", cfg=self.config)
        
        # 记录验证文件数量
        self.logger.info(f"Validation dataset contains {len(val_files)} images")
        
        # 创建数据集和数据加载器
        val_dataset = PlantDiseaseDataset(
            val_files,
            sampling_threshold=self.config.sampling_threshold,
            sample_size=self.config.sample_size,
            seed=self.config.seed,
            img_width=self.config.img_width,
            img_height=self.config.img_height,
            use_data_aug=False,
            train=False,
            test=False,
            enable_sampling=self.config.enable_sampling,
            validate_images=self.config.enable_image_validation,
            validation_workers=self.config.image_validation_workers,
            cfg=self.config,
        )
        return DataLoader(
            val_dataset, 
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def _get_progressive_size(self, epoch):
        """计算当前epoch应使用的图像尺寸（用于progressive resizing）

        参数:
            epoch: 当前训练轮次

        返回:
            (height, width) 元组
        """
        if not self.config.progressive_resizing:
            return self.config.img_height, self.config.img_width

        # 如果超过progressive_epochs，使用最终尺寸
        if epoch >= self.config.progressive_epochs:
            final_size = self.config.progressive_end_size
            return final_size, final_size

        # 根据progressive_sizes列表计算当前尺寸
        sizes = self.config.progressive_sizes
        if not sizes:
            # 如果列表为空，使用线性插值
            progress = epoch / self.config.progressive_epochs
            size = int(self.config.progressive_start_size +
                      (self.config.progressive_end_size - self.config.progressive_start_size) * progress)
            return size, size

        # 将progressive_epochs分成len(sizes)个阶段
        epochs_per_phase = self.config.progressive_epochs / len(sizes)
        phase = min(int(epoch / epochs_per_phase), len(sizes) - 1)
        size = sizes[phase]
        return size, size

    def _build_wandb_config_payload(self) -> Dict[str, Any]:
        """构建发送到wandb的核心训练配置。"""
        return {
            'model_name': self.config.model_name,
            'device': self.config.device,
            'epochs': self.config.epoch,
            'train_batch_size': self.config.train_batch_size,
            'val_batch_size': self.config.val_batch_size,
            'img_height': self.config.img_height,
            'img_width': self.config.img_width,
            'lr': self.config.lr,
            'weight_decay': self.config.weight_decay,
            'optimizer': self.config.optimizer,
            'scheduler': self.config.scheduler,
            'warmup_epochs': self.config.warmup_epochs,
            'warmup_factor': self.config.warmup_factor,
            'use_amp': self.config.use_amp,
            'use_ema': self.config.use_ema,
            'ema_decay': self.config.ema_decay,
            'use_mixup': self.config.use_mixup,
            'mixup_alpha': self.config.mixup_alpha,
            'cutmix_prob': self.config.cutmix_prob,
            'use_random_erasing': self.config.use_random_erasing,
            'use_data_aug': self.config.use_data_aug,
            'use_weighted_sampler': self.config.use_weighted_sampler,
            'weighted_sampler_power': self.config.weighted_sampler_power,
            'label_smoothing': self.config.label_smoothing,
            'gradient_clip_val': self.config.gradient_clip_val,
            'seed': self.config.seed,
            'progressive_resizing': self.config.progressive_resizing,
            'progressive_sizes': list(self.config.progressive_sizes),
            'progressive_epochs': self.config.progressive_epochs,
            'tta_views': self.config.tta_views,
        }

    def _init_wandb(self, epochs: int, start_epoch: int, force_train: bool) -> None:
        """初始化wandb运行，失败时优雅降级。"""
        self.wandb_module = None
        self.wandb_run = None
        self.wandb_enabled = False

        if not getattr(self.config, 'use_wandb', False):
            return

        wandb_mode = getattr(self.config, 'wandb_mode', 'online')
        if wandb_mode == 'disabled':
            self.logger.info('wandb mode is disabled; skipping experiment tracking')
            return

        try:
            wandb_module = importlib.import_module('wandb')
        except Exception as exc:
            self.logger.warning(f"wandb is enabled but unavailable: {str(exc)}. Continuing without wandb.")
            return

        try:
            run = wandb_module.init(
                project=getattr(self.config, 'wandb_project', None),
                entity=getattr(self.config, 'wandb_entity', None),
                name=getattr(self.config, 'wandb_run_name', None),
                tags=getattr(self.config, 'wandb_tags', None),
                mode=wandb_mode,
                config={
                    **self._build_wandb_config_payload(),
                    'planned_epochs': epochs,
                    'start_epoch': start_epoch,
                    'force_train': force_train,
                },
            )
        except Exception as exc:
            self.logger.warning(f"Failed to initialize wandb: {str(exc)}. Continuing without wandb.")
            return

        self.wandb_module = wandb_module
        self.wandb_run = run
        self.wandb_enabled = True

    def _log_wandb_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_top1: float,
        train_top2: float,
        best_acc: float,
        lr: float,
        current_img_size,
        val_loss: Optional[float] = None,
        val_top1: Optional[float] = None,
    ) -> None:
        """记录epoch级别的wandb指标。"""
        if not self.wandb_enabled or self.wandb_module is None:
            return

        payload = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/top1': train_top1,
            'train/top2': train_top2,
            'best_acc': best_acc,
            'lr': lr,
            'image_size/height': current_img_size[0],
            'image_size/width': current_img_size[1],
        }
        if val_loss is not None:
            payload['val/loss'] = val_loss
        if val_top1 is not None:
            payload['val/top1'] = val_top1

        try:
            self.wandb_module.log(payload)
        except Exception as exc:
            self.logger.warning(f"wandb log failed: {str(exc)}. Disabling wandb logging for the rest of training.")
            self.wandb_enabled = False

    def _finish_wandb(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """安全结束wandb运行。"""
        if self.wandb_module is None and self.wandb_run is None:
            return

        try:
            if summary and self.wandb_run is not None and hasattr(self.wandb_run, 'summary'):
                self.wandb_run.summary.update(summary)
            finish = getattr(self.wandb_module, 'finish', None)
            if callable(finish):
                finish()
        except Exception as exc:
            self.logger.warning(f"wandb finish failed: {str(exc)}")
        finally:
            self.wandb_enabled = False
            self.wandb_run = None
            self.wandb_module = None

    def _recreate_dataloaders_with_size(self, img_height, img_width):
        """使用新的图像尺寸重新创建数据加载器

        参数:
            img_height: 新的图像高度
            img_width: 新的图像宽度

        返回:
            (train_loader, val_loader) 元组
        """
        # 临时修改config中的尺寸
        original_height = self.config.img_height
        original_width = self.config.img_width

        self.config.img_height = img_height
        self.config.img_width = img_width

        # 重新创建数据加载器
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()

        # 恢复原始尺寸（虽然会在下次调用时再次修改）
        self.config.img_height = original_height
        self.config.img_width = original_width

        return train_loader, val_loader

    def train(self, epochs=None, train_loader=None, val_loader=None, force_train: bool = False):
        """完整的训练循环，包含验证

        参数:
            epochs: 轮次数（默认使用config.epoch）
            train_loader: 训练数据加载器（如果为None则创建）
            val_loader: 验证数据加载器（如果为None则创建）
            force_train: 是否忽略已有模型并强制从头训练

        返回:
            包含训练结果的字典
        """
        epochs = epochs or self.config.epoch

        # 获取训练和验证数据加载器
        train_loader = train_loader or self._get_train_loader()
        if val_loader is None:
            val_loader = self._get_val_loader()
            if val_loader is None:
                self.logger.warning("No validation data available. Training will proceed without validation.")

        # 初始化progressive resizing状态
        current_img_size = (self.config.img_height, self.config.img_width)
        
        # 加载检查点状态
        checkpoint_path = os.path.join(self.config.weights, self.config.model_name, "0", "_latest_model.pth.tar")
        best_model_path = os.path.join(self.config.best_weights, self.config.model_name, "0", "best_model.pth.tar")
        
        start_epoch, best_acc, model_path = load_training_state(
            checkpoint_path, best_model_path, self.device, epochs, force_train, self.logger
        )
        
        if start_epoch >= epochs and not force_train:
            return {"completed": True, "epochs_trained": start_epoch, "best_acc": best_acc}
        
        # 设置模型
        model = setup_model(self.config, self.device, start_epoch, self.logger)
        
        # 加载权重（如果是继续训练）
        if start_epoch > 0 and model_path:
            load_model_weights(model, model_path, self.device, self.logger)
        
        # 验证CUDA
        if torch.cuda.is_available():
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            self.logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            self.logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # 创建日志
        train_log = init_logger('train_detailed.log', cfg=self.config)
        self.logger.info("Training started")
        
        # 获取损失函数和优化器
        criterion = get_loss_function(self.device, self.config)
        optimizer = get_optimizer(model, self.config.optimizer, cfg=self.config)
        
        # 恢复优化器状态
        if start_epoch > 0:
            setup_optimizer_state(optimizer, start_epoch, model_path, self.device, self.logger)
        
        # 设置学习率调度器
        scheduler = get_scheduler(optimizer, epochs, len(train_loader), cfg=self.config)
        
        # 设置混合精度训练
        scaler = None
        if self.config.use_amp and self.device.type == 'cuda':
            scaler = GradScaler()
            
        # 创建EMA模型
        model_ema = None
        if self.config.use_ema:
            model_ema = create_model_ema(model, cfg=self.config)

        self._init_wandb(epochs=epochs, start_epoch=start_epoch, force_train=force_train)

        try:
            # 训练循环
            for epoch in range(start_epoch, epochs):
                # Progressive resizing
                if self.config.progressive_resizing:
                    new_size = self._get_progressive_size(epoch)
                    if new_size != current_img_size:
                        self.logger.info(f"Progressive resizing: {current_img_size} -> {new_size} at epoch {epoch}")
                        train_loader, val_loader = self._recreate_dataloaders_with_size(new_size[0], new_size[1])
                        current_img_size = new_size
                        scheduler = get_scheduler(optimizer, epochs, len(train_loader), cfg=self.config)

                # 训练一个epoch
                train_loss, train_acc, train_top2 = self.train_epoch(
                    model, train_loader, criterion, optimizer, epoch, train_log, scaler, model_ema, scheduler
                )

                # 更新学习率
                if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step(epoch)

                current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else self.config.lr

                # 验证
                val_loss, val_acc = 0.0, 0.0
                if val_loader is not None:
                    val_loss, val_acc = self.validate(model, val_loader, criterion, epoch)
                    is_best = val_acc > best_acc
                    best_acc = max(val_acc, best_acc)
                    log_epoch_results(self.logger, epoch, epochs, train_loss, train_acc, val_loss, val_acc)
                else:
                    is_best = train_acc > best_acc
                    best_acc = max(train_acc, best_acc)
                    log_epoch_results(self.logger, epoch, epochs, train_loss, train_acc)

                # 保存检查点
                save_latest_model({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best, fold=0, cfg=self.config)

                # 记录wandb
                self._log_wandb_epoch(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    train_top1=train_acc,
                    train_top2=train_top2,
                    val_loss=val_loss if val_loader else None,
                    val_top1=val_acc if val_loader else None,
                    best_acc=best_acc,
                    lr=current_lr,
                    current_img_size=current_img_size,
                )

                # 早停
                if val_loader and self.config.use_early_stopping and \
                   self.performance_metrics.should_stop(val_loss, self.config.early_stopping_patience):
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        finally:
            self._finish_wandb({
                'best_acc': best_acc,
                'epochs_trained': epoch + 1 if 'epoch' in locals() else start_epoch,
            })

        self.logger.info("Training completed successfully")
        return self.performance_metrics.get_summary()
    
    def validate(self, model, val_loader, criterion, epoch):
        """验证模型
        
        参数:
            model: 要验证的模型
            val_loader: 验证数据加载器
            criterion: 损失函数
            epoch: 当前轮次
            
        返回:
            验证损失和准确率
        """
        val_losses = AverageMeter()
        val_top1 = AverageMeter()
        
        # Switch to evaluation mode
        model.eval()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
            for input, target in pbar:
                input = input.to(self.device)
                target = torch.tensor(target).to(self.device)
                
                # Forward pass
                output = model(input)
                loss = criterion(output, target)
                
                # Measure accuracy and record loss
                prec1, _ = accuracy(output, target, topk=(1, 2))
                val_losses.update(loss.item(), input.size(0))
                val_top1.update(prec1.item(), input.size(0))
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{val_losses.avg:.3f}',
                    'top1': f'{val_top1.avg:.3f}'
                })
        
        return val_losses.avg, val_top1.avg
    
    def _get_train_loader(self):
        """基于配置设置创建并返回训练数据加载器
        
        返回基于合并数据集的适当数据加载器（如果有），
        否则返回最佳可用数据集。
        """
        # 使用handle_datasets函数获取适当的训练数据路径
        # 该函数已经修改为优先使用合并的数据集(如果有)，否则使用配置策略选择单个数据集
        train_path = handle_datasets(data_type="train", cfg=self.config)
        
        self.logger.info(f"Using training data from: {train_path}")
        
        # 检查数据集是否存在
        if not os.path.exists(train_path):
            self.logger.error(f"Training data path does not exist: {train_path}")
            if os.path.exists(self.config.train_data):
                self.logger.info(f"Falling back to default training data: {self.config.train_data}")
                train_path = self.config.train_data
            else:
                raise FileNotFoundError(f"Cannot find training data at {train_path} or {self.config.train_data}")
        
        # 检查目录中是否有图像文件
        image_files = 0
        for root, _, files in os.walk(train_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files += 1
        
        if image_files == 0:
            self.logger.error(f"No image files found in training path: {train_path}")
            raise ValueError(f"No training images found in {train_path}")
        
        # 获取数据集文件列表
        train_files = get_files(train_path, mode="train", cfg=self.config)
        
        # 记录训练文件数量
        self.logger.info(f"Training dataset contains {len(train_files)} images")
        
        # 创建数据集和数据加载器
        train_dataset = PlantDiseaseDataset(
            train_files,
            sampling_threshold=self.config.sampling_threshold,
            sample_size=self.config.sample_size,
            seed=self.config.seed,
            img_width=self.config.img_width,
            img_height=self.config.img_height,
            use_data_aug=self.config.use_data_aug,
            enable_sampling=self.config.enable_sampling,
            validate_images=self.config.enable_image_validation,
            validation_workers=self.config.image_validation_workers,
            cfg=self.config,
        )
        sampler = None
        shuffle = True

        if self.config.use_weighted_sampler:
            labels = [label for _, label in train_dataset.imgs if isinstance(label, int)]
            if labels and len(labels) == len(train_dataset.imgs):
                class_counts = Counter(labels)
                sample_weights = []
                for _, label in train_dataset.imgs:
                    count = max(class_counts.get(label, 1), self.config.weighted_sampler_min_count)
                    sample_weights.append((1.0 / float(count)) ** self.config.weighted_sampler_power)

                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                shuffle = False
                self.logger.info(
                    "Enabled weighted sampler for %d classes (min_count=%d, power=%.2f)",
                    len(class_counts),
                    self.config.weighted_sampler_min_count,
                    self.config.weighted_sampler_power,
                )
            else:
                self.logger.warning("Weighted sampler requested, but training labels were not available in expected format")

        return DataLoader(
            train_dataset, 
            batch_size=self.config.train_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

class MemoryTracker:
    """内存使用监视器"""
    
    def __init__(self, warning_threshold: float = 0.9, track_cuda: bool = True):
        """初始化内存监视器
        
        参数:
            warning_threshold: 内存使用警告阈值（占总内存的比例）
        """
        self.warning_threshold = warning_threshold
        self.track_cuda = track_cuda
        self.current_usage = 0
        self.peak_usage = 0
        
    def update(self) -> None:
        """更新内存使用统计"""
        if not self.track_cuda or not torch.cuda.is_available():
            self.current_usage = 0
            return

        max_allocated = torch.cuda.max_memory_allocated()
        if max_allocated <= 0:
            self.current_usage = 0
            return

        current = torch.cuda.memory_allocated() / max_allocated
        self.peak_usage = max(self.peak_usage, current)
        self.current_usage = current
        
    def should_warn(self) -> bool:
        """检查是否应该发出警告"""
        return self.current_usage > self.warning_threshold
        
    def get_warning(self) -> str:
        """获取警告消息"""
        return f"High memory usage: {self.current_usage:.1%} of available memory"
        
    def get_current_usage(self) -> float:
        """获取当前内存使用率"""
        return self.current_usage

class PerformanceMetrics:
    """性能指标跟踪器"""
    
    def __init__(self):
        """初始化性能指标跟踪器"""
        self.metrics = {
            'loss': [],
            'top1': [],
            'top2': [],
            'batch_time': [],
            'memory_usage': [],
            'val_loss': []  # For early stopping
        }
        self.epochs = []
        self.patience_counter = 0
        
    def update_epoch_metrics(self, epoch: int, loss: float, top1: float, 
                           top2: float, batch_time: float, memory_usage: float,
                           val_loss: float = None) -> None:
        """更新每轮性能指标
        
        参数:
            epoch: 轮次
            loss: 损失值
            top1: Top-1准确率
            top2: Top-2准确率
            batch_time: 平均批次时间
            memory_usage: 内存使用率
            val_loss: 验证损失（可选）
        """
        self.epochs.append(epoch)
        self.metrics['loss'].append(loss)
        self.metrics['top1'].append(top1)
        self.metrics['top2'].append(top2)
        self.metrics['batch_time'].append(batch_time)
        self.metrics['memory_usage'].append(memory_usage)
        
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        
    def should_stop(self, val_loss: float, patience: int) -> bool:
        """检查训练是否应该提前停止
        
        参数:
            val_loss: 当前验证损失
            
        返回:
            表示是否停止训练的布尔值
        """
        if not self.metrics['val_loss']:
            self.metrics['val_loss'].append(val_loss)
            return False
            
        if val_loss >= min(self.metrics['val_loss']):
            self.patience_counter += 1
        else:
            self.patience_counter = 0
            
        self.metrics['val_loss'].append(val_loss)
        return self.patience_counter >= patience
        
    def get_summary(self) -> Dict[str, Any]:
        """获取性能指标摘要"""
        if not self.metrics['top1']:
            return {
                'best_top1': 0.0,
                'best_epoch': None,
                'avg_batch_time': 0.0,
                'peak_memory': 0.0
            }

        return {
            'best_top1': float(max(self.metrics['top1'])),
            'best_epoch': int(self.epochs[np.argmax(self.metrics['top1'])]),
            'avg_batch_time': float(np.mean(self.metrics['batch_time'])) if self.metrics['batch_time'] else 0.0,
            'peak_memory': float(max(self.metrics['memory_usage'])) if self.metrics['memory_usage'] else 0.0
        }

def init_logger(log_name='train_details.log', cfg: Optional[Any] = None) -> Logger:
    """初始化Logger对象
    
    参数:
        log_name: 日志文件名
        
    返回:
        初始化后的Logger对象
    """
    cfg = cfg or config
    log = Logger()
    log.open(log_name, log_dir=cfg.paths.log_dir)
    return log

def train_model(cfg=None, force_train: bool = False):
    """使用给定配置训练模型
    
    参数:
        cfg: 可选配置对象（如果为None则使用默认配置）
        force_train: 是否忽略已有模型并强制从头训练
        
    返回:
        包含训练结果的字典
    """
    try:
        # Use provided config or default
        trainer = Trainer(cfg or config)
        return trainer.train(force_train=force_train)
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        return {"error": str(e)} 
