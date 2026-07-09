"""
training_checkpoint.py - Checkpoint management for training module
Extracted from training.py to reduce function complexity
"""

import os
import torch
from typing import Dict, Any, Tuple, Optional
from models.model import get_net


def load_training_state(checkpoint_path: str, best_model_path: str,
                        device, epochs: int, force_train: bool, logger) -> Tuple[int, float, str, Optional[Dict[str, Any]]]:
    """Load training state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        best_model_path: Path to best model
        device: Device to load on
        epochs: Total planned epochs
        force_train: Whether to force training from scratch
        logger: Logger
        
    Returns:
        (start_epoch, best_acc, model_path, checkpoint_dict) tuple.
        checkpoint_dict is the loaded checkpoint (or None) so callers
        can reuse it without calling torch.load() again.
    """
    start_epoch = 0
    best_acc = 0.0
    model_path = None
    checkpoint = None

    if force_train:
        logger.info("Force training enabled: ignoring existing checkpoints and starting from epoch 0.")
        return start_epoch, best_acc, model_path, checkpoint

    if not os.path.exists(checkpoint_path) and not os.path.exists(best_model_path):
        return start_epoch, best_acc, model_path, checkpoint

    model_path = best_model_path if os.path.exists(best_model_path) else checkpoint_path
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        
        if start_epoch >= epochs:
            logger.info(f"Model already trained for {start_epoch} epochs (configured: {epochs})")
        else:
            logger.info(f"Continuing training from epoch {start_epoch}/{epochs}")
    except Exception as e:
        logger.warning(f"Error loading existing checkpoint: {str(e)}. Starting from epoch 0.")
        start_epoch = 0
        checkpoint = None

    return start_epoch, best_acc, model_path, checkpoint


def load_model_weights(model, model_path: str, device, logger,
                       checkpoint: Optional[Dict[str, Any]] = None) -> bool:
    """Load model weights from checkpoint.

    Args:
        model: Model to load weights into
        model_path: Path to checkpoint
        device: Device
        logger: Logger
        checkpoint: Pre-loaded checkpoint dict (avoids a redundant torch.load).

    Returns:
        True if successful
    """
    if not model_path or not os.path.exists(model_path):
        return False

    try:
        if checkpoint is None:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Loaded weights from %s", model_path)
        return True
    except Exception as e:
        logger.error("Failed to load weights: %s. Starting with fresh model.", e)
        return False


def setup_model(config, device, start_epoch: int, logger):
    """Setup model for training.
    
    Args:
        config: Configuration
        device: Device
        start_epoch: Starting epoch
        logger: Logger
        
    Returns:
        Model instance
    """
    use_pretrained = config.pretrained and start_epoch == 0
    model = get_net(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=use_pretrained,
    )
    
    if use_pretrained:
        logger.info("Initializing model with pretrained backbone weights")
    else:
        logger.info("Skipping pretrained backbone initialization for resumed training")

    if config.use_gradient_checkpointing:
        grad_checkpointing_setter = getattr(model, "set_grad_checkpointing", None)
        if callable(grad_checkpointing_setter):
            try:
                grad_checkpointing_setter(enable=True)
            except TypeError:
                grad_checkpointing_setter(True)
            logger.info("Enabled model gradient checkpointing")
        else:
            logger.info("Gradient checkpointing requested, but model does not expose set_grad_checkpointing()")
    
    model = model.to(device)
    logger.info(f"Model moved to {device}")
    
    return model


def setup_optimizer_state(optimizer, start_epoch: int, model_path: str, device, logger,
                          checkpoint: Optional[Dict[str, Any]] = None):
    """Setup optimizer state from checkpoint.

    Args:
        optimizer: Optimizer
        start_epoch: Starting epoch
        model_path: Model checkpoint path
        device: Device
        logger: Logger
        checkpoint: Pre-loaded checkpoint dict (avoids a redundant torch.load).
    """
    if start_epoch <= 0 or not model_path:
        return

    try:
        if checkpoint is None:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("Restored optimizer state")
    except Exception as e:
        logger.warning(f"Failed to restore optimizer state: {str(e)}")


def restore_scheduler_state(scheduler, start_epoch: int, model_path: str, device, logger,
                            checkpoint: Optional[Dict[str, Any]] = None):
    """Restore learning rate scheduler state from checkpoint.

    Without this, a resumed cosine/onecycle scheduler restarts from the
    initial learning rate instead of continuing the planned schedule,
    which silently degrades training quality on every resume.

    Args:
        scheduler: Learning rate scheduler (or None)
        start_epoch: Starting epoch
        model_path: Model checkpoint path
        device: Device
        logger: Logger
        checkpoint: Pre-loaded checkpoint dict (avoids a redundant torch.load).
    """
    if scheduler is None or start_epoch <= 0 or not model_path:
        return

    try:
        if checkpoint is None:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("Restored learning rate scheduler state")
        else:
            logger.warning(
                "Checkpoint does not contain scheduler state; "
                "LR schedule will restart from the beginning"
            )
    except Exception as e:
        logger.warning(f"Failed to restore scheduler state: {str(e)}")


def restore_ema_state(model_ema, model_path: str, device, logger,
                      checkpoint: Optional[Dict[str, Any]] = None):
    """Restore EMA model weights from checkpoint.

    Without this, a resumed training run loses all accumulated EMA weights
    and starts averaging from the freshly-loaded model — effectively discarding
    the EMA benefit for the first few resumed epochs.

    Args:
        model_ema: EMA model (or None)
        model_path: Model checkpoint path
        device: Device
        logger: Logger
        checkpoint: Pre-loaded checkpoint dict (avoids a redundant torch.load).
    """
    if model_ema is None or not model_path:
        return

    try:
        if checkpoint is None:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if "model_ema" in checkpoint:
            ema_state = checkpoint["model_ema"]
            if isinstance(ema_state, dict) and "module" in ema_state:
                model_ema.module.load_state_dict(ema_state["module"])
            elif isinstance(ema_state, dict):
                model_ema.module.load_state_dict(ema_state)
            logger.info("Restored EMA model state from checkpoint")
        else:
            logger.warning(
                "Checkpoint does not contain EMA state; "
                "EMA will start fresh from the loaded model weights"
            )
    except Exception as e:
        logger.warning(f"Failed to restore EMA state: {str(e)}")


def log_epoch_results(logger, epoch: int, epochs: int, train_loss: float, 
                      train_acc: float, val_loss: float = None, val_acc: float = None):
    """Log epoch results.
    
    Args:
        logger: Logger
        epoch: Current epoch
        epochs: Total epochs
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss (optional)
        val_acc: Validation accuracy (optional)
    """
    if val_loss is not None and val_acc is not None:
        logger.info(f'Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | '
                   f'Val loss: {val_loss:.4f}, acc: {val_acc:.4f}')
    else:
        logger.info(f'Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f}')
