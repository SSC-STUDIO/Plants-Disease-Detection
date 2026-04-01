"""
training_helpers.py - Helper functions for training module
Extracted from training.py to reduce function complexity
"""

import torch
import numpy as np
import gc
from timeit import default_timer as timer
from torch.cuda.amp import autocast


def validate_batch(batch, iter, log, device):
    """Validate batch data before processing.
    
    Args:
        batch: The batch data
        iter: Current iteration
        log: Logger
        device: Target device
        
    Returns:
        (input_tensor, target_tensor, is_valid) tuple
    """
    if len(batch) != 2:
        log.write(f"Skipping error batch in iteration {iter}\n")
        return None, None, False
        
    input_data, target = batch
    
    if input_data is None or len(input_data) == 0:
        log.write(f"Skipping empty input in iteration {iter}\n")
        return None, None, False
    
    if torch.isnan(input_data).any() or torch.isinf(input_data).any():
        log.write(f"Skipping batch with NaN or Inf values in iteration {iter}\n")
        return None, None, False
    
    input_tensor = input_data.to(device)
    target_tensor = torch.tensor(target).to(device)
    return input_tensor, target_tensor, True


def apply_augmentation(input_tensor, target, config, iter, log):
    """Apply Mixup or CutMix data augmentation.
    
    Args:
        input_tensor: Input tensor
        target: Target tensor
        config: Configuration object
        iter: Current iteration
        log: Logger
        
    Returns:
        (input_tensor, target_a, target_b, lam, use_mixup) tuple
    """
    if not config.use_mixup:
        return input_tensor, target, None, None, False
        
    try:
        from utils.utils import cutmix_data, mixup_data
        r = np.random.rand(1)
        if r < config.cutmix_prob:
            input_tensor, target_a, target_b, lam = cutmix_data(
                input_tensor, target, config.mixup_alpha
            )
        else:
            input_tensor, target_a, target_b, lam = mixup_data(
                input_tensor, target, config.mixup_alpha
            )
        return input_tensor, target_a, target_b, lam, True
    except Exception as e:
        log.write(f"Error applying mixup/cutmix in iteration {iter}: {str(e)}\n")
        return input_tensor, target, None, None, False


def forward_backward_amp(model, input_tensor, target, target_a, target_b, lam,
                         criterion, optimizer, scaler, scheduler, config, use_mixup):
    """Forward and backward pass with Automatic Mixed Precision.
    
    Args:
        model: The model
        input_tensor: Input tensor
        target: Target tensor (for non-mixup)
        target_a, target_b, lam: Mixup parameters
        criterion: Loss criterion
        optimizer: Optimizer
        scaler: Gradient scaler
        scheduler: Learning rate scheduler
        config: Configuration
        use_mixup: Whether mixup is used
        
    Returns:
        (loss, output) tuple
    """
    with autocast():
        output = model(input_tensor)
        if use_mixup:
            from utils.utils import mixup_criterion
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            loss = criterion(output, target)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    
    if config.gradient_clip_val > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
    
    scaler.step(optimizer)
    scaler.update()
    
    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step()
        
    return loss, output


def forward_backward_standard(model, input_tensor, target, target_a, target_b, lam,
                              criterion, optimizer, scheduler, config, use_mixup):
    """Standard forward and backward pass without AMP.
    
    Args:
        model: The model
        input_tensor: Input tensor
        target: Target tensor (for non-mixup)
        target_a, target_b, lam: Mixup parameters
        criterion: Loss criterion
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration
        use_mixup: Whether mixup is used
        
    Returns:
        (loss, output) tuple
    """
    output = model(input_tensor)
    
    if use_mixup:
        from utils.utils import mixup_criterion
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
    else:
        loss = criterion(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    
    if config.gradient_clip_val > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
    
    optimizer.step()
    
    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step()
        
    return loss, output


def cleanup_memory(device, output=None):
    """Clean up GPU memory.
    
    Args:
        device: The device
        output: Optional output tensor to delete
    """
    if output is not None:
        del output
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def format_error_message(e, batch, iter):
    """Format error message for logging.
    
    Args:
        e: Exception
        batch: The batch that caused the error
        iter: Iteration number
        
    Returns:
        (error_message, error_files) tuple
    """
    error_files = []
    if (hasattr(batch, '__getitem__') and len(batch) > 1 and 
        isinstance(batch[1], list) and len(batch[1]) > 0):
        error_files.extend(batch[1])
        msg = f"Error in training iteration {iter}: {str(e)} - these files will be skipped"
    else:
        msg = f"Error in training iteration {iter}: {str(e)}"
    return msg, error_files
