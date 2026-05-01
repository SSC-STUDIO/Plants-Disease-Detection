"""
Minimal memory training config - disable memory-heavy features
Run with: python main.py train --epochs 50 --batch-size 4 --no-prepare --no-wandb
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config

# Minimal memory settings
config.train_batch_size = 4
config.val_batch_size = 8
config.num_workers = 0  # No multiprocessing
config.use_ema = False  # Disable EMA - saves ~2x model memory
config.use_gradient_checkpointing = False
config.use_amp = True   # Keep AMP - helps with GPU memory
config.use_mixup = False  # Disable mixup to save CPU memory
config.use_data_aug = False  # Disable augmentation
config.pretrained = False  # Don't download pretrained weights

print("Minimal memory config loaded:")
print(f"  batch_size: {config.train_batch_size}")
print(f"  num_workers: {config.num_workers}")
print(f"  use_ema: {config.use_ema}")
print(f"  use_amp: {config.use_amp}")
print(f"  pretrained: {config.pretrained}")
