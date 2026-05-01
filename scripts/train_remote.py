#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
远程服务器训练脚本 - ConvNeXt V2 Base 384
自动完成数据准备、模型训练和推理
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plants Disease Detection - Remote Training")
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='Skip data preparation (if already done)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training (default: 8 for 384x384 input)')
    parser.add_argument('--model', type=str, default='convnextv2_base_384',
                        choices=['convnextv2_base_384', 'eva02_large', 'efficientnetv2_s'],
                        help='Model architecture (default: convnextv2_base_384)')
    parser.add_argument('--offline', action='store_true',
                        help='Offline mode - disable pretrained weight download')
    parser.add_argument('--force-train', action='store_true',
                        help='Force training from scratch, ignoring checkpoints')
    args = parser.parse_args()

    print("=" * 60)
    print("Plants Disease Detection - Remote Training Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Offline mode: {args.offline}")
    print("=" * 60)

    # 设置环境变量
    env = os.environ.copy()
    if args.offline:
        env['HF_HUB_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'
        print("\nOffline mode enabled - using local weights or random init")

    # Step 1: 数据准备
    if not args.skip_data_prep:
        print("\n[Step 1/3] Data Preparation")
        print("Preparing training and validation datasets...")

        cmd = [
            sys.executable,
            "main.py",
            "prepare",
            "--all",
            "--merge-augmented",
            "--cleanup"
        ]

        result = subprocess.run(cmd, cwd=os.getcwd(), env=env)
        if result.returncode != 0:
            print("ERROR: Data preparation failed!")
            return 1

        print("✓ Data preparation completed successfully")
    else:
        print("\n[Step 1/3] Data Preparation - SKIPPED")

    # Step 2: 模型训练
    print("\n[Step 2/3] Model Training")
    print(f"Training {args.model} model for {args.epochs} epochs...")

    cmd = [
        sys.executable,
        "main.py",
        "train",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--model", args.model,
        "--wandb",
        "--wandb-project", f"plants-disease-detection-{args.model}",
        "--cleanup"
    ]

    if args.force_train:
        cmd.append("--force-train")

    result = subprocess.run(cmd, cwd=os.getcwd(), env=env)
    if result.returncode != 0:
        print("ERROR: Training failed!")
        return 1

    print("✓ Training completed successfully")

    # Step 3: 模型推理
    print("\n[Step 3/3] Model Inference")
    print("Running inference on test dataset...")

    cmd = [
        sys.executable,
        "main.py",
        "predict",
        "--model-name", args.model,
        "--output-format", "submit",
        "--topk", "3"
    ]

    result = subprocess.run(cmd, cwd=os.getcwd(), env=env)
    if result.returncode != 0:
        print("ERROR: Inference failed!")
        return 1

    print("✓ Inference completed successfully")

    print("\n" + "=" * 60)
    print("All steps completed successfully!")
    print(f"Model weights saved in: checkpoints/best/{args.model}/")
    print("Predictions saved in: submit/prediction.json")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())