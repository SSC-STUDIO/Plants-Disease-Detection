#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在远程服务器上合并分卷文件并启动训练
"""

import os
import subprocess
import sys

def merge_files(parts_dir, output_file, prefix):
    """合并分卷文件"""
    print(f"\n合并 {prefix} 分卷文件...")

    # 获取所有分卷文件
    parts = sorted([f for f in os.listdir(parts_dir) if f.startswith(prefix)])

    if not parts:
        print(f"错误: 未找到 {prefix} 分卷文件")
        return False

    print(f"找到 {len(parts)} 个分卷文件")

    # 合并文件
    with open(output_file, 'wb') as outfile:
        for i, part in enumerate(parts, 1):
            part_path = os.path.join(parts_dir, part)
            print(f"  [{i}/{len(parts)}] 合并 {part}...")

            with open(part_path, 'rb') as infile:
                outfile.write(infile.read())

    print(f"✓ 合并完成: {output_file}")
    return True

def verify_file(filepath, expected_size):
    """验证文件大小"""
    actual_size = os.path.getsize(filepath)
    print(f"文件大小: {actual_size / (1024**3):.2f} GB")

    if expected_size and abs(actual_size - expected_size) > 1024*1024:  # 允许1MB误差
        print(f"警告: 文件大小不匹配! 预期: {expected_size / (1024**3):.2f} GB")
        return False

    return True

def main():
    base_dir = os.path.expanduser("~/Plants-Disease-Detection")
    parts_dir = os.path.join(base_dir, "data/parts")
    data_dir = os.path.join(base_dir, "data")

    print("=" * 60)
    print("分卷文件合并工具")
    print("=" * 60)

    # 合并训练集
    trainingset = os.path.join(data_dir, "ai_challenger_pdr2018_trainingset_20181023.zip")
    if not merge_files(parts_dir, trainingset, "trainingset_part_"):
        return 1
    verify_file(trainingset, 3110000000)  # 约2.9GB

    # 合并验证集
    valset = os.path.join(data_dir, "ai_challenger_pdr2018_validationset_20181023.zip")
    if not merge_files(parts_dir, valset, "valset_part_"):
        return 1
    verify_file(valset, 432000000)  # 约412MB

    print("\n" + "=" * 60)
    print("✓ 所有文件合并完成！")
    print("=" * 60)

    # 询问是否开始训练
    print("\n准备启动 EVA-02 模型训练...")
    subprocess.run([sys.executable, os.path.join(base_dir, "scripts/train_eva02_remote.py")])

    return 0

if __name__ == "__main__":
    sys.exit(main())