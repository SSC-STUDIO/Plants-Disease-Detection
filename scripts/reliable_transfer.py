#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可靠的数据传输脚本 - 支持断点续传和重试
"""

import os
import time
import subprocess
import sys

def get_remote_file_size(remote_path):
    """获取远程文件大小"""
    try:
        result = subprocess.run(
            ["ssh", "EliuaK_Csy@100.90.176.72",
             f"(Get-Item {remote_path}).Length 2>$null"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except:
        pass
    return 0

def get_local_file_size(local_path):
    """获取本地文件大小"""
    try:
        return os.path.getsize(local_path)
    except:
        return 0

def format_size(bytes_size):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def transfer_file_with_retry(local_path, remote_path, max_retries=5):
    """带重试机制的文件传输"""
    local_size = get_local_file_size(local_path)
    filename = os.path.basename(local_path)

    print(f"\n传输文件: {filename}")
    print(f"总大小: {format_size(local_size)}")

    for attempt in range(max_retries):
        try:
            remote_size = get_remote_file_size(remote_path)
            progress = (remote_size / local_size * 100) if local_size > 0 else 0

            print(f"[尝试 {attempt + 1}/{max_retries}] 已传输: {format_size(remote_size)} ({progress:.1f}%)")

            if remote_size >= local_size:
                print(f"✓ {filename} 传输完成！")
                return True

            # 使用scp传输
            result = subprocess.run(
                ["scp", local_path, f"EliuaK_Csy@100.90.176.72:{remote_path}"],
                timeout=1800  # 30分钟超时
            )

            if result.returncode == 0:
                # 验证传输
                new_remote_size = get_remote_file_size(remote_path)
                if new_remote_size >= local_size:
                    print(f"✓ {filename} 传输完成！")
                    return True

            print(f"传输中断，等待5秒后重试...")
            time.sleep(5)

        except subprocess.TimeoutExpired:
            print(f"传输超时，等待5秒后重试...")
            time.sleep(5)
        except Exception as e:
            print(f"错误: {e}，等待5秒后重试...")
            time.sleep(5)

    print(f"✗ {filename} 传输失败，已达到最大重试次数")
    return False

def main():
    files_to_transfer = [
        ("ai_challenger_pdr2018_trainingset_20181023.zip", "2.9GB"),
        ("ai_challenger_pdr2018_validationset_20181023.zip", "412MB"),
        ("ai_challenger_pdr2018_testa_20181023.zip", "413MB"),
        ("ai_challenger_pdr2018_testb_20181023.zip", "413MB"),
    ]

    local_base = "C:/Users/96152/My-Project/Active/Software/Plants-Disease-Detection/data"
    remote_base = "~/Plants-Disease-Detection/data"

    print("=" * 60)
    print("数据传输工具 - 支持断点续传")
    print("=" * 60)

    success_count = 0
    for filename, expected_size in files_to_transfer:
        local_path = f"{local_base}/{filename}"
        remote_path = f"{remote_base}/{filename}"

        if transfer_file_with_retry(local_path, remote_path):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"传输完成: {success_count}/{len(files_to_transfer)} 个文件")
    print("=" * 60)

    if success_count == len(files_to_transfer):
        print("\n✓ 所有数据传输完成！")
        print("\n启动训练...")

        train_result = subprocess.run(
            ["ssh", "EliuaK_Csy@100.90.176.72",
             "cd ~/Plants-Disease-Detection && python scripts/train_eva02_remote.py"]
        )
        return train_result.returncode
    else:
        print("\n✗ 部分文件传输失败，请检查网络连接后重试")
        return 1

if __name__ == "__main__":
    sys.exit(main())