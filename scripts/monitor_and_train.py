#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动监控数据传输并在完成后启动训练
"""

import os
import time
import subprocess
import sys

def get_file_sizes():
    """获取远程服务器上数据文件的大小"""
    try:
        result = subprocess.run(
            ["ssh", "EliuaK_Csy@100.90.176.72",
             "Get-ChildItem -Path ~/Plants-Disease-Detection/data -File | Select-Object Name,Length | ConvertTo-Json"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            import json
            files = json.loads(result.stdout)
            if isinstance(files, dict):
                files = [files]

            total_size = sum(f.get('Length', 0) for f in files)
            return total_size, files
    except Exception as e:
        print(f"Error checking file sizes: {e}")

    return 0, []

def format_size(bytes_size):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def main():
    print("=" * 60)
    print("数据传输监控 & 自动训练启动器")
    print("=" * 60)

    expected_size = 3.5 * 1024 * 1024 * 1024  # 约3.5GB
    check_interval = 30  # 每30秒检查一次

    print(f"\n预期数据大小: {format_size(expected_size)}")
    print(f"检查间隔: {check_interval}秒\n")

    consecutive_same_size = 0
    last_size = 0

    while True:
        current_size, files = get_file_sizes()

        if current_size > 0:
            progress = (current_size / expected_size) * 100
            print(f"[{time.strftime('%H:%M:%S')}] 已传输: {format_size(current_size)} ({progress:.1f}%)")

            # 显示各文件大小
            for f in files:
                print(f"  - {f['Name']}: {format_size(f.get('Length', 0))}")

            # 检查传输是否完成
            if current_size == last_size and current_size > expected_size * 0.95:
                consecutive_same_size += 1
                if consecutive_same_size >= 3:
                    print(f"\n✓ 数据传输完成！总大小: {format_size(current_size)}")
                    break
            else:
                consecutive_same_size = 0

            last_size = current_size

        time.sleep(check_interval)

    # 启动训练
    print("\n" + "=" * 60)
    print("开始训练 EVA-02 模型...")
    print("=" * 60 + "\n")

    train_cmd = [
        "ssh", "EliuaK_Csy@100.90.176.72",
        "cd ~/Plants-Disease-Detection && python scripts/train_eva02_remote.py"
    ]

    subprocess.run(" ".join(train_cmd), shell=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        sys.exit(0)