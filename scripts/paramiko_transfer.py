#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 paramiko 进行可靠的文件传输
支持断点续传和进度显示
"""

import os
import sys
import paramiko
import time
from stat import S_ISDIR

def format_size(bytes_size):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def get_remote_file_size(sftp, remote_path):
    """获取远程文件大小"""
    try:
        return sftp.stat(remote_path).st_size
    except:
        return 0

def transfer_file(sftp, local_path, remote_path, chunk_size=1024*1024):
    """带进度显示的文件传输 - 支持断点续传"""
    local_size = os.path.getsize(local_path)
    filename = os.path.basename(local_path)
    remote_size = get_remote_file_size(sftp, remote_path)

    print(f"\n传输文件: {filename}")
    print(f"总大小: {format_size(local_size)}")
    print(f"远程路径: {remote_path}")

    if remote_size >= local_size:
        print(f"✓ 文件已完整传输，跳过")
        return True

    # 断点续传
    transferred = 0
    if remote_size > 0:
        print(f"发现部分文件 ({format_size(remote_size)})，将继续传输")
        transferred = remote_size

    try:
        with open(local_path, 'rb') as local_file:
            local_file.seek(remote_size)

            with sftp.file(remote_path, 'ab' if remote_size > 0 else 'wb') as remote_file:
                remote_file.set_pipelined(True)
                start_time = time.time()

                while transferred < local_size:
                    chunk = local_file.read(chunk_size)
                    if not chunk:
                        break

                    remote_file.write(chunk)
                    transferred += len(chunk)

                    progress = (transferred / local_size) * 100
                    elapsed = time.time() - start_time
                    speed = (transferred - remote_size) / elapsed if elapsed > 0 else 0

                    print(f"\r进度: {progress:.1f}% ({format_size(transferred)}/{format_size(local_size)}) "
                          f"速度: {format_size(speed)}/s", end='', flush=True)

        print(f"\n✓ {filename} 传输完成！")
        return True

    except Exception as e:
        print(f"\n✗ 传输中断: {e}")
        return False

def main():
    # 连接参数
    hostname = "100.90.176.72"
    username = "EliuaK_Csy"
    port = 22

    # 要传输的文件
    files = [
        ("ai_challenger_pdr2018_validationset_20181023.zip", 412*1024*1024),
        ("ai_challenger_pdr2018_testa_20181023.zip", 413*1024*1024),
        ("ai_challenger_pdr2018_testb_20181023.zip", 413*1024*1024),
        ("ai_challenger_pdr2018_trainingset_20181023.zip", 2.9*1024*1024*1024),
    ]

    local_base = "C:/Users/96152/My-Project/Active/Software/Plants-Disease-Detection/data"
    # Windows 服务器路径 - 使用反斜杠
    remote_base = r"C:\Users\EliuaK_Csy\Plants-Disease-Detection\data"

    print("=" * 60)
    print("Paramiko 文件传输工具")
    print("=" * 60)

    # 连接服务器
    print(f"\n连接服务器 {hostname}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, port=port, username=username, timeout=30)
        sftp = client.open_sftp()
        print("✓ 连接成功！")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return 1

    # 传输文件
    success_count = 0
    for filename, expected_size in files:
        local_path = os.path.join(local_base, filename).replace('\\', '/')
        remote_path = f"{remote_base}/{filename}"

        if transfer_file(sftp, local_path, remote_path):
            success_count += 1

    sftp.close()
    client.close()

    print("\n" + "=" * 60)
    print(f"传输完成: {success_count}/{len(files)} 个文件")
    print("=" * 60)

    if success_count == len(files):
        print("\n✓ 所有数据传输完成！")

        # 启动训练
        print("\n启动训练...")
        client.connect(hostname, port=port, username=username, timeout=30)
        stdin, stdout, stderr = client.exec_command(
            "cd ~/Plants-Disease-Detection && python scripts/train_eva02_remote.py"
        )
        print(stdout.read().decode())
        client.close()

    return 0 if success_count == len(files) else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n传输已停止")
        sys.exit(1)
