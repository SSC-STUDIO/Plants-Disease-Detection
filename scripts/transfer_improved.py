#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的文件传输工具 - 使用小分块传输
解决网络不稳定导致的传输中断问题
"""

import os
import sys
import paramiko
import time

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

def transfer_file_small_chunks(sftp, local_path, remote_path, chunk_size=50*1024*1024):
    """使用小分块传输文件 - 更稳定"""
    local_size = os.path.getsize(local_path)
    filename = os.path.basename(local_path)
    remote_size = get_remote_file_size(sftp, remote_path)

    print(f"\n传输文件: {filename}")
    print(f"本地大小: {format_size(local_size)}")
    print(f"远程大小: {format_size(remote_size)}")

    if remote_size >= local_size:
        print(f"✓ 文件已完整传输，跳过")
        return True

    transferred = remote_size if remote_size > 0 else 0
    remaining = local_size - transferred
    print(f"需要传输: {format_size(remaining)} ({transferred}/{local_size} 已存在)")

    try:
        with open(local_path, 'rb') as local_file:
            local_file.seek(transferred)

            # 使用更小的缓冲区
            buffer_size = 64 * 1024  # 64KB 缓冲区
            start_time = time.time()
            last_progress_time = start_time

            with sftp.file(remote_path, 'ab' if remote_size > 0 else 'wb') as remote_file:
                remote_file.set_pipelined(True)

                while transferred < local_size:
                    # 每次只读取 buffer_size，而不是 chunk_size
                    data = local_file.read(buffer_size)
                    if not data:
                        break

                    remote_file.write(data)
                    transferred += len(data)

                    # 每 5 秒显示一次进度
                    current_time = time.time()
                    if current_time - last_progress_time >= 5:
                        elapsed = current_time - start_time
                        speed = (transferred - remote_size) / elapsed if elapsed > 0 else 0
                        progress = (transferred / local_size) * 100

                        print(f"进度: {progress:.1f}% ({format_size(transferred)}/{format_size(local_size)}) "
                              f"速度: {format_size(speed)}/s")
                        last_progress_time = current_time

            elapsed = time.time() - start_time
            avg_speed = remaining / elapsed if elapsed > 0 else 0
            print(f"\n✓ {filename} 传输完成！平均速度: {format_size(avg_speed)}/s")
            return True

    except Exception as e:
        print(f"\n✗ 传输中断: {e}")
        print(f"已传输: {format_size(transferred)}")
        return False

def main():
    hostname = "100.90.176.72"
    username = "EliuaK_Csy"
    port = 22

    local_base = r"C:\Users\96152\My-Project\Active\Software\Plants-Disease-Detection\data"
    remote_base = r"C:\Users\EliuaK_Csy\Plants-Disease-Detection\data"

    # 只传输未完成的文件
    files_to_transfer = [
        "ai_challenger_pdr2018_validationset_20181023.zip",
        "ai_challenger_pdr2018_trainingset_20181023.zip"
    ]

    print("=" * 60)
    print("改进的文件传输工具 - 小分块传输")
    print("=" * 60)

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

    success_count = 0
    for filename in files_to_transfer:
        local_path = os.path.join(local_base, filename)
        remote_path = remote_base + "\\" + filename

        if transfer_file_small_chunks(sftp, local_path, remote_path):
            success_count += 1

    sftp.close()
    client.close()

    print("\n" + "=" * 60)
    print(f"传输完成: {success_count}/{len(files_to_transfer)} 个文件")
    print("=" * 60)

    return 0 if success_count == len(files_to_transfer) else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n传输已停止")
        sys.exit(1)