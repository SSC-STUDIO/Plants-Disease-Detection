#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
远程训练监控脚本
"""

import pexpect
import time
import sys
import re
import os
import getpass

HOST = "100.90.176.72"
USER = "ssh_local"
PROJECT_PATH = r"D:\EliuaK_Csy\Working-Paper\My-Program\Plants-Disease-Detection"
PASSWORD_ENV = "REMOTE_SSH_PASSWORD"


def get_ssh_password():
    """从环境变量或交互输入获取 SSH 密码。"""
    password = os.getenv(PASSWORD_ENV)
    if password:
        return password
    return getpass.getpass(f"请输入 {USER}@{HOST} 的 SSH 密码: ")


def check_training_status():
    """检查远程训练状态"""
    password = get_ssh_password()

    child = pexpect.spawn(
        f'ssh -o StrictHostKeyChecking=no {USER}@{HOST}',
        timeout=60,
        encoding='latin-1'
    )

    try:
        child.expect('password:', timeout=20)
        child.sendline(password)
        time.sleep(2)

        # 切换到 cmd
        child.sendline('cmd')
        time.sleep(1)

        # 检查 Python 进程
        child.sendline('tasklist /fi "imagename eq python*" /fo table')
        time.sleep(2)

        # 检查训练日志
        child.sendline(f'type "{PROJECT_PATH}\\training_output.txt" 2>nul')
        time.sleep(3)

        # 读取输出
        time.sleep(2)
        data = child.before or ''

        # 清理控制字符
        clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', data)
        clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', clean)

        return clean

    except Exception as e:
        return f"错误: {e}"
    finally:
        child.close()


def start_training():
    """启动远程训练"""
    password = get_ssh_password()

    print("=" * 60)
    print("启动远程训练")
    print(f"主机: {HOST}")
    print(f"项目: {PROJECT_PATH}")
    print("=" * 60)

    child = pexpect.spawn(
        f'ssh -o StrictHostKeyChecking=no {USER}@{HOST}',
        timeout=300,
        encoding='latin-1'
    )

    try:
        child.expect('password:', timeout=20)
        child.sendline(password)
        time.sleep(3)

        # 切换到 cmd
        child.sendline('cmd')
        time.sleep(1)

        # 切换到项目目录
        child.sendline(f'cd /d {PROJECT_PATH}')
        time.sleep(1)

        # 检查是否有正在运行的训练
        child.sendline('tasklist /fi "imagename eq python*"')
        time.sleep(2)

        # 启动训练（后台模式）
        print("\n启动训练...")
        child.sendline(
            f'start /B py main.py train --model convnextv2_base_384 --epochs 50 '
            f'> training_output.txt 2> training_error.txt'
        )
        time.sleep(5)

        # 检查是否启动成功
        child.sendline('tasklist /fi "imagename eq python*"')
        time.sleep(2)

        # 读取输出
        time.sleep(3)
        data = child.before or ''

        clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', data)
        clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', clean)

        print("\n训练启动结果:")
        print(clean[-2000:] if len(clean) > 2000 else clean)

        print("\n✓ 训练已在后台启动")
        print(f"日志文件: {PROJECT_PATH}\\training_output.txt")
        print(f"错误文件: {PROJECT_PATH}\\training_error.txt")

        return True

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        return False
    finally:
        child.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="远程训练监控")
    parser.add_argument('--start', action='store_true', help='启动训练')
    parser.add_argument('--check', action='store_true', help='检查状态')
    args = parser.parse_args()

    if args.start:
        start_training()
    elif args.check:
        result = check_training_status()
        print(result)
    else:
        # 默认：先检查再启动
        print("检查当前状态...")
        result = check_training_status()
        if 'python.exe' in result.lower():
            print("\n✓ 训练正在运行中")
            print(result[-1000:])
        else:
            print("\n没有发现运行中的训练，准备启动...")
            start_training()
