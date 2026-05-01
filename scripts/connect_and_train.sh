#!/bin/bash
# connect_and_train.sh - 连接远程服务器并启动训练
# 用法: ./connect_and_train.sh

REMOTE_HOST="100.90.176.72"
REMOTE_USER="ssh_local"
REMOTE_DIR="~/Plants-Disease-Detection"

echo "=========================================="
echo "远程训练服务器连接"
echo "=========================================="
echo ""
echo "主机: $REMOTE_HOST"
echo "用户: $REMOTE_USER"
echo ""
echo "密码: 通过交互输入（不在脚本中保存）"
echo ""
echo "=========================================="
echo ""

# 连接并切换到项目目录
echo "正在连接..."
ssh "$REMOTE_USER@$REMOTE_HOST" -t "cd $REMOTE_DIR && bash"

# 如果需要直接执行训练，取消下面的注释
# ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR && python scripts/train_remote.py --model convnextv2_base_384 --epochs 50"
