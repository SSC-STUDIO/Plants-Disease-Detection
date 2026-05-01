# 远程训练操作指南

## 服务器信息
- **主机**: 100.90.176.72
- **用户**: ssh_local
- **密码**: 通过安全渠道获取（不要写入代码仓库）
- **项目路径**: `D:\EliuaK_Csy\Working-Paper\My-Program\Plants-Disease-Detection`
- **GPU**: NVIDIA RTX 4060 Laptop GPU (8GB)
- **CUDA**: 12.8

## 快速开始

### 方法一：直接在远程服务器上执行（推荐）

1. **远程桌面或 SSH 登录服务器**
   ```
   用户名: ssh_local
   密码: [通过安全渠道获取]
   ```

2. **打开 PowerShell 或 CMD**

3. **切换到项目目录**
   ```powershell
   cd "D:\EliuaK_Csy\Working-Paper\My-Program\Plants-Disease-Detection"
   ```

4. **启动训练**
   ```powershell
   # 使用已更新的配置（convnextv2_base_384, epochs=50）
   py main.py train --model convnextv2_base_384 --epochs 50 --skip-data-prep
   
   # 或后台运行
   start /B py main.py train --model convnextv2_base_384 --epochs 50 --skip-data-prep > training_remote.log 2>&1
   ```

### 方法二：使用提供的脚本

在项目目录下已创建 `run_training.ps1`，可以直接运行：

```powershell
cd "D:\EliuaK_Csy\Working-Paper\My-Program\Plants-Disease-Detection"
.\run_training.ps1 -OfflineWeights
```

## 监控训练

### 查看训练日志
```powershell
cd "D:\EliuaK_Csy\Working-Paper\My-Program\Plants-Disease-Detection"
Get-Content training_remote.log -Tail 50 -Wait
```

### 查看 Python 进程
```powershell
tasklist /fi "imagename eq python*"
```

### 停止训练
```powershell
taskkill /f /im python.exe
```

## 配置说明

当前配置 (`config.py`) 已修改为：
- **模型**: convnextv2_base_384
- **预训练**: 启用
- **批次大小**: 8
- **输入尺寸**: 384x384
- **学习率**: 1e-4
- **训练轮数**: 50

## 数据准备

如果数据尚未准备：
```powershell
py main.py prepare --all --merge-augmented --cleanup
```

## 常见问题

### 1. Python 命令找不到
使用 `py` 而不是 `python`：
```powershell
py --version
```

### 2. CUDA 不可用
检查 CUDA 安装：
```powershell
nvidia-smi
py -c "import torch; print(torch.cuda.is_available())"
```

### 3. 内存不足
减小批次大小：
```powershell
py main.py train --model convnextv2_base_384 --batch-size 4
```

## 文件同步

如需从本地同步代码到远程：

### 本地执行 (WSL)
```bash
# 使用 scp 同步修改的文件
scp config.py ssh_local@100.90.176.72:"D:\EliuaK_Csy\Working-Paper\My-Program\Plants-Disease-Detection\"
scp models/model.py ssh_local@100.90.176.72:"D:\EliuaK_Csy\Working-Paper\My-Program\Plants-Disease-Detection\models\"
scp dataset/dataloader.py ssh_local@100.90.176.72:"D:\EliuaK_Csy\Working-Paper\My-Program\Plants-Disease-Detection\dataset\"
scp libs/training.py ssh_local@100.90.176.72:"D:\EliuaK_Csy\Working-Paper\My-Program\Plants-Disease-Detection\libs\"
```

## 本地已完成的工作

1. ✅ 修改 `config.py` - 默认模型改为 convnextv2_base_384
2. ✅ 修改 `models/model.py` - 修复预训练权重回退逻辑
3. ✅ 修改 `dataset/dataloader.py` - 修复 collate_fn
4. ✅ 修改 `libs/training.py` - 添加类别数同步
5. ✅ 创建 `scripts/monitor_remote_training.py` - 远程监控脚本
6. ✅ 添加错误胶囊 Hook - 会话结束时自动检查

## 下一步操作

由于 SSH 到 Windows 的终端兼容性问题，建议在远程服务器上直接执行训练命令。

训练启动后，模型权重将保存在：
- `checkpoints/best/convnextv2_base_384/0/best_model.pth.tar`
- `checkpoints/best/convnextv2_base_384/0/_latest_model.pth.tar`
