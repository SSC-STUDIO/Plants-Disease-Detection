# EVA-02 模型训练指南

## 当前状态

### ✓ 已完成
1. **模型升级** - 已添加EVA-02 Base模型架构到项目
2. **配置优化** - 已调整训练参数适配RTX 4060 GPU (8GB显存)
3. **环境配置** - 远程服务器已安装CUDA PyTorch 2.6.0 + CUDA 12.4
4. **依赖安装** - timm、einops和所有项目依赖已安装
5. **代码传输** - 项目代码已传输到远程服务器

### ⏳ 进行中
- **数据传输** - 训练数据集(2.9GB)和验证数据集(412MB)正在传输到远程服务器

## 远程服务器配置

- **主机**: 100.90.176.72
- **用户**: EliuaK_Csy
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB)
- **Python**: 3.13.12
- **PyTorch**: 2.6.0+cu124
- **CUDA**: 12.4

## EVA-02 模型配置

### 优化调整（适配8GB显存）
- **模型**: EVA-02 Base (替代Large版本)
- **输入尺寸**: 224x224 (替代336x336)
- **批次大小**: 16 (充分利用GPU显存)
- **训练轮次**: 50
- **学习率**: 3e-4
- **Drop Path Rate**: 0.05

## 训练启动方式

### 方式1: 自动训练脚本（推荐）

当数据传输完成后，SSH连接到远程服务器并执行：

```bash
ssh EliuaK_Csy@100.90.176.72
cd ~/Plants-Disease-Detection
python scripts/train_eva02_remote.py
```

此脚本会自动完成：
1. 数据解压和预处理
2. 模型训练 (50轮)
3. 测试集推理

### 方式2: 手动分步执行

```bash
ssh EliuaK_Csy@100.90.176.72
cd ~/Plants-Disease-Detection

# 1. 数据准备
python main.py prepare --all --merge-augmented --cleanup

# 2. 模型训练
python main.py train --epochs 50 --batch-size 16 --model eva02_large --wandb --cleanup

# 3. 模型推理
python main.py predict --model-name eva02_large --output-format submit --topk 3
```

## 训练监控

### Weights & Biases (实验追踪)
训练会自动记录到W&B，可在训练完成后查看：
- 损失曲线
- 准确率变化
- 学习率调整
- GPU使用情况

访问方式：https://wandb.ai/plants-disease-detection-eva02

### 实时监控训练进度

在远程服务器上，可以实时查看训练日志：

```bash
tail -f ~/Plants-Disease-Detection/log/training.log
```

或通过SSH在本地监控：

```bash
ssh EliuaK_Csy@100.90.176.72 "tail -f ~/Plants-Disease-Detection/log/training.log"
```

## 预期性能

### EVA-02 Base 模型优势
- **迁移学习能力**: 使用MIM预训练，泛化能力强
- **参数效率**: Base版本参数适中，适合中等规模数据集
- **训练稳定性**: 渐进式解冻策略，只训练最后几层
- **推理速度**: 相比Large版本更快，适合实时应用

### 训练时间预估
- **单轮训练**: ~2-3分钟 (取决于数据集大小)
- **总训练时间**: ~100-150分钟 (50轮)
- **推理时间**: ~5-10分钟

## 输出文件位置

训练完成后，以下文件会在远程服务器上生成：

- **最佳模型**: `~/Plants-Disease-Detection/checkpoints/best/eva02_large/0/best_model.pth.tar`
- **最终模型**: `~/Plants-Disease-Detection/checkpoints/eva02_large/0/_latest_model.pth.tar`
- **训练日志**: `~/Plants-Disease-Detection/log/training.log`
- **预测结果**: `~/Plants-Disease-Detection/submit/prediction.json`

## 后续操作

训练完成后，可以将结果文件传输回本地：

```bash
scp EliuaK_Csy@100.90.176.72:~/Plants-Disease-Detection/checkpoints/best/eva02_large/0/best_model.pth.tar ./checkpoints/best/eva02_large/0/
scp EliuaK_Csy@100.90.176.72:~/Plants-Disease-Detection/submit/prediction.json ./submit/
scp EliuaK_Csy@100.90.176.72:~/Plants-Disease-Detection/log/training.log ./log/
```

## 技术亮点

### EVA-02 架构特点
- **自注意力机制**: Vision Transformer架构
- **MIM预训练**: Masked Image Modeling预训练策略
- **LayerScale**: 稳定深层Transformer训练
- **渐进式解冻**: 只训练最后2个blocks和分类头

### 现代训练技术
- **混合精度训练**: AMP自动混合精度
- **梯度检查点**: 减少显存占用
- **EMA**: 模型参数平滑
- **Focal Loss**: 处理类别不平衡
- **标签平滑**: 提升泛化能力
- **Mixup + CutMix**: 数据增强策略
- **加权采样**: 平衡类别分布

## 注意事项

1. **显存监控**: 使用 `nvidia-smi` 监控GPU显存使用
2. **训练中断**: 如果训练中断，可以从最新checkpoint继续
3. **数据完整性**: 确保数据传输完成后再开始训练
4. **网络稳定性**: 建议在稳定网络环境下进行训练

---

**创建时间**: 2026-04-04
**模型版本**: EVA-02 Base
**训练设备**: RTX 4060 Laptop GPU