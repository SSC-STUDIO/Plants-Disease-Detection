# Plants Disease Detection

植物病害检测系统，基于深度学习技术实现高效的病害分类。

## 功能特性

- 支持多种深度学习模型
- 数据增强和预处理
- 训练过程可视化
- 模型评估和测试

## 运行方式

### 标准版本
```bash
python main.py
```

改进版本包含以下优化：
- 自适应学习率调度
- 类别不平衡处理
- 混合精度训练
- 梯度裁剪
- 更详细的训练统计

### 新增命令（全面升级）
```bash
# 查看版本与环境信息（JSON）
python main.py version

# 只输出项目版本
python main.py --version

# 导出当前配置
python main.py config --output ./reports/config.json

# 列出可用模型
python main.py models

# 评估模型（生成指标报告）
python main.py evaluate --model checkpoints/best/efficientnetv2_s/0/best_model.pth.tar

# 数据集统计（输出类别分布与缺失类别）
python main.py stats --data ./data/train --output ./reports/train_stats.json

# 推理增强（输出Top-K与完整概率）
python main.py predict --input ./data/test/images --output ./submit/prediction_full.json --output-format full --topk 5 --save-probs
```

## 安装依赖

```bash
uv sync
```

如果系统没有 `uv`（或 `uv` 不在 PATH）：
```bash
python -m pip install uv
python -m uv sync
```

如果不使用 `uv`，可直接用 `pip`：
```bash
python -m pip install -r requirements.txt
```

可选扩展依赖：
```bash
python -m pip install -r requirements-extras.txt
```

说明：当前 PyTorch 在 Windows 上对 Python 3.14 的支持尚不完善，推荐使用 Python 3.12 或 3.13。

快速创建虚拟环境（无 uv）：
```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```
## 数据集使用 [https://github.com/spytensor/plants_disease_detection]
## 项目结构

```
├── config/          # 配置文件
├── dataset/         # 数据集处理
├── libs/            # 训练和推理库
├── models/          # 模型定义
├── tools/           # 辅助工具
├── utils/           # 实用工具
├── checkpoints/     # 模型检查点
├── main.py          # 主程序入口
└── README.md        # 项目说明
```

## 模型支持

- EfficientNet系列
- DenseNet169
- ConvNeXt
- Swin Transformer
- 混合模型

## 贡献指南

欢迎提交Pull Request改进项目。
