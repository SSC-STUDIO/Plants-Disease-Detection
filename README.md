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

### Smoke（语法编译）
```powershell
.\scripts\smoke_py_compile.ps1
```
详见 `docs/RUN_SMOKE.md`；变更记录见 `CHANGELOG.md`。

改进版本包含以下优化：
- 自适应学习率调度
- 类别不平衡处理
- 混合精度训练
- 梯度裁剪
- 更详细的训练统计
- 默认主力模型升级为 ConvNeXt V2 Base 384
- 加入 Weighted Random Sampler 做类别均衡采样
- 推理支持最多 4 视角 TTA 概率平均
- 训练增广升级为 RandomResizedCrop + RandAugment + ColorJitter + RandomErasing

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
python main.py evaluate --model checkpoints/best/convnextv2_base_384/0/best_model.pth.tar

# 评估时同样支持 TTA
python main.py evaluate --model checkpoints/best/convnextv2_base_384/0/best_model.pth.tar --tta-views 4

# 数据集统计（输出类别分布与缺失类别）
python main.py stats --data ./data/train --output ./reports/train_stats.json

# 推理增强（输出Top-K与完整概率）
python main.py predict --input ./data/test/images --output ./submit/prediction_full.json --output-format full --topk 5 --save-probs

# 使用 4 视角 TTA 提升推理稳健性
python main.py predict --input ./data/test/images --tta-views 4

# 如类别分布很不均衡，显式启用加权采样训练
python main.py train --enable-weighted-sampler
```

## 安装依赖

> 运行环境：Python 3.14.x（与 `pyproject.toml` 的 requires-python 保持一致）。

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

说明：当前推荐使用最新稳定版 Python 3.14.x（截至 2026-02-03 为 3.14.3）。若在 Windows 上遇到 PyTorch 兼容性问题，可临时回退到 3.13。

快速创建虚拟环境（无 uv）：
```bash
py -3.14 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 数据集方案

数据集策略文档见 `docs/DATASET_STRATEGY.md`。

当前已经实际组装好的本地数据包在：

- `C:\Users\96152\My-Project\Datasets\PlantDisease-10GB-Bundle`

它会把项目自带数据和已有公开数据汇总到一个统一入口，当前逻辑体量约 `11.005 GB`。数据包清单见：

- `C:\Users\96152\My-Project\Datasets\PlantDisease-10GB-Bundle\manifest.json`

重新生成该数据包：

```powershell
python tools/dataset_collector/build_dataset_bundle.py
```

此外已经生成两个可直接训练的处理后数据集：

- `C:\Users\96152\My-Project\Datasets\Processed\AI-Challenger-PDR2018-Classification`
- `C:\Users\96152\My-Project\Datasets\Processed\PlantDoc-Crops-Classification`

用新的 headless 采集器清洗本地自制数据：

```powershell
python tools/dataset_collector/app.py `
  --headless `
  --source-dir C:\path\to\raw_leaf_images `
  --output-dir C:\path\to\prepared_dataset `
  --quality-filter `
  --deduplicate `
  --enable-size-filter `
  --generate-manifest
```

把 AI Challenger 压缩包转成分类目录：

```powershell
python tools/dataset_collector/convert_ai_challenger.py --overwrite
```

把 PlantDoc 检测框转成分类裁剪图：

```powershell
python tools/dataset_collector/extract_plantdoc_crops.py --overwrite
```

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

- ConvNeXt V2 Base 384（默认推荐）
- EfficientNet 系列
- DenseNet169
- ConvNeXt
- Swin Transformer
- 混合模型

## 当前默认技术路线

- 默认训练骨干：`convnextv2_base_384`
- 默认启用预训练迁移学习
- 默认更保守的批次大小：`8`
- `timm` 不可用时会自动回退到 `convnext_small`
- 默认启用 `WeightedRandomSampler` 缓解长尾类别不平衡
- 默认训练增广升级为 `RandomResizedCrop + RandAugment + ColorJitter + RandomErasing`
- 默认推理启用 `4` 视角 TTA
- 继续训练/推理/评估时不再重复拉取预训练权重，直接加载 checkpoint

## 贡献指南

欢迎提交Pull Request改进项目。
