# CHANGELOG

## [0.2.0] - 2026-03-23

### Added
- 新增 `export` 子命令，支持将训练好的模型导出为 ONNX 格式，便于部署和跨平台推理。
- ONNX 导出支持自定义输入尺寸、opset 版本、动态批次大小和模型验证。
- 新增 pytest 测试套件，覆盖配置默认值、模型注册/实例化、工具函数、progressive resizing 辅助逻辑，以及 `export` CLI 子命令解析。

### Fixed
- 修复 `main.py` 中 `prepare_data()` 调用不存在的 `_load_setup_data()` 导致 NameError 的问题，改为直接使用已导入的 `setup_data`。
- 修复 `models/model.py` 中 `get_densenet169()` 使用已弃用的 `pretrained=True` 参数，改为与其他模型一致的 `_build_torchvision_model()` + weights enum 模式。
- 移除 `dataset/data_prep.py` 中 `extract_images_directly()` 空实现桩代码及其调用点，消除无效的快速路径分支。
- 修复 `config.py` 中 `img_weight` 字段拼写错误，重命名为 `img_width`，并同步更新所有引用该字段的代码。
- 修正 `config.py` 中关于 `progressive_resizing` 的注释说明：该功能已在训练流程中实现，仅默认关闭。

### Changed
- `num_workers` 默认值从硬编码的 `32` 改为 `'auto'`，自动适配不同机器的 CPU 核心数。
- `progressive_resizing` 默认值从 `True` 改为 `False`，因为该功能默认关闭可避免训练时意外改变输入尺寸。

## [0.1.9] - 2026-03-22

### Added
- 新增 `tools/dataset_collector/build_dataset_bundle.py`，可把项目自带数据和本地公开植物病害数据组装成统一的数据包目录与 `manifest.json`。
- 新增 `docs/DATASET_STRATEGY.md`，记录推荐数据集组合、项目自制数据定位以及本地约 10GB 数据包方案。
- 新增 `tools/dataset_collector/convert_ai_challenger.py`，把 AI Challenger PDR2018 压缩包转换成项目可直接训练的分类目录。
- 新增 `tools/dataset_collector/extract_plantdoc_crops.py`，把 PlantDoc VOC 框标注裁成分类病斑图块。

### Changed
- 默认分类骨干升级为 `convnextv2_base_384`，替换原来的 `efficientnetv2_s`。
- 默认训练超参数改为更适合大型现代骨干网络的配置，包括启用预训练权重、降低批次大小、调低学习率并提高权重衰减。
- 训练图像增广升级为 `RandomResizedCrop + RandAugment + ColorJitter + RandomErasing`。
- 默认启用按类别频次加权的 `WeightedRandomSampler` 以改善长尾训练分布。
- 推理新增 1 到 4 视角 TTA 概率平均，并暴露 `--tta-views` CLI 参数。
- 评估链路新增 1 到 4 视角 TTA 概率平均，并支持 `--tta-views`。
- 数据集采集器的 headless CLI 现在支持质量过滤、重复图像去除、尺寸过滤、可配置搜索源和 `manifest.json` 输出。
- README 补充数据集策略入口、本地 10GB 数据包路径和新的 headless 数据清洗示例。
- 数据集文档现在同时记录 AI Challenger 转换结果和 PlantDoc 分类裁剪结果。

### Fixed
- 继续训练、推理和评估加载 checkpoint 时不再重复初始化预训练权重，减少不必要的下载和环境依赖。
- 评估与推理现在可以从 checkpoint 路径自动推断 `convnextv2_base_384` 模型架构。
- 修复数据集采集器在仅使用 headless 模式时仍强依赖 `PyQt6` 的问题。
- 修复本地导入阶段在 Windows 上因大小写扩展名重复扫描导致同一图像重复计数的问题。

## [0.1.8] - 2026-03-21

### Added
- Track repository maintainer guidance in CLAUDE.md for local coding-agent workflows.

### Changed
- Ignore local .mcp.json machine-specific MCP configuration so git status reflects real repo work.

### Fixed
- Restored a trustworthy working-tree baseline by classifying both previously untracked files.

## [0.1.7] - 2026-03-14
### Changed
- README 补充 smoke 入口与 Python 3.14 运行环境说明，便于直接按仓库文档完成最小语法编译验证。
- `libs/inference.py` 在输入目录或文件不存在时改为输出绝对路径，并记录跳过的非图片条目数量，降低排查输入问题的成本。

### Evidence
- Smoke 编译日志：`C:\Users\96152\.openclaw\workspace\attachments\plants-disease-detection\smoke-py-compile-latest.log`

## [0.1.6] - 2026-03-12
### Changed
- RUN_SMOKE 补充 smoke 仅做语法编译检查的说明。

## [0.1.5] - 2026-03-12
### Changed
- README 增加 CHANGELOG 入口提示，方便查看变更记录。

## [0.1.4] - 2026-03-12
### Changed
- RUN_SMOKE 增加执行策略拦截的 PowerShell 绕过说明。

## [0.1.3] - 2026-03-12
### Changed
- README 补充运行环境说明（Python 3.14.x 与 requires-python 一致）。

## [0.1.2] - 2026-03-12
### Changed
- smoke 说明补充失败返回码与错误输出提示。

## [0.1.1] - 2026-03-12
### Changed
- 推理路径校验日志输出为绝对路径，便于定位无效输入目录/文件。

