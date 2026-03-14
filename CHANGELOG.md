# Changelog

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
