# RUN_SMOKE

## 最小依赖
- Python 3.14（见 `pyproject.toml` 的 `requires-python`）
- 无需 GPU、无需模型文件（本 smoke 只做语法编译检查）

## 最小可跑命令（PowerShell）
```powershell
.\scripts\smoke_py_compile.ps1
```

## 说明
- 该脚本会对核心模块（`libs/`, `dataset/`, `models/`, `utils/` 以及 `main.py`, `config.py`）做 `py_compile`。
- 如需指定 Python，可设置环境变量 `PYTHON`：
  ```powershell
  $env:PYTHON = "C:\Path\To\python.exe"
  .\scripts\smoke_py_compile.ps1
  ```
