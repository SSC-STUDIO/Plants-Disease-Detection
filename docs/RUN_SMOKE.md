# RUN_SMOKE

## 最小依赖
- Python 3.14（见 `pyproject.toml` 的 `requires-python`）
- 无需 GPU、无需模型文件（本 smoke 只做语法编译检查）

## 最小可跑命令（PowerShell）
```powershell
.\scripts\smoke_py_compile.ps1
```

## 常见问题
- 如果看到执行策略拦截，请使用：
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\smoke_py_compile.ps1
  ```

## 说明
- 该脚本会对核心模块（`libs/`, `dataset/`, `models/`, `utils/` 以及 `main.py`, `config.py`）做 `py_compile`。
- 仅做语法编译检查，不生成模型或推理输出文件。
- 失败时会返回非 0 并打印首个失败文件路径与错误信息。
- 如需指定 Python，可设置环境变量 `PYTHON`：
  ```powershell
  $env:PYTHON = "C:\Path\To\python.exe"
  .\scripts\smoke_py_compile.ps1
  ```
