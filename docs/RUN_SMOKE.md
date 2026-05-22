# Smoke Test

This smoke test verifies that the core Python modules compile. It does not require a GPU, dataset, checkpoint, or model download.

## Requirements

- Python `>=3.10,<3.15`
- Core dependencies installed with `python -m pip install -r requirements-core.txt`

## PowerShell

```powershell
.\scripts\smoke_py_compile.ps1
```

If PowerShell blocks script execution, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_py_compile.ps1
```

## Notes

- The script compiles `libs/`, `dataset/`, `models/`, `utils/`, `main.py`, and `config.py`.
- It does not generate predictions, checkpoints, or reports.
- Set `PYTHON` to use a specific interpreter:

```powershell
$env:PYTHON = "C:\Path\To\python.exe"
.\scripts\smoke_py_compile.ps1
```
