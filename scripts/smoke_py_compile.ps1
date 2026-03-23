$ErrorActionPreference = "Stop"

$python = "python"
if ($env:PYTHON) {
  $python = $env:PYTHON
}

$targets = @(
  "libs",
  "dataset",
  "models",
  "utils",
  "main.py",
  "config.py"
)

$script = @'
import sys
import os
import py_compile

def compile_path(path: str) -> None:
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in files:
                if filename.endswith(".py"):
                    file_path = os.path.join(root, filename)
                    try:
                        py_compile.compile(file_path, doraise=True)
                    except Exception as exc:
                        print(f"[SMOKE] compile failed: {file_path}\n{exc}")
                        sys.exit(1)
    elif path.endswith(".py") and os.path.isfile(path):
        try:
            py_compile.compile(path, doraise=True)
        except Exception as exc:
            print(f"[SMOKE] compile failed: {path}\n{exc}")
            sys.exit(1)

for target in sys.argv[1:]:
    compile_path(target)

print("[SMOKE] py_compile ok")
'@

$tempScript = Join-Path $env:TEMP "smoke_py_compile.py"
Set-Content -Path $tempScript -Value $script -Encoding UTF8

& $python $tempScript @targets
$exitCode = $LASTEXITCODE
Remove-Item $tempScript -ErrorAction SilentlyContinue

if ($exitCode -ne 0) {
  exit $exitCode
}
