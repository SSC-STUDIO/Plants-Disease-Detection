#!/bin/bash
# Portable setup helper for a Linux training machine.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-0}"

echo "Plants Disease Detection - remote setup"
echo "Python: $($PYTHON_BIN --version)"

$PYTHON_BIN -m pip install --upgrade pip setuptools wheel
$PYTHON_BIN -m pip install -r requirements-core.txt

if [ "$INSTALL_EXTRAS" = "1" ]; then
  $PYTHON_BIN -m pip install -r requirements-demo.txt
  $PYTHON_BIN -m pip install -r requirements-collector.txt
  $PYTHON_BIN -m pip install -r requirements-extras.txt
fi

$PYTHON_BIN main.py version
echo "Setup complete. Prepare a numeric dataset, then run scripts/train_remote.py."
