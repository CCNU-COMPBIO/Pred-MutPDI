#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${PRED_MUTPDI_PYTHON:-}" ]]; then
  PYTHON="$PRED_MUTPDI_PYTHON"
elif [[ -x /home/lenovo/env1/bin/python3 ]]; then
  PYTHON=/home/lenovo/env1/bin/python3
else
  PYTHON="$(command -v python3 || true)"
fi
[[ -n "$PYTHON" && -x "$PYTHON" ]] || { echo '[ERROR] Set PRED_MUTPDI_PYTHON to your Python executable.' >&2; exit 2; }
exec "$PYTHON" -B "$SCRIPT_DIR/predict_prismdna_xgb_DNA.py" "$@"
