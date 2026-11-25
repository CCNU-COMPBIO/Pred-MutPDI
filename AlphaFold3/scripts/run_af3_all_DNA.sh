#!/usr/bin/env bash
set -euo pipefail
# Internal AF3 stage; the public entry point is predict_DNA.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDB=""; CHAIN=""; MUT=""; PDB_FILE=""; RESOURCE_DIR="$SCRIPT_DIR"; ALLOW_MISSING=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -PDB) PDB="${2:?Missing PDB value}"; shift 2 ;;
    -CHAIN) CHAIN="${2:?Missing chain value}"; shift 2 ;;
    -Mut) MUT="${2:?Missing mutation value}"; shift 2 ;;
    --pdb-path) PDB_FILE="${2:?Missing PDB path}"; shift 2 ;;
    --resource-dir) RESOURCE_DIR="${2:?Missing resource directory}"; shift 2 ;;
    --allow-missing-es) ALLOW_MISSING=1; shift ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$PDB" && -n "$CHAIN" && -n "$MUT" && -f "$PDB_FILE" ]] || {
  echo '[ERROR] Missing mutation arguments or input PDB.' >&2; exit 2; }
[[ -n "${PRED_MUTPDI_RESULT_FD:-}" ]] || {
  echo '[ERROR] Internal result channel missing. Use predict_DNA.sh.' >&2; exit 2; }
AF3_DIR="${PRED_MUTPDI_AF3_DIR:-${RESOURCE_DIR}/alphafold3}"
[[ -f "${AF3_DIR}/run_alphafold.py" ]] || {
  echo "[ERROR] AF3 runner not found: ${AF3_DIR}/run_alphafold.py" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${PRED_MUTPDI_AF3_PYTHON:-}" ]]; then
  AF3_PY="$PRED_MUTPDI_AF3_PYTHON"
else
  command -v conda >/dev/null || { echo '[ERROR] Set PRED_MUTPDI_AF3_PYTHON or initialize conda.' >&2; exit 2; }
  # Avoid depending on the caller's shell activation state.
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${PRED_MUTPDI_AF3_ENV:-af3}"
  AF3_PY="${CONDA_PREFIX}/bin/python"
fi
[[ -x "$AF3_PY" ]] || { echo "[ERROR] AF3 Python is not executable: $AF3_PY" >&2; exit 2; }
TMP_BASE="${PRED_MUTPDI_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$TMP_BASE"
WORK_TMP="$(mktemp -d "${TMP_BASE%/}/predmutpdi_af3_DNA.XXXXXX")"
chmod 700 "$WORK_TMP"
cleanup() { rm -rf -- "$WORK_TMP"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
JSON_PATH="${WORK_TMP}/input_DNA.json"
MAPPING_PATH="${WORK_TMP}/mapping_DNA.json"
OUT_DIR="${WORK_TMP}/af3_outputs_DNA"
AF3_LOG="${WORK_TMP}/af3_DNA.log"
"$AF3_PY" -B "${SCRIPT_DIR}/run_af3_mutation_DNA.py" \
  -PDB "$PDB" -CHAIN "$CHAIN" -Mut "$MUT" --pdb-path "$PDB_FILE" \
  --json_path "$JSON_PATH" --mapping-json "$MAPPING_PATH"
AF3_ARGS=("--json_path=$JSON_PATH" "--output_dir=$OUT_DIR")
[[ -n "${PRED_MUTPDI_AF3_MODEL_DIR:-}" ]] && AF3_ARGS+=("--model_dir=$PRED_MUTPDI_AF3_MODEL_DIR")
[[ -n "${PRED_MUTPDI_AF3_DB_DIR:-}" ]] && AF3_ARGS+=("--db_dir=$PRED_MUTPDI_AF3_DB_DIR")
set +e
(cd "$AF3_DIR" && "$AF3_PY" -B run_alphafold.py "${AF3_ARGS[@]}") >"$AF3_LOG" 2>&1
AF3_RC=$?
set -e
if [[ "$AF3_RC" -ne 0 ]]; then
  echo "[ERROR] AlphaFold3 failed (exit $AF3_RC)." >&2
  if [[ "${PRED_MUTPDI_DEBUG:-0}" == 1 ]]; then tail -n 120 "$AF3_LOG" >&2; fi
  exit "$AF3_RC"
fi
ES_ARGS=()
[[ "$ALLOW_MISSING" == 1 ]] && ES_ARGS+=(--allow-missing-es)
"$AF3_PY" -B "${SCRIPT_DIR}/calculating_es_DNA.py" \
  -PDB "$PDB" -CHAIN "$CHAIN" -Mut "$MUT" --pdb-path "$PDB_FILE" \
  --af3-root "$OUT_DIR" --mapping-json "$MAPPING_PATH" "${ES_ARGS[@]}"
