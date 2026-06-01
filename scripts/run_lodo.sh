#!/usr/bin/env bash
# run_lodo.sh — Driver para una configuracion LODO (A|B|C|D).
#
# Pasos:
#   1) Genera el split LODO de la config y escribe data/splits.json
#   2) Lanza train_3d.py con configs/train_lodo.yaml
#      (recreate_splits=false -> usa el splits.json que acabamos de escribir)
#   3) Localiza el ultimo run dentro de outputs/checkpoints/lodo/
#   4) Lanza evaluate_3d.py sobre el split TEST y guarda en
#      outputs/evaluation/lodo_<CONFIG>/
#
# Uso:
#   bash scripts/run_lodo.sh A
#
# Recomendado lanzarlo en background y monitorizar con:
#   tail -f outputs/checkpoints/lodo/<timestamp>/...

set -euo pipefail

CONFIG="${1:?Falta config (A|B|C|D)}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/uaxlab/miniconda3/envs/igsan/bin/python
CFG_YAML=configs/train_lodo.yaml
CKPT_BASE=outputs/checkpoints/lodo
EVAL_DIR="outputs/evaluation/lodo_${CONFIG}"

echo "============================================================"
echo "[LODO ${CONFIG}] start: $(date -Iseconds)"
echo "============================================================"

echo "[1/3] Generando splits LODO ${CONFIG}..."
"$PY" -m src.audit.make_lodo_splits --config "$CONFIG"

# Marca cuantas subcarpetas hay antes para detectar la nueva
PREV_RUNS=$(ls -1 "$CKPT_BASE" 2>/dev/null | wc -l || echo 0)

echo
echo "[2/3] Entrenando CNN 3D con configs/train_lodo.yaml..."
"$PY" -m src.training.train_3d --config "$CFG_YAML"

# Coger el run mas reciente (el que acaba de crearse)
LATEST_RUN=$(ls -1t "$CKPT_BASE" | head -1)
CKPT="${CKPT_BASE}/${LATEST_RUN}/best.pt"
echo
echo "[2/3] Run: ${CKPT_BASE}/${LATEST_RUN}"
echo "       Checkpoint: ${CKPT}"

if [[ ! -f "$CKPT" ]]; then
  echo "AVISO: no se encontro best.pt (no se cumplio min_sensitivity_for_save)."
  echo "       Saltando evaluacion automatica. Inspecciona ${CKPT_BASE}/${LATEST_RUN}/history.json"
  exit 0
fi

echo
echo "[3/3] Evaluando en split test y guardando en ${EVAL_DIR}..."
mkdir -p "$EVAL_DIR"
"$PY" -m src.evaluation.evaluate_3d \
  --config "$CFG_YAML" \
  --checkpoint "$CKPT" \
  --split test \
  --output-dir "$EVAL_DIR"

echo
echo "============================================================"
echo "[LODO ${CONFIG}] done: $(date -Iseconds)"
echo "Resultados: ${EVAL_DIR}"
echo "============================================================"
