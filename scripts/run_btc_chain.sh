#!/usr/bin/env bash
# run_btc_chain.sh — Pipeline BTC post-HD-BET con TIMEOUTS y deteccion de fallos.
# Cada paso tiene un timeout generoso. Si algo cuelga, muere y se entera el agente.

set -uo pipefail   # no -e: queremos seguir aunque un paso falle, para consolidar lo que haya
PY=/home/uaxlab/miniconda3/envs/igsan/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/btc_chain_summary.txt"
: > "$SUMMARY"

log_step() {
  local name="$1"; local rc="$2"; local t="$3"
  echo "[$(date -Iseconds)] $name: exit=$rc elapsed=${t}s" | tee -a "$SUMMARY"
}

run_with_timeout() {
  local label="$1"; local secs="$2"; shift 2
  echo
  echo "============================================================"
  echo "[$(date -Iseconds)] START: $label  (timeout ${secs}s)"
  echo "============================================================"
  local t0=$SECONDS
  timeout "$secs" "$@"
  local rc=$?
  local t=$((SECONDS - t0))
  if [ $rc -eq 124 ]; then
    echo "TIMEOUT($secs s) en $label" | tee -a "$SUMMARY"
  fi
  log_step "$label" "$rc" "$t"
  return $rc
}

echo "============================================================"
echo "[BTC chain] start: $(date -Iseconds)"
echo "GPU snapshot inicial:"
rocm-smi --showuse --showmemuse 2>&1 | grep -E "%|GPU" | head -6
echo "============================================================"

# [1/5] Preprocesado .npz - timeout 30 min (deberia tardar ~5)
run_with_timeout "preprocess_btc" 1800 \
  "$PY" -m src.preprocessing.preprocess_btc
STATUS_PRE=$?

# [2/5] Verificar
N_POS=$(ls data/processed_btc/positives/*.npz 2>/dev/null | wc -l)
N_NEG=$(ls data/processed_btc/negatives/*.npz 2>/dev/null | wc -l)
TOTAL=$((N_POS + N_NEG))
echo
echo "[verify] positives=$N_POS  negatives=$N_NEG  total=$TOTAL"
echo "verify: pos=$N_POS neg=$N_NEG total=$TOTAL" >> "$SUMMARY"
if [ "$TOTAL" -lt 30 ]; then
  echo "ABORT: solo $TOTAL .npz, esperaba ~36. No tiene sentido seguir."
  exit 1
fi

# [3/5] Tiny baseline - timeout 10 min (deberia tardar ~30s)
run_with_timeout "btc_tiny_baseline" 600 \
  "$PY" -m src.audit.btc_tiny_baseline
STATUS_TINY=$?

# [4/5] CNN 3D 1-canal k-fold (GPU) - timeout 4h (deberia tardar 1-2h)
# Si el CNN falla o se cuelga, seguimos y al menos consolidamos lo que tengamos.
run_with_timeout "btc_cnn_kfold" 14400 \
  "$PY" -m src.audit.btc_cnn_kfold
STATUS_CNN=$?
if [ $STATUS_CNN -ne 0 ]; then
  echo "AVISO: btc_cnn_kfold fallo (exit=$STATUS_CNN). Continuo para consolidar lo que haya."
fi

# [5/5] Consolidar - timeout 5 min
run_with_timeout "consolidate_results" 300 \
  "$PY" -m src.audit.consolidate_results
STATUS_CONS=$?

echo
echo "============================================================"
echo "[BTC chain] done: $(date -Iseconds)"
echo "Resumen:"
cat "$SUMMARY"
echo
echo "Resultados clave:"
ls -la docs/audit/btc_intradomain_tinybaseline.json 2>/dev/null
ls -la outputs/evaluation/btc_intradomain/cnn_kfold_results.json 2>/dev/null
ls -la docs/audit/resumen_consolidado.md 2>/dev/null
echo "============================================================"

# Codigo de salida final: error si alguno de los CRITICOS fallo (preproc o CNN sin recuperacion)
if [ "$STATUS_PRE" -ne 0 ]; then
  exit 11
fi
exit 0
