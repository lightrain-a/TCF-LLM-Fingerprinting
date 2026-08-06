#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_STEPS="${MAX_STEPS:-200}"
SEED="${SEED:-20260418}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/runs/qwen3_1p7b_instruct_c2_step${MAX_STEPS}_seed${SEED}}"
MODEL_CSV="${MODEL_CSV:-${SCRIPT_DIR}/configs/qwen3_1p7b_instruct_runtime_models.csv}"
SPARSE_STEPS="${SPARSE_STEPS:-0,1,5,10,20,40,60,80,100,120,140,160,180,200}"
SELECT_CATEGORIES="${SELECT_CATEGORIES:-Chemistry_val,Health_dev,Physics_val,Engineering_dev}"
SELECT_RULE="${SELECT_RULE:-scm}"
COMPUTE_SCM="${COMPUTE_SCM:-1}"
SCM_CSV="${SCM_CSV:-${WORK_DIR}/source_scm/source_scm_by_sample_step.csv}"
PROFLINGO_CSV="${PROFLINGO_CSV:-${SCRIPT_DIR}/baselines/proflingo_results.csv}"
TRAP_CSV="${TRAP_CSV:-${SCRIPT_DIR}/baselines/trap_results.csv}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/auc}"

case "${MODE}" in
  generate|eval|all)
    PYTHON_BIN="${PYTHON_BIN}" MODEL_CSV="${MODEL_CSV}" WORK_DIR="${WORK_DIR}" MAX_STEPS="${MAX_STEPS}" \
      SPARSE_STEPS="${SPARSE_STEPS}" SEED="${SEED}" \
      "${SCRIPT_DIR}/scripts/run_qwen3_1p7b_instruct_tcf.sh" "${MODE}"
    ;;
  select-step)
    ;;
  auc)
    ;;
  *)
    echo "Unsupported mode=${MODE}; use generate|eval|select-step|auc|all" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "select-step" || "${MODE}" == "all" ]]; then
  if [[ "${COMPUTE_SCM}" == "1" && ! -f "${SCM_CSV}" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/compute_source_scm.py" \
      --task-table-csv "${WORK_DIR}/aligned_sparse_eval/sparse_eval_task_table.csv" \
      --model-csv "${MODEL_CSV}" \
      --official-demo-csv "${WORK_DIR}/generation/base_filtered_pool.csv" \
      --categories "${SELECT_CATEGORIES}" \
      --source-model-name "Qwen3-1.7B" \
      --trigger-placement question_prefix \
      --continuation-style space \
      --official-prompt-wrapper plain \
      --out-csv "${SCM_CSV}"
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/select_scm_step.py" \
    --detail-csv "${WORK_DIR}/aligned_sparse_eval/sparse_eval_detail.csv" \
    --task-table-csv "${WORK_DIR}/aligned_sparse_eval/sparse_eval_task_table.csv" \
    --scm-csv "${SCM_CSV}" \
    --categories "${SELECT_CATEGORIES}" \
    --rule "${SELECT_RULE}" \
    --out-dir "${WORK_DIR}/step_selection"
fi

if [[ "${MODE}" == "auc" || "${MODE}" == "all" ]]; then
  if [[ -f "${WORK_DIR}/step_selection/selected_step.csv" ]]; then
    OURS_STEP="$("${PYTHON_BIN}" - "${WORK_DIR}/step_selection/selected_step.csv" <<'PY'
import pandas as pd, sys
print(int(pd.read_csv(sys.argv[1]).iloc[0]["step"]))
PY
)"
  else
    OURS_STEP="${OURS_STEP:-${MAX_STEPS}}"
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/build_tcf_auc.py" \
    --model-key "qwen3_1p7b_instruct_tcf_seed${SEED}" \
    --model-csv "${MODEL_CSV}" \
    --proflingo-csv "${PROFLINGO_CSV}" \
    --trap-csv "${TRAP_CSV}" \
    --ours-csv "${WORK_DIR}/aligned_sparse_eval/sparse_eval_detail.csv" \
    --ours-step "${OURS_STEP}" \
    --out-dir "${OUT_DIR}"
fi

echo "[DONE] TCF Qwen3-1.7B Instruct mode=${MODE} seed=${SEED} work_dir=${WORK_DIR}"
