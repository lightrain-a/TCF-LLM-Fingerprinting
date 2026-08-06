#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TCF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TCF_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
N_WORKERS="${N_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-200}"
CHUNK_STEPS="${CHUNK_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TOPK="${TOPK:-64}"
POLL_SEC="${POLL_SEC:-120}"
TARGET_DONE_COUNT="${TARGET_DONE_COUNT:-20}"
SPARSE_STEPS="${SPARSE_STEPS:-0,1,5,10,20,40,60,80,100,120,140,160,180,200}"
SEED="${SEED:-20260418}"

MODEL_CSV="${MODEL_CSV:-${TCF_ROOT}/configs/qwen3_1p7b_instruct_runtime_models.csv}"
WORK_DIR="${WORK_DIR:-${TCF_ROOT}/runs/qwen3_1p7b_instruct_c2_step${MAX_STEPS}_seed${SEED}}"
GEN_OUT="${GEN_OUT:-${WORK_DIR}/generation}"
EVAL_OUT="${EVAL_OUT:-${WORK_DIR}/aligned_sparse_eval}"
TARGET_SRC_DIR="${TARGET_SRC_DIR:-${REPO_ROOT}/results/mmlu_cf/gcg_pipeline_csv_v13_C2_official_space_questionprefix_wrongtop1_qwen3_1p7b_4cat5each_step200_2w_bs128_topk64}"

if [[ "${MODE}" != "generate" && "${MODE}" != "eval" && "${MODE}" != "all" ]]; then
  echo "Unsupported mode=${MODE}; use generate|eval|all" >&2
  exit 2
fi
if [[ ! -f "${MODEL_CSV}" ]]; then
  echo "Missing runtime model csv: ${MODEL_CSV}" >&2
  exit 3
fi
MODEL_CSV="$(readlink -f "${MODEL_CSV}")"

SOURCE_MODEL_PATH="$("${PYTHON_BIN}" - "${MODEL_CSV}" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
src = df[df["model_group"].astype(str).eq("source")]
if src.empty:
    raise SystemExit("missing source row")
print(str(src.iloc[0]["model_path"]))
PY
)"
SOURCE_MODEL_NAME="$("${PYTHON_BIN}" - "${MODEL_CSV}" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
src = df[df["model_group"].astype(str).eq("source")]
if src.empty:
    raise SystemExit("missing source row")
print(str(src.iloc[0]["model_name"]))
PY
)"

mkdir -p "${GEN_OUT}" "${EVAL_OUT}" "${WORK_DIR}/logs"

prepare_targets() {
  cp "${TARGET_SRC_DIR}/group_C2_targets.csv" "${GEN_OUT}/group_C2_targets.csv"
  cp "${TARGET_SRC_DIR}/base_filtered_pool.csv" "${GEN_OUT}/base_filtered_pool.csv"
  "${PYTHON_BIN}" - "${GEN_OUT}/group_C2_targets.csv" "${SOURCE_MODEL_NAME}" "${ALLOW_TARGET_SOURCE_MISMATCH:-0}" <<'PY'
import sys
import pandas as pd

target_csv, expected, allow = sys.argv[1:4]
df = pd.read_csv(target_csv, nrows=50)
cols = [c for c in ("source_model_for_target_selection", "attack_source_model") if c in df.columns]
if not cols:
    raise SystemExit(f"target csv missing source metadata columns: {target_csv}")
bad = {}
for col in cols:
    vals = sorted({str(v) for v in df[col].dropna().unique()})
    if vals != [expected]:
        bad[col] = vals
if bad and str(allow).strip() != "1":
    raise SystemExit(
        "target source mismatch for TCF: "
        f"expected {expected}, got {bad}. "
        "Set TARGET_SRC_DIR to a source-matched target directory, or ALLOW_TARGET_SOURCE_MISMATCH=1 only for diagnostics."
    )
PY
}

count_done() {
  "${PYTHON_BIN}" - "${GEN_OUT}/progress" <<'PY'
import csv, sys
from pathlib import Path
done = 0
for path in Path(sys.argv[1]).glob("group_C2_worker_*_manifest.csv"):
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if str(row.get("status", "")).strip() == "done":
                done += 1
print(done)
PY
}

combine_manifests() {
  "${PYTHON_BIN}" - "${GEN_OUT}/progress" "${EVAL_OUT}/group_C2_combined_manifest.csv" <<'PY'
import sys
from pathlib import Path
import pandas as pd
progress = Path(sys.argv[1])
out = Path(sys.argv[2])
frames = [pd.read_csv(p) for p in sorted(progress.glob("group_C2_worker_*_manifest.csv"))]
if not frames:
    fallback = progress.parent / "group_C2_suffix_manifest.csv"
    if fallback.is_file():
        frames = [pd.read_csv(fallback)]
if not frames:
    raise SystemExit(f"no manifests under {progress}")
df = pd.concat(frames, ignore_index=True)
df = df.drop_duplicates(subset=["sample_id"], keep="last") if "sample_id" in df.columns else df
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"[OK] combined_manifest={out} rows={len(df)}")
PY
}

if [[ "${MODE}" == "generate" || "${MODE}" == "all" ]]; then
  prepare_targets
  echo "seed=${SEED}" > "${WORK_DIR}/logs/repro_seed.txt"
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    QWEN3_LOCAL_MODEL="${SOURCE_MODEL_PATH}" \
    CONFIG="configs/individual_qwen3_local_official_plain.py" \
    N_WORKERS="${N_WORKERS}" \
    MAX_STEPS="${MAX_STEPS}" \
    CHUNK_STEPS="${CHUNK_STEPS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    TOPK="${TOPK}" \
    PROMPT_STYLE="official" \
    CONTINUATION_STYLE="space" \
    TRIGGER_PLACEMENT="question_prefix" \
    OFFICIAL_DEMO_CSV="${GEN_OUT}/base_filtered_pool.csv" \
    OFFICIAL_TOKENIZER_PATH="${SOURCE_MODEL_PATH}" \
    bash "${REPO_ROOT}/results/mmlu_cf/run_group_gcg_parallel.sh" C2 "${GEN_OUT}"

  while true; do
    done_count="$(count_done)"
    echo "[$(date -Is)] TCF Qwen3-1.7B Instruct C2 done=${done_count}/${TARGET_DONE_COUNT}" | tee -a "${WORK_DIR}/logs/ours_wait.log"
    if [[ "${done_count}" -ge "${TARGET_DONE_COUNT}" ]]; then
      break
    fi
    sleep "${POLL_SEC}"
  done
fi

if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then
  prepare_targets
  combine_manifests
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" "${REPO_ROOT}/results/mmlu_cf/eval_from_excel_sparse_steps.py" \
    --group_c2_targets_csv "${GEN_OUT}/group_C2_targets.csv" \
    --manifest_c2_csv "${EVAL_OUT}/group_C2_combined_manifest.csv" \
    --groups C2 \
    --sparse_steps "${SPARSE_STEPS}" \
    --model_csv "${MODEL_CSV}" \
    --source_model_name "${SOURCE_MODEL_NAME}" \
    --qwen_base_path "${SOURCE_MODEL_PATH}" \
    --prompt_style official \
    --official_demo_csv "${GEN_OUT}/base_filtered_pool.csv" \
    --official_ntrain 5 \
    --official_max_prompt_tokens 2048 \
    --official_prompt_wrapper plain \
    --continuation_style space \
    --trigger_placement question_prefix \
    --eval_mode logprob \
    --output_dir "${EVAL_OUT}" \
    --resume
fi

echo "[DONE] TCF Qwen3-1.7B Instruct mode=${MODE} seed=${SEED} work_dir=${WORK_DIR}"
