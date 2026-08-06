# Targeted Counterfactual Fingerprinting for Black-Box LLM Ownership Verification

This repository contains the public code and supplementary material for **Targeted Counterfactual Fingerprinting for Black-Box LLM Ownership Verification**.

Repository: <https://github.com/lightrain-a/TCF-LLM-Fingerprinting>

This artifact provides a reproduction wrapper for the TCF implementation on the Qwen3-1.7B Instruct lineage. It focuses on one representative source-family setting rather than rerunning every source family and baseline reported in the paper. It covers three stages:

1. TCF fingerprint generation with GCG over C2/wrong-top1 MMLU-CF targets.
2. Source-only SCM computation and SCM-based step selection.
3. AUC evaluation using source+family models as positives and independent models as negatives.

## Method Sketch

TCF first selects source-matched MMLU-CF questions for `Qwen/Qwen3-1.7B`. For this reproduction package, each target answer comes from the released C2/wrong-top1 target table, i.e., the source model's highest-logprob wrong option among the valid labels. The paper describes the target-selection principle more generally as selecting a valid target different from the clean source answer and checking it with the source-side SCM condition. GCG then searches for a short optimized text prefix that moves the source model toward the selected target answer under the official MMLU multiple-choice prompt.

The evaluator inserts each step's optimized text back into the question prefix and scores A/B/C/D continuations with log probabilities. It writes per-sample/per-model step traces to `sparse_eval_detail.csv` and the corresponding optimized-prefix table (called a suffix table in the implementation files) to `sparse_eval_task_table.csv`.

Step selection follows the paper's source-model counterfactual margin. For target label `t`, prompt `p`, and constrained answer distribution `q_M(.|p)` over A/B/C/D, the target log-odds is:

```text
L_M(p,t) = log(q_M(t|p) / (1 - q_M(t|p))).
```

For a clean prompt `p0(x)` and a prefixed prompt `pu(x)`, source-model counterfactual margin is:

```text
SCM = Gamma(x,u,t) = min(L_M0(pu(x),t), -L_M0(p0(x),t)).
```

`scripts/compute_source_scm.py` computes this quantity using only the protected source model. It scores A/B/C/D continuations under the official MMLU prompt, normalizes the four log-probabilities into a constrained distribution, clips probabilities before log-odds, and writes `source_scm_by_sample_step.csv`. `select-step` uses `SELECT_RULE=scm` by default.

## Files

- `run_tcf_qwen3_1p7b_instruct.sh`: top-level driver for `generate`, `eval`, `select-step`, `auc`, or `all`.
- `scripts/run_qwen3_1p7b_instruct_tcf.sh`: generation/evaluation wrapper.
- `scripts/compute_source_scm.py`: source-only SCM computation from clean/prefixed target log-odds.
- `scripts/select_scm_step.py`: SCM step selection.
- `scripts/build_tcf_auc.py`: score and AUC aggregation.
- `configs/qwen3_1p7b_instruct_runtime_models.csv`: anonymized runtime model list.
- `environment.yml`: conda environment snapshot.
- `requirements.txt`: lightweight pip dependency reference.
- `supplementary_material.pdf`: supplementary material.
- `repro_results/qwen3_1p7b_instruct_seed20260418/`: one reproduced aggregate result.

## Environment

Strict reproduction uses conda:

```bash
conda env create -n trap-tcf-repro -f environment.yml
conda activate trap-tcf-repro
```

If you use an existing environment, install the lightweight dependency list:

```bash
pip install -r requirements.txt
```

The scripts default to the active shell's `python`. Override if needed:

```bash
export PYTHON_BIN=/path/to/python
```

## Model Download

The runtime CSV uses relative local paths of the form `models/<org>/<repo>`. The Hugging Face source is in the `model_id` column. Example:

```bash
huggingface-cli download Qwen/Qwen3-1.7B \
  --local-dir models/Qwen/Qwen3-1.7B
```

Models used in this Qwen3 Instruct pool:

| role        | model_id                                       |
| ----------- | ---------------------------------------------- |
| source      | `Qwen/Qwen3-1.7B`                              |
| family      | `contextboxai/Qwen3-1.7B-FC`                   |
| family      | `swapnillo/Bangla-AI-1.7B`                     |
| family      | `activeDap/Qwen3-1.7B_tldr`                    |
| family      | `prithivMLmods/Demeter-LongCoT-Qwen3-1.7B`     |
| family      | `prithivMLmods/Panacea-MegaScience-Qwen3-1.7B` |
| family      | `wzx111/Qwen3-1.7B-Open-R1-GRPO-Baseline`      |
| independent | `meta-llama/Llama-2-7b-hf`                     |
| independent | `meta-llama/Meta-Llama-3-8B`                   |
| independent | `mistralai/Mistral-7B-v0.3`                    |
| independent | `Qwen/Qwen2.5-7B`                              |
| independent | `01-ai/Yi-6B`                                  |
| independent | `tiiuae/falcon-7b`                             |
| independent | `EleutherAI/llemma_7b`                         |
| independent | `microsoft/Phi-3-small-8k-instruct`            |
| independent | `THUDM/chatglm3-6b-base`                       |

Some gated models require accepting the corresponding Hugging Face license before download.

## Run

Clone the repository and run the commands below from its root directory:

```bash
git clone https://github.com/lightrain-a/TCF-LLM-Fingerprinting.git
cd TCF-LLM-Fingerprinting
```

End-to-end:

```bash
SEED=20260418 bash run_tcf_qwen3_1p7b_instruct.sh all
```

Stage-by-stage:

```bash
SEED=20260418 bash run_tcf_qwen3_1p7b_instruct.sh generate
SEED=20260418 bash run_tcf_qwen3_1p7b_instruct.sh eval
SEED=20260418 bash run_tcf_qwen3_1p7b_instruct.sh select-step
SEED=20260418 bash run_tcf_qwen3_1p7b_instruct.sh auc
```

Important variables:

```bash
export CUDA_DEVICE=0
export N_WORKERS=4
export MAX_STEPS=200
export SEED=20260418
export MODEL_CSV=configs/qwen3_1p7b_instruct_runtime_models.csv
export SPARSE_STEPS=0,1,5,10,20,40,60,80,100,120,140,160,180,200
export SELECT_RULE=scm
```

For a machine with an existing local runtime CSV, override `MODEL_CSV` to point to that file. This is useful when local model paths differ from the anonymized `models/<org>/<repo>` convention.

## Outputs

Default output root:

```text
runs/qwen3_1p7b_instruct_c2_step200_seed20260418/
```

Key outputs:

- `generation/group_C2_targets.csv`: selected C2 targets.
- `generation/progress/group_C2_worker_*_manifest.csv`: GCG suffix manifests.
- `aligned_sparse_eval/sparse_eval_detail.csv`: per-model/per-step target-hit details.
- `aligned_sparse_eval/sparse_eval_task_table.csv`: suffix table.
- `source_scm/source_scm_by_sample_step.csv`: per-sample/per-step source SCM.
- `source_scm/source_scm_by_sample_step_summary.csv`: mean/median SCM by step.
- `step_selection/selected_step.csv`: selected step and rule.
- `auc/*_basic_final_auc.csv`: aggregate AUC.
- `auc/*_basic_final_scores.csv`: model-level scores.

## Reproduced Result

We reproduced the Qwen3 Instruct TCF evaluator with fixed `SEED=20260418` on a single A100 80GB GPU. The reproduced aggregate files are:

- `repro_results/qwen3_1p7b_instruct_seed20260418/selected_step.csv`
- `repro_results/qwen3_1p7b_instruct_seed20260418/auc.csv`
- `repro_results/qwen3_1p7b_instruct_seed20260418/step_selection_summary.csv`
- `repro_results/qwen3_1p7b_instruct_seed20260418/source_scm_summary.csv`

The selected step was `197` with `selection_rule=scm`. At this step, mean SCM is `6.9020` and median SCM is `6.5100` over 20 source-side fingerprints.

| method   | selected step | positives | negatives | positive mean | negative mean |    AUC |
| -------- | ------------: | --------: | --------: | ------------: | ------------: | -----: |
| TCF/Ours |           197 |         7 |         9 |        0.9214 |        0.1389 | 1.0000 |

The reproduced run focuses on TCF/Ours and does not recompute TRAP, ProFLingo, or ZeroPrint by default. TRAP and ProFLingo rows are left as missing in `auc.csv` unless the corresponding baseline CSVs are supplied. ZeroPrint is not recomputed by this wrapper.

Optional TRAP and ProFLingo CSVs can be supplied with:

```bash
export PROFLINGO_CSV=path/to/proflingo_results.csv
export TRAP_CSV=path/to/trap_results.csv
```

## Standalone SCM Selection

To recompute SCM directly from a task table:

```bash
python scripts/compute_source_scm.py \
  --task-table-csv path/to/sparse_eval_task_table.csv \
  --model-csv configs/qwen3_1p7b_instruct_runtime_models.csv \
  --official-demo-csv path/to/base_filtered_pool.csv \
  --categories Chemistry_val,Health_dev,Physics_val,Engineering_dev \
  --source-model-name Qwen3-1.7B \
  --trigger-placement question_prefix \
  --continuation-style space \
  --official-prompt-wrapper plain \
  --out-csv path/to/source_scm_by_sample_step.csv
```

Then select a step by SCM:

```bash
python scripts/select_scm_step.py \
  --detail-csv path/to/sparse_eval_detail.csv \
  --task-table-csv path/to/sparse_eval_task_table.csv \
  --scm-csv path/to/source_scm_by_sample_step.csv \
  --rule scm \
  --out-dir path/to/step_selection
```


## Public Release

This repository is the public release of the TCF reproduction artifact. Model checkpoints are not redistributed; use the public model identifiers listed above or compatible local paths.