#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


LETTERS = ["A", "B", "C", "D"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_eval_helpers():
    mmlu_dir = repo_root() / "results" / "mmlu_cf"
    if str(mmlu_dir) not in sys.path:
        sys.path.insert(0, str(mmlu_dir))
    import eval_from_excel_sparse_steps as ev

    return ev


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def clipped_log_odds_from_log_scores(scores: dict[str, float], target: str, eps: float) -> tuple[float, float]:
    vals = np.array([float(scores[l]) for l in LETTERS], dtype=np.float64)
    m = float(np.max(vals))
    probs = np.exp(vals - m)
    probs = probs / float(np.sum(probs))
    q = float(probs[LETTERS.index(str(target).strip().upper())])
    q = min(max(q, float(eps)), 1.0 - float(eps))
    return float(math.log(q) - math.log1p(-q)), q


def source_row(model_csv: Path, source_model_name: str) -> pd.Series:
    df = pd.read_csv(model_csv)
    if source_model_name:
        src = df[df["model_name"].astype(str).eq(source_model_name)].copy()
    else:
        src = df[df["model_group"].astype(str).eq("source")].copy()
    if src.empty:
        raise SystemExit(f"Cannot find source row in {model_csv}")
    return src.iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute source-model counterfactual margin (SCM) for TCF task rows.")
    parser.add_argument("--task-table-csv", required=True, type=Path)
    parser.add_argument("--model-csv", required=True, type=Path)
    parser.add_argument("--official-demo-csv", required=True, type=Path)
    parser.add_argument("--source-model-name", default="", type=str)
    parser.add_argument("--qwen-base-path", default="", type=str)
    parser.add_argument("--categories", default="", type=str)
    parser.add_argument("--official-ntrain", default=5, type=int)
    parser.add_argument("--official-max-prompt-tokens", default=2048, type=int)
    parser.add_argument("--official-prompt-wrapper", choices=["plain", "tokenizer_chat_template"], default="plain")
    parser.add_argument("--continuation-style", choices=["space", "bare"], default="space")
    parser.add_argument("--trigger-placement", choices=["question_append", "prompt_end", "question_prefix"], default="question_prefix")
    parser.add_argument("--eps-clip", default=1e-6, type=float)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()

    ev = load_eval_helpers()
    src = source_row(args.model_csv, args.source_model_name)
    source_name = str(src["model_name"])
    source_path = str(src["model_path"])
    model_kind = str(src.get("model_kind", "") or "")
    qwen_base_path = str(args.qwen_base_path or source_path)

    task = pd.read_csv(args.task_table_csv).copy()
    task["step"] = pd.to_numeric(task["step"], errors="coerce").astype("Int64")
    task = task[task["step"].notna()].copy()
    task["step"] = task["step"].astype(int)
    categories = parse_list(args.categories)
    if categories and "category" in task.columns:
        task = task[task["category"].astype(str).isin(categories)].copy()
    task = task[task["target_letter"].astype(str).str.upper().isin(LETTERS)].copy()
    if task.empty:
        raise SystemExit("No task rows left after filtering")

    demo_df = pd.read_csv(args.official_demo_csv).copy()
    print(f"[SCM] source={source_name} path={source_path}")
    print(f"[SCM] rows={len(task)} samples={task['sample_id'].nunique()} steps={sorted(task['step'].unique().tolist())}")
    model, tokenizer, base_model = ev.load_model(source_path, model_kind, qwen_base_path)

    clean_log_odds: dict[str, float] = {}
    clean_q: dict[str, float] = {}
    rows = []
    try:
        # Compute clean side once per sample from the step-0/no-suffix prompt.
        for sample_id, g in task.sort_values(["sample_id", "step"]).groupby("sample_id", sort=False):
            clean_candidates = g[g["step"].eq(0)]
            row = clean_candidates.iloc[0] if not clean_candidates.empty else g.iloc[0].copy()
            row = row.copy()
            row["suffix_text"] = ""
            target = str(row["target_letter"]).strip().upper()
            _pred, scores, _prompt_mode = ev.predict_letter_by_logprob_official(
                model=model,
                tokenizer=tokenizer,
                row=row,
                suffix="",
                demo_df=demo_df,
                ntrain=int(args.official_ntrain),
                max_prompt_tokens=int(args.official_max_prompt_tokens),
                answer_letters=LETTERS,
                continuation_style=args.continuation_style,
                suffix_placement=args.trigger_placement,
                official_prompt_wrapper=args.official_prompt_wrapper,
                official_letter_only_instruction=False,
            )
            lo, q = clipped_log_odds_from_log_scores(scores, target, args.eps_clip)
            clean_log_odds[str(sample_id)] = lo
            clean_q[str(sample_id)] = q

        for _, row in task.sort_values(["sample_id", "step"]).iterrows():
            sample_id = str(row["sample_id"])
            suffix = "" if pd.isna(row.get("suffix_text", "")) else str(row.get("suffix_text", ""))
            target = str(row["target_letter"]).strip().upper()
            _pred, scores, prompt_mode = ev.predict_letter_by_logprob_official(
                model=model,
                tokenizer=tokenizer,
                row=row,
                suffix=suffix,
                demo_df=demo_df,
                ntrain=int(args.official_ntrain),
                max_prompt_tokens=int(args.official_max_prompt_tokens),
                answer_letters=LETTERS,
                continuation_style=args.continuation_style,
                suffix_placement=args.trigger_placement,
                official_prompt_wrapper=args.official_prompt_wrapper,
                official_letter_only_instruction=False,
            )
            prefixed_lo, prefixed_q = clipped_log_odds_from_log_scores(scores, target, args.eps_clip)
            clean_lo = float(clean_log_odds[sample_id])
            scm = min(prefixed_lo, -clean_lo)
            rec = {
                "group": row.get("group", ""),
                "sample_id": sample_id,
                "category": row.get("category", ""),
                "step": int(row["step"]),
                "target_letter": target,
                "clean_target_q": clean_q[sample_id],
                "prefixed_target_q": prefixed_q,
                "clean_target_log_odds": clean_lo,
                "prefixed_target_log_odds": prefixed_lo,
                "clean_side_margin": -clean_lo,
                "prefixed_side_margin": prefixed_lo,
                "SCM": scm,
                "prompt_mode": prompt_mode,
                "source_model_name": source_name,
            }
            rows.append(rec)
    finally:
        del model
        if base_model is not None:
            del base_model
        torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    summary = (
        out.groupby("step", as_index=False)
        .agg(mean_SCM=("SCM", "mean"), median_SCM=("SCM", "median"), n=("SCM", "count"))
        .sort_values("step")
    )
    summary.to_csv(args.out_csv.with_name(args.out_csv.stem + "_summary.csv"), index=False)
    print(f"[OK] wrote {args.out_csv}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
