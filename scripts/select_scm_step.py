#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LETTERS = ("A", "B", "C", "D")


def pairwise_auc(pos: list[float], neg: list[float]) -> float:
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            if pd.isna(p) or pd.isna(n):
                continue
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total if total else float("nan")


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def normalize_step(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "step" not in df.columns:
        raise SystemExit(f"{path} must contain a step column")
    out = df.copy()
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out = out[out["step"].notna()].copy()
    out["step"] = out["step"].astype(int)
    return out


def add_scm_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    for col in ["SCM", "scm", "mean_SCM", "mean_scm"]:
        if col in out.columns:
            out["__scm"] = pd.to_numeric(out[col], errors="coerce")
            return out, f"direct:{col}"

    if {"clean_side_margin", "prefixed_side_margin"}.issubset(out.columns):
        clean = pd.to_numeric(out["clean_side_margin"], errors="coerce")
        prefixed = pd.to_numeric(out["prefixed_side_margin"], errors="coerce")
        out["__scm"] = np.minimum(clean, prefixed)
        return out, "derived:min(clean_side_margin,prefixed_side_margin)"

    if {"clean_margin", "prefixed_margin"}.issubset(out.columns):
        clean = pd.to_numeric(out["clean_margin"], errors="coerce")
        prefixed = pd.to_numeric(out["prefixed_margin"], errors="coerce")
        out["__scm"] = np.minimum(clean, prefixed)
        return out, "derived:min(clean_margin,prefixed_margin)"

    out["__scm"] = np.nan
    return out, "unavailable"


def summarize_detail(detail: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    df = detail.copy()
    if categories and "category" in df.columns:
        df = df[df["category"].astype(str).isin(categories)].copy()
    if "hit_target" not in df.columns:
        raise SystemExit("detail CSV must contain hit_target for proxy/AUC summaries")
    if "model_group" not in df.columns:
        raise SystemExit("detail CSV must contain model_group")
    if "model_name" not in df.columns:
        raise SystemExit("detail CSV must contain model_name")

    df["hit_target"] = pd.to_numeric(df["hit_target"], errors="coerce")
    rows = []
    for step, g in df.groupby("step", sort=True):
        source = g[g["model_group"].astype(str).eq("source")]
        family = g[g["model_group"].astype(str).isin(["source", "family"])]
        independent = g[g["model_group"].astype(str).eq("independent")]

        model_scores = (
            g.groupby(["model_name", "model_group"], as_index=False)["hit_target"]
            .mean()
            .rename(columns={"hit_target": "model_score"})
        )
        pos = model_scores[model_scores["model_group"].astype(str).isin(["source", "family"])]["model_score"].dropna().tolist()
        neg = model_scores[model_scores["model_group"].astype(str).eq("independent")]["model_score"].dropna().tolist()

        rows.append(
            {
                "step": int(step),
                "source_target_hit": float(source["hit_target"].mean()) if not source.empty else np.nan,
                "positive_target_hit": float(family["hit_target"].mean()) if not family.empty else np.nan,
                "independent_target_hit": float(independent["hit_target"].mean()) if not independent.empty else np.nan,
                "family_independent_gap": (
                    float(family["hit_target"].mean() - independent["hit_target"].mean())
                    if not family.empty and not independent.empty
                    else np.nan
                ),
                "hardlabel_auc": pairwise_auc(pos, neg),
                "n_source_rows": int(len(source)),
                "n_positive_models": int(len(pos)),
                "n_independent_models": int(len(neg)),
            }
        )
    return pd.DataFrame(rows)


def summarize_scm(source: pd.DataFrame, categories: list[str]) -> tuple[pd.DataFrame, str]:
    df = source.copy()
    if categories and "category" in df.columns:
        df = df[df["category"].astype(str).isin(categories)].copy()
    df, mode = add_scm_column(df)
    if df["__scm"].notna().any():
        return (
            df.groupby("step", as_index=False)
            .agg(mean_scm=("__scm", "mean"), median_scm=("__scm", "median"), n_scm_rows=("__scm", "count")),
            mode,
        )
    return pd.DataFrame({"step": sorted(df["step"].dropna().astype(int).unique())}), mode


def choose_step(summary: pd.DataFrame, rule: str) -> pd.Series:
    df = summary.copy()
    if rule == "auto":
        rule = "scm" if "mean_scm" in df.columns and df["mean_scm"].notna().any() else "proxy"
    df["selection_rule"] = rule
    if rule == "scm":
        if "mean_scm" not in df.columns or not df["mean_scm"].notna().any():
            raise SystemExit("SCM rule requested, but no SCM column or derivable SCM margins were found")
        ordered = df.sort_values(
            ["mean_scm", "source_target_hit", "family_independent_gap", "hardlabel_auc", "step"],
            ascending=[False, False, False, False, True],
        )
    elif rule == "proxy":
        df["proxy_score"] = (
            df["source_target_hit"].fillna(0.0)
            + 0.5 * df["positive_target_hit"].fillna(0.0)
            - 0.5 * df["independent_target_hit"].fillna(0.0)
        )
        ordered = df.sort_values(
            ["proxy_score", "source_target_hit", "family_independent_gap", "hardlabel_auc", "step"],
            ascending=[False, False, False, False, True],
        )
    elif rule == "auc":
        ordered = df.sort_values(
            ["hardlabel_auc", "family_independent_gap", "source_target_hit", "step"],
            ascending=[False, False, False, True],
        )
    else:
        raise SystemExit(f"Unsupported rule={rule}")
    return ordered.iloc[0]


def write_report(path: Path, selected: pd.Series, scm_mode: str, detail_path: Path, task_path: Path | None) -> None:
    lines = [
        "# TCF Step Selection",
        "",
        f"- detail_csv: `{detail_path}`",
        f"- task_table_csv: `{task_path}`" if task_path else "- task_table_csv: not provided",
        f"- selected_step: `{int(selected['step'])}`",
        f"- selection_rule: `{selected['selection_rule']}`",
        f"- scm_mode: `{scm_mode}`",
        "",
        "If SCM is unavailable in the input CSV, `auto` uses a hard-label proxy:",
        "`source_target_hit + 0.5 * positive_target_hit - 0.5 * independent_target_hit`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a TCF fingerprint step by SCM, or by a documented proxy when SCM is absent.")
    parser.add_argument("--detail-csv", required=True, type=Path)
    parser.add_argument("--task-table-csv", type=Path)
    parser.add_argument("--scm-csv", type=Path, help="Optional CSV with per-step/per-sample SCM columns.")
    parser.add_argument("--categories", default="", help="Comma-separated category filter.")
    parser.add_argument("--rule", choices=["auto", "scm", "proxy", "auc"], default="auto")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    categories = parse_list(args.categories)
    detail = normalize_step(pd.read_csv(args.detail_csv), args.detail_csv)
    detail_summary = summarize_detail(detail, categories)

    scm_source_path = args.scm_csv or args.task_table_csv or args.detail_csv
    scm_df = normalize_step(pd.read_csv(scm_source_path), scm_source_path)
    scm_summary, scm_mode = summarize_scm(scm_df, categories)
    summary = detail_summary.merge(scm_summary, on="step", how="left")
    selected = choose_step(summary, args.rule)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "step_selection_summary.csv"
    selected_path = args.out_dir / "selected_step.csv"
    report_path = args.out_dir / "step_selection_report.md"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([selected.to_dict()]).to_csv(selected_path, index=False)

    if args.task_table_csv:
        task = normalize_step(pd.read_csv(args.task_table_csv), args.task_table_csv)
        if categories and "category" in task.columns:
            task = task[task["category"].astype(str).isin(categories)].copy()
        task = task[task["step"].eq(int(selected["step"]))].copy()
        task.to_csv(args.out_dir / "selected_step_task_table.csv", index=False)

    write_report(report_path, selected, scm_mode, args.detail_csv, args.task_table_csv)
    print(f"[OK] selected_step={int(selected['step'])} rule={selected['selection_rule']} scm_mode={scm_mode}")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {selected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
