#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def pairwise_auc(pos: Iterable[float], neg: Iterable[float]) -> float:
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


def normalize_models(model_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(model_csv).copy()
    if "model_name" not in df.columns or "model_path" not in df.columns:
        raise SystemExit(f"{model_csv} must contain model_name and model_path")
    if "order" not in df.columns:
        df["order"] = range(1, len(df) + 1)
    if "model_group" in df.columns:
        group_raw = df["model_group"].astype(str)
        df["basic_group"] = group_raw
    elif "group" in df.columns:
        group_raw = df["group"].astype(str)
        df["basic_group"] = np.where(group_raw.eq("Non-homologous"), "independent", "family")
        source_mask = group_raw.eq("Main") | df["model_name"].astype(str).str.contains("Meta-Llama-3-8B", regex=False)
        df.loc[source_mask, "basic_group"] = "source"
    else:
        raise SystemExit(f"{model_csv} must contain model_group or group")

    df["basic_group"] = df["basic_group"].replace(
        {
            "Main": "source",
            "Non-homologous": "independent",
            "SFT/Task": "family",
            "RLHF": "family",
            "Pruning": "family",
        }
    )
    keep = [
        "order",
        "model_name",
        "model_path",
        "basic_group",
        "model_group",
        "group",
        "template",
        "testacc",
        "testacc_common",
        "testacc_source",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sort_values("order").copy()


def load_proflingo(path: Path, models: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = models[["model_name", "model_path"]].copy()
    out[score_col] = np.nan
    out[f"{score_col}_status"] = "missing"
    if not path or not path.is_file():
        return out[["model_name", score_col, f"{score_col}_status"]]
    df = pd.read_csv(path).copy()
    if df.empty:
        return out[["model_name", score_col, f"{score_col}_status"]]

    if "model_path" in df.columns:
        df["model_path"] = df["model_path"].astype(str)
        raw_score_col = "asr" if "asr" in df.columns else "proflingo_regen" if "proflingo_regen" in df.columns else "proflingo"
        cols = ["model_path", raw_score_col]
        if "status" in df.columns:
            cols.append("status")
        got = df[cols].rename(columns={raw_score_col: score_col, "status": f"{score_col}_status"})
        merged = out.drop(columns=[score_col, f"{score_col}_status"]).merge(got, on="model_path", how="left")
    elif "model_name" in df.columns:
        raw_score_col = "proflingo_regen" if "proflingo_regen" in df.columns else "proflingo" if "proflingo" in df.columns else "asr"
        cols = ["model_name", raw_score_col]
        if "status" in df.columns:
            cols.append("status")
        got = df[cols].rename(columns={raw_score_col: score_col, "status": f"{score_col}_status"})
        merged = out.drop(columns=[score_col, f"{score_col}_status"]).merge(got, on="model_name", how="left")
    else:
        raise SystemExit(f"Cannot join ProFLingo file without model_path/model_name: {path}")

    merged[score_col] = pd.to_numeric(merged[score_col], errors="coerce")
    merged[f"{score_col}_status"] = merged[f"{score_col}_status"].fillna("missing")
    return merged[["model_name", score_col, f"{score_col}_status"]]


def load_trap(path: Path, models: pd.DataFrame, score_col: str, step: int | None) -> pd.DataFrame:
    out = models[["model_name"]].copy()
    out[score_col] = np.nan
    out[f"{score_col}_status"] = "missing"
    if not path or not path.is_file():
        return out
    df = pd.read_csv(path).copy()
    if df.empty:
        return out

    if "step" in df.columns and step is not None:
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df = df[df["step"].eq(int(step))].copy()
    if "model_name" not in df.columns and "model" in df.columns:
        df["model_name"] = df["model"].astype(str)
    if "model_name" not in df.columns:
        raise SystemExit(f"Cannot join TRAP file without model_name/model: {path}")
    raw_score_col = "score" if "score" in df.columns else "retrieval_rate"
    if raw_score_col not in df.columns:
        raise SystemExit(f"Cannot find TRAP score/retrieval_rate in {path}")
    df[score_col] = pd.to_numeric(df[raw_score_col], errors="coerce")
    if "status" not in df.columns:
        df["status"] = np.where(df[score_col].notna(), "ok", "missing")
    got = (
        df[["model_name", score_col, "status"]]
        .drop_duplicates("model_name", keep="last")
        .rename(columns={"status": f"{score_col}_status"})
    )
    merged = out.drop(columns=[score_col, f"{score_col}_status"]).merge(got, on="model_name", how="left")
    merged[f"{score_col}_status"] = merged[f"{score_col}_status"].fillna("missing")
    return merged


def load_ours(path: Path, models: pd.DataFrame, score_col: str, step: int) -> pd.DataFrame:
    out = models[["model_name"]].copy()
    out[score_col] = np.nan
    out[f"{score_col}_status"] = "missing"
    if not path or not path.is_file():
        return out
    df = pd.read_csv(path).copy()
    if df.empty:
        return out
    if "model_name" not in df.columns:
        raise SystemExit(f"Cannot join Ours file without model_name: {path}")
    if "step" not in df.columns:
        raise SystemExit(f"Ours file must contain step: {path}")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df[df["step"].eq(int(step))].copy()
    if df.empty:
        return out
    if "score" in df.columns:
        got = df[["model_name", "score"]].rename(columns={"score": score_col})
    elif "hit_target" in df.columns:
        got = (
            df.assign(hit_target=pd.to_numeric(df["hit_target"], errors="coerce"))
            .groupby("model_name", as_index=False)["hit_target"]
            .mean()
            .rename(columns={"hit_target": score_col})
        )
    else:
        raise SystemExit(f"Cannot find Ours score/hit_target in {path}")
    got[score_col] = pd.to_numeric(got[score_col], errors="coerce")
    got[f"{score_col}_status"] = np.where(got[score_col].notna(), "ok", "missing")
    merged = out.drop(columns=[score_col, f"{score_col}_status"]).merge(got, on="model_name", how="left")
    merged[f"{score_col}_status"] = merged[f"{score_col}_status"].fillna("missing")
    return merged


def build_auc(scores: pd.DataFrame, method_cols: list[str]) -> pd.DataFrame:
    rows = []
    for method in method_cols:
        pos_df = scores[scores["basic_group"].isin(["source", "family"]) & scores[method].notna()].copy()
        neg_df = scores[scores["basic_group"].eq("independent") & scores[method].notna()].copy()
        pos = pd.to_numeric(pos_df[method], errors="coerce").dropna().tolist()
        neg = pd.to_numeric(neg_df[method], errors="coerce").dropna().tolist()
        rows.append(
            {
                "method": method,
                "positive_group": "source+family",
                "negative_group": "independent",
                "n_positive": len(pos),
                "n_negative": len(neg),
                "n_pairs": len(pos) * len(neg),
                "positive_mean": float(np.mean(pos)) if pos else np.nan,
                "negative_mean": float(np.mean(neg)) if neg else np.nan,
                "auc": pairwise_auc(pos, neg),
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.fillna("")
    cols = list(text.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final score/AUC table for one model family.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-csv", required=True, type=Path)
    parser.add_argument("--proflingo-csv", type=Path)
    parser.add_argument("--trap-csv", type=Path)
    parser.add_argument("--trap-step", type=int)
    parser.add_argument("--ours-csv", type=Path)
    parser.add_argument("--ours-step", type=int, default=200)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    models = normalize_models(args.model_csv)
    scores = models.copy()
    method_cols = []

    pro_col = "proflingo_final"
    trap_col = "trap_final"
    ours_col = f"ours_step{args.ours_step}"
    scores = scores.merge(load_proflingo(args.proflingo_csv, models, pro_col), on="model_name", how="left")
    scores = scores.merge(load_trap(args.trap_csv, models, trap_col, args.trap_step), on="model_name", how="left")
    scores = scores.merge(load_ours(args.ours_csv, models, ours_col, args.ours_step), on="model_name", how="left")
    method_cols.extend([pro_col, trap_col, ours_col])

    auc = build_auc(scores, method_cols)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.out_dir / f"{args.model_key}_basic_final_scores.csv"
    auc_path = args.out_dir / f"{args.model_key}_basic_final_auc.csv"
    report_path = args.out_dir / f"{args.model_key}_basic_final_report.md"
    scores.to_csv(score_path, index=False)
    auc.to_csv(auc_path, index=False)

    score_cols = [
        "basic_group",
        "model_name",
        pro_col,
        f"{pro_col}_status",
        trap_col,
        f"{trap_col}_status",
        ours_col,
        f"{ours_col}_status",
    ]
    score_cols = [c for c in score_cols if c in scores.columns]
    lines = [
        f"# {args.model_key} Basic Final Results",
        "",
        f"- scores: `{score_path}`",
        f"- auc: `{auc_path}`",
        f"- AUC rule: source+family positives vs independent negatives.",
        f"- Ours step: `{args.ours_step}`",
        f"- TRAP step: `{args.trap_step if args.trap_step is not None else 'last/raw'}`",
        "",
        "## AUC",
        "",
        md_table(auc),
        "",
        "## Scores",
        "",
        md_table(scores[score_cols]),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {score_path}")
    print(f"[OK] wrote {auc_path}")
    print(f"[OK] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
