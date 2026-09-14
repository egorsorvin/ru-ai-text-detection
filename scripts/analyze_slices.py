"""Slice metrics from per-text predictions (outputs/preds/<run>__<test>.parquet).

Summary (default): one row per run and test with AUROC overall, per generation task, per length band,
balanced accuracy of the model's own decisions, and the share of machine texts caught when 5% / 1%
of human texts are flagged. Saved to outputs/analysis/slices.csv.
Detail (--detail): per-generator table for each run.

Usage:
    python scripts/analyze_slices.py --tests gen_qwen38_27b_test,gen_tpro2_32b_test
    python scripts/analyze_slices.py --tests coat_test --detail
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "outputs" / "preds"
OUT = ROOT / "outputs" / "analysis"
TASKS = ["continue", "paraphrase", "simplify"]
BANDS = [(0, 20), (21, 40), (41, 10_000)]
MIN_CLASS = 30


def enough(d: pd.DataFrame) -> bool:
    return (d.label == 0).sum() >= MIN_CLASS and (d.label == 1).sum() >= MIN_CLASS


def auroc(d: pd.DataFrame) -> float:
    return roc_auc_score(d.label, d.score) if "score" in d and enough(d) else np.nan


def balanced_acc(d: pd.DataFrame) -> float:
    return float(((d.pred[d.label == 1] == 1).mean() + (d.pred[d.label == 0] == 0).mean()) / 2)


def tpr_at_fpr(d: pd.DataFrame, fpr: float) -> float:
    """Share of machine texts flagged when the threshold flags `fpr` of human texts (higher score = machine)."""
    if "score" not in d or not enough(d):
        return np.nan
    thr = np.quantile(d.score[d.label == 0], 1 - fpr)
    return float((d.score[d.label == 1] > thr).mean())


def summarize(d: pd.DataFrame) -> dict:
    human = d[d.label == 0]
    row = {"n": len(d), "auroc": auroc(d), "bacc": balanced_acc(d),
           "tpr@fpr5": tpr_at_fpr(d, 0.05), "tpr@fpr1": tpr_at_fpr(d, 0.01)}
    if "task" in d and d.task.isin(TASKS).any():
        for t in TASKS:
            row[f"auroc_{t}"] = auroc(pd.concat([human, d[(d.label == 1) & (d.task == t)]]))
    for lo, hi in BANDS:
        name = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        row[f"auroc_len{name}"] = auroc(d[(d.words >= lo) & (d.words <= hi)])
    row["words_human"] = float(human.words.median())
    row["words_machine"] = float(d[d.label == 1].words.median())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True, help="comma-separated test names, e.g. gen_tpro2_32b_test")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    rows = []
    for test in args.tests.split(","):
        for f in sorted(PREDS.glob(f"*__{test}.parquet")):
            run = f.name[: -len(f"__{test}.parquet")]
            if run.startswith("smoke"):
                continue
            d = pd.read_parquet(f)
            rows.append({"run": run, "test": test, **summarize(d)})
            if args.detail and "generator" in d and d[d.label == 1].generator.nunique() > 1:
                human = d[d.label == 0]
                g = [{"generator": k, "n": len(m), "auroc": auroc(pd.concat([human, m])), "recall": float((m.pred == 1).mean())}
                     for k, m in d[d.label == 1].groupby("generator")]
                print(f"\n== {run} on {test}\n" + pd.DataFrame(g).sort_values("auroc").to_string(index=False, float_format="%.3f"))

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "slices.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["n"]).to_string(index=False, float_format="%.2f"))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
