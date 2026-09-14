"""Slice metrics from per-text predictions (outputs/preds/<run>__<test>.parquet).

For a given test set: AUROC/accuracy overall, per generation task, per generator, and on a
length-matched slice (both classes restricted to the same word-count band) to check that a
detector is not exploiting text length.

Usage:
    python scripts/analyze_slices.py --test gen_tpro2_32b_test
    python scripts/analyze_slices.py --test coat_test --runs ruroberta,binoculars_qwen3_14b_bf16
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PREDS = Path(__file__).resolve().parents[1] / "outputs" / "preds"


def auroc(d: pd.DataFrame) -> float:
    if "score" not in d or d.label.nunique() < 2:
        return float("nan")
    return roc_auc_score(d.label, d.score)


def slice_table(d: pd.DataFrame, by: str) -> pd.DataFrame:
    """Metrics of each machine group against ALL human texts of the set."""
    human = d[d.label == 0]
    rows = []
    for g, m in d[d.label == 1].groupby(by):
        both = pd.concat([human, m])
        rows.append({by: g, "n_machine": len(m), "auroc": auroc(both), "recall": (m.pred == 1).mean()})
    return pd.DataFrame(rows).sort_values("auroc")


def length_matched(d: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return d[(d.words >= lo) & (d.words <= hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--runs", default="", help="comma-separated run names; default: all runs with predictions on this test")
    ap.add_argument("--band", default="20,60", help="word-count band for the length-matched slice")
    args = ap.parse_args()
    lo, hi = map(int, args.band.split(","))

    files = sorted(PREDS.glob(f"*__{args.test}.parquet"))
    if args.runs:
        files = [f for f in files if f.name.split("__")[0] in args.runs.split(",")]
    for f in files:
        run = f.name.split("__")[0]
        d = pd.read_parquet(f)
        lm = length_matched(d, lo, hi)
        print(f"\n===== {run} on {args.test}: n={len(d)}  AUROC={auroc(d):.3f}  acc={accuracy_score(d.label, d.pred):.3f}")
        print(f"  human words median {d[d.label==0].words.median():.0f} | machine {d[d.label==1].words.median():.0f}")
        print(f"  length-matched [{lo},{hi}] words: n={len(lm)} (human {int((lm.label==0).sum())}, machine {int((lm.label==1).sum())})"
              f"  AUROC={auroc(lm):.3f}  acc={accuracy_score(lm.label, lm.pred):.3f}")
        if "task" in d and d.task.nunique() > 1:
            print(slice_table(d, "task").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if "generator" in d and d[d.label == 1].generator.nunique() > 1:
            print(slice_table(d, "generator").head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
