"""Binoculars decision thresholds that do not peek at the test corpus.

For every Binoculars pair with cached features and every test set:
  auroc          threshold-free
  bacc_coatdev   threshold fitted on CoAT dev (the original protocol)
  bacc_otherdev  threshold fitted on the dev sets of the *other* corpora, each corpus weighted equally;
                 for generated sets all three corpora count as other (honest zero-shot calibration)
  bacc_owndev    threshold fitted on the dev set of the same corpus (optimistic upper bound; n/a for generated sets)
  tpr@fpr5       share of machine texts caught when 5% of human texts are flagged, on the test set itself
Balanced accuracy is used because LLMTrace and AINL test sets are not 50/50. Lower Binoculars score = machine.

Usage:
    python scripts/analyze_thresholds.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.corpora import CORPORA, load_corpus, test_sets

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "outputs" / "scores"
OUT = ROOT / "outputs" / "analysis"


def features(pair: str, corpus: str, split: str) -> pd.DataFrame | None:
    p = SCORES / f"{pair}_bf16" / f"{corpus}__{split}.parquet"
    return pd.read_parquet(p)[["id", "score"]] if p.exists() else None


def bacc(d: pd.DataFrame, thr: float) -> float:
    if np.isnan(thr):
        return np.nan
    flagged = d.score < thr
    return float((flagged[d.label == 1].mean() + (~flagged)[d.label == 0].mean()) / 2)


def fit_threshold(devs: list[pd.DataFrame]) -> float:
    if not devs:
        return np.nan
    pooled = pd.concat(devs).score
    cands = np.quantile(pooled, np.linspace(0.005, 0.995, 399))
    means = [np.mean([bacc(d, t) for d in devs]) for t in cands]
    return float(cands[int(np.argmax(means))])


def tpr_at_fpr(d: pd.DataFrame, fpr: float) -> float:
    thr = np.quantile(d.score[d.label == 0], fpr)
    return float((d.score[d.label == 1] < thr).mean())


def main():
    pairs = sorted(p.name[: -len("_bf16")] for p in SCORES.glob("*_bf16") if p.is_dir())
    tests = test_sets()
    rows = []
    for pair in pairs:
        devs = {}
        for c in CORPORA:
            f = features(pair, c, "dev")
            if f is not None:
                devs[c] = load_corpus(c).query("split == 'dev'")[["id", "label"]].merge(f, on="id")
        thr_coat = fit_threshold([devs["coat"]]) if "coat" in devs else np.nan
        for name, df in tests.items():
            corpus = name[: -len("_test")]
            f = features(pair, corpus, "test")
            if f is None:
                continue
            d = df[["id", "label"]].merge(f, on="id")
            other = [v for k, v in devs.items() if k != corpus]
            rows.append({
                "pair": pair, "test": name, "n": len(d),
                "auroc": roc_auc_score(d.label, -d.score),
                "bacc_coatdev": bacc(d, thr_coat),
                "bacc_otherdev": bacc(d, fit_threshold(other)),
                "bacc_owndev": bacc(d, fit_threshold([devs[corpus]])) if corpus in devs else np.nan,
                "tpr@fpr5": tpr_at_fpr(d, 0.05),
            })
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "binoculars_thresholds.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.drop(columns=["n"]).to_string(index=False, float_format="%.2f"))
    print(f"\nsaved {OUT / 'binoculars_thresholds.csv'}")


if __name__ == "__main__":
    main()
