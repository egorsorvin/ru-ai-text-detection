"""Report Binoculars zero-shot metrics on every test set from cached features (no GPU).

Threshold is fitted on cached CoAT dev scores and applied unchanged to every test set.

Usage:
    python scripts/eval_binoculars_cache.py --tag qwen3_14b --quant bf16
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.score_binoculars import cache_path, load_features
from src.corpora import load_corpus, test_sets
from src.evaluate import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--quant", default="bf16")
    args = ap.parse_args()
    run = f"binoculars_{args.tag}_{args.quant}"

    dev = load_corpus("coat").query("split == 'dev'").merge(load_features(args.tag, args.quant, "coat", "dev"), on="id")
    cands = np.quantile(dev.score, np.linspace(0.01, 0.99, 197))
    accs = [((dev.score < t).astype(int) == dev.label).mean() for t in cands]
    thr = float(cands[int(np.argmax(accs))])
    print(f"threshold from coat dev ({len(dev)} texts): {thr:.4f}, dev acc {max(accs):.4f}")

    for name, df in test_sets().items():
        corpus = name[: -len("_test")]
        if not cache_path(args.tag, args.quant, corpus, "test").exists():
            print(f"{name}: no cached features, skipped")
            continue
        m = df.merge(load_features(args.tag, args.quant, corpus, "test"), on="id")
        report(run, name, m, (m.score < thr).astype(int), -m.score.values)


if __name__ == "__main__":
    main()
