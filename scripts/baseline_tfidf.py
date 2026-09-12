"""Baseline 1: TF-IDF (word + char n-grams) + logistic regression.

RuATD 2022 reported 0.642 accuracy for a TF-IDF baseline on the binary task.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import get_splits
from src.evaluate import report

ROOT = Path(__file__).resolve().parents[1]


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=300_000, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features), ("lr", LogisticRegression(C=2.0, max_iter=2000, solver="liblinear"))])


def main():
    train, dev, test = get_splits()
    t0 = time.time()
    pipe = build_pipeline()
    pipe.fit(train.text, train.label)
    print(f"fit {time.time()-t0:.0f}s")
    out = ROOT / "outputs" / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "tfidf_lr.joblib")

    for name, df in [("dev", dev), ("test", test)]:
        score = pipe.predict_proba(df.text)[:, 1]
        report("tfidf_lr", name, df, (score >= 0.5).astype(int), score)


if __name__ == "__main__":
    main()
