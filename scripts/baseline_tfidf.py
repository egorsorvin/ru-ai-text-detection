"""Baseline 1: TF-IDF (word + char n-grams) + logistic regression.

RuATD 2022 reported 0.642 accuracy for a TF-IDF baseline on the binary task.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import get_splits
from src.evaluate import report


def main():
    train, dev, test = get_splits()
    t0 = time.time()
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=300_000, sublinear_tf=True)
    Xtr = hstack([word_vec.fit_transform(train.text), char_vec.fit_transform(train.text)]).tocsr()
    print(f"features: {Xtr.shape[1]:,}  fit-vectorizers {time.time()-t0:.0f}s")

    clf = LogisticRegression(C=2.0, max_iter=2000, solver="liblinear")
    clf.fit(Xtr, train.label)
    print(f"fit-lr {time.time()-t0:.0f}s")

    for name, df in [("dev", dev), ("test", test)]:
        X = hstack([word_vec.transform(df.text), char_vec.transform(df.text)]).tocsr()
        score = clf.predict_proba(X)[:, 1]
        report("tfidf_lr", name, df, (score >= 0.5).astype(int), score)


if __name__ == "__main__":
    main()
