"""Shared evaluation utilities: metrics, per-generator breakdown, result persistence."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "results"


def metrics(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_score is not None and len(set(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    return out


def per_generator(df: pd.DataFrame, y_pred) -> pd.DataFrame:
    """Accuracy per generator (Human row = specificity, others = recall)."""
    d = df[["generator", "label"]].copy()
    d["correct"] = (d["label"].values == np.asarray(y_pred)).astype(float)
    g = d.groupby("generator")["correct"].agg(["mean", "size"]).rename(columns={"mean": "acc", "size": "n"})
    return g.sort_values("n", ascending=False)


def save_result(name: str, split: str, m: dict, extra: dict | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": name, "split": split, "time": datetime.now().isoformat(timespec="seconds"), **m}
    if extra:
        payload.update(extra)
    path = RESULTS_DIR / f"{name}__{split}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def report(name: str, split: str, df: pd.DataFrame, y_pred, y_score=None, save=True) -> dict:
    m = metrics(df["label"].values, y_pred, y_score)
    print(f"[{name}] {split}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
    pg = per_generator(df, y_pred)
    print(pg.to_string(float_format=lambda x: f"{x:.3f}"))
    if save:
        save_result(name, split, m, {"per_generator": pg["acc"].round(4).to_dict()})
    return m
