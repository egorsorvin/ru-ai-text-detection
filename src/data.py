"""Data loading for CoAT (Corpus of Artificial Texts, RussianNLP/coat).

Protocol
--------
The official CoAT test split has no public labels (they are held on Kaggle),
so we use:
  * train  -> 90% of official train (stratified by generator)
  * dev    -> 10% of official train (model selection / early stopping)
  * test   -> official *validation* split (24,628 texts), used once for reporting

Both HF configs (`binary`, `authorship`) share ids and texts, so we merge them
into a single frame with columns: id, text, label (0 human / 1 machine), generator.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE = DATA_DIR / "coat.parquet"
SEED = 42


def _download() -> pd.DataFrame:
    from datasets import load_dataset

    binary = load_dataset("RussianNLP/coat", "binary")
    author = load_dataset("RussianNLP/coat", "authorship")
    frames = []
    for split in ["train", "validation"]:  # official test is unlabeled
        b = binary[split].to_pandas()
        a = author[split].to_pandas()
        assert (b["id"].values == a["id"].values).all()
        b["generator"] = a["label"].values
        b["official_split"] = split
        frames.append(b)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].astype(int)
    return df


def load_coat(force: bool = False) -> pd.DataFrame:
    """Return a frame with columns id, text, label, generator, split (train/dev/test)."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    df = _download()
    df["split"] = "test"
    train_mask = df["official_split"] == "train"
    train = df[train_mask]
    # stratified 10% dev by generator (keeps rare generators in both parts)
    dev_idx = (
        train.groupby("generator", group_keys=False)
        .apply(lambda g: g.sample(frac=0.10, random_state=SEED))
        .index
    )
    df.loc[train_mask, "split"] = "train"
    df.loc[dev_idx, "split"] = "dev"
    df = df.drop(columns=["official_split"])
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def get_splits(df: pd.DataFrame | None = None):
    df = load_coat() if df is None else df
    return (df[df.split == s].reset_index(drop=True) for s in ("train", "dev", "test"))


if __name__ == "__main__":
    df = load_coat(force=True)
    print(df.groupby("split").agg(n=("id", "size"), machine_rate=("label", "mean")))
    print(df[df.split == "dev"]["generator"].value_counts())
