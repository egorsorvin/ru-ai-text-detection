"""External Russian corpora brought to the CoAT frame format.

Common columns: id, text, label (0 human / 1 machine), generator, domain, split, corpus.

LLMTrace (iitolstykh/LLMTrace_detection): Russian subset (lang == "ru"). Texts with
label "mixed" (partly human, partly AI) are dropped for the binary task.
AINL-Eval 2025 (iis-research-team/AINL-Eval-2025): scientific abstracts; the official
test has no public labels, so `dev_full.csv` (which contains generators unseen in train)
is used as test.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_llmtrace_ru(force: bool = False) -> pd.DataFrame:
    cache = DATA_DIR / "llmtrace_ru.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    from datasets import load_dataset

    ds = load_dataset("iitolstykh/LLMTrace_detection")
    frames = []
    for split, name in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        d = ds[split].to_pandas()
        d = d[(d.lang == "ru") & (d.label != "mixed")]
        frames.append(pd.DataFrame({
            "text": d.text.values,
            "label": (d.label == "ai").astype(int).values,
            "generator": d.model.where(d.label == "ai", "Human").values,
            "domain": d.data_type.values,
            "split": name,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "id", range(len(df)))
    df["corpus"] = "llmtrace"
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def _drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.text.map(lambda x: isinstance(x, str) and x.strip() != "")
    return df[ok].reset_index(drop=True)


def load_ainl(force: bool = False) -> pd.DataFrame:
    cache = DATA_DIR / "ainl.parquet"
    if cache.exists() and not force:
        return _drop_empty(pd.read_parquet(cache))  # AINL train contains one empty text
    from huggingface_hub import snapshot_download

    snap = snapshot_download("iis-research-team/AINL-Eval-2025", repo_type="dataset")
    frames = []
    human_tags = {"human", "abstract"}  # train uses "human", dev_full uses "abstract"
    for fname, name in [("train.csv", "train"), ("dev_full.csv", "test")]:
        a = pd.read_csv(Path(snap) / fname)
        is_human = a.label.isin(human_tags)
        frames.append(pd.DataFrame({
            "text": a.text.values,
            "label": (~is_human).astype(int).values,
            "generator": a.label.where(~is_human, "Human").values,
            "domain": "sci_abstract",
            "split": name,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "id", range(len(df)))
    df["corpus"] = "ainl"
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache, index=False)
    return _drop_empty(df)


if __name__ == "__main__":
    for name, fn in [("llmtrace", load_llmtrace_ru), ("ainl", load_ainl)]:
        df = fn(force=True)
        print(f"=== {name}")
        print(df.groupby("split").agg(n=("id", "size"), machine_rate=("label", "mean")))
        print(df[df.split == "test"].generator.value_counts().head(25))
