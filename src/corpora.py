"""Unified access to all corpora for mixed-corpus training and leave-one-corpus-out evaluation.

Every frame has columns: id, text, label, generator, domain, split, corpus.
Train splits of external corpora are balanced 50/50 by downsampling the majority class,
so that mixing does not change the class prior seen by the model.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import load_coat
from src.external_data import load_ainl, load_llmtrace_ru

CORPORA = ("coat", "llmtrace", "ainl")
SEED = 42


GEN_DIR = Path(__file__).resolve().parents[1] / "data" / "gen"


def gen_corpora() -> list[str]:
    """Names of generated held-out test sets present on disk, e.g. gen_qwen38_27b."""
    return sorted(f"gen_{p.name[: -len('_testset.parquet')]}" for p in GEN_DIR.glob("*_testset.parquet"))


def load_corpus(name: str) -> pd.DataFrame:
    if name.startswith("gen_"):
        return pd.read_parquet(GEN_DIR / f"{name[4:]}_testset.parquet")
    if name == "coat":
        df = load_coat().copy()
        df["domain"] = "coat"
        df["corpus"] = "coat"
        return df
    if name == "llmtrace":
        return load_llmtrace_ru()
    if name == "ainl":
        df = load_ainl().copy()
        # AINL ships no dev split: carve a fixed 5% of train (stratified by generator) for model selection
        train = df[df.split == "train"]
        dev_idx = pd.concat([g.sample(frac=0.05, random_state=SEED) for _, g in train.groupby("generator")]).index
        df.loc[dev_idx, "split"] = "dev"
        return df
    raise ValueError(name)


def balance(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    n = df.label.value_counts().min()
    parts = [g.sample(n, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def train_mix(corpora: list[str], per_corpus_cap: int | None = None) -> pd.DataFrame:
    """Balanced train split of each corpus, concatenated. `per_corpus_cap` limits rows per corpus."""
    parts = []
    for c in corpora:
        df = load_corpus(c)
        tr = balance(df[df.split == "train"])  # already shuffled with SEED
        if per_corpus_cap:
            tr = tr.head(per_corpus_cap)  # same rows as scripts/score_binoculars.py --cap
        parts.append(tr)
    return pd.concat(parts, ignore_index=True)


def dev_mix(corpora: list[str], cap: int = 3000) -> pd.DataFrame:
    """Balanced dev set drawn from the same corpora as training, capped per corpus."""
    parts = [balance(load_corpus(c).query("split == 'dev'")).head(cap) for c in corpora]
    return pd.concat(parts, ignore_index=True)


def test_sets(include_gen: bool = True) -> dict[str, pd.DataFrame]:
    names = list(CORPORA) + (gen_corpora() if include_gen else [])
    return {f"{c}_test": load_corpus(c).query("split == 'test'").reset_index(drop=True) for c in names}
