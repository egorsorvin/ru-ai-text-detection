"""Compute and cache Binoculars features (score, log-PPL, log-X-PPL) for any corpus split.

Cache: outputs/scores/<tag>_<quant>/<corpus>__<split>.parquet with columns id, score, ppl, xppl.
Re-running only scores ids that are not cached yet.

Usage:
    python scripts/score_binoculars.py --corpus coat --split train --cap 10000
    python scripts/score_binoculars.py --corpus llmtrace --split test
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.binoculars import load_model
from src.corpora import balance, load_corpus
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def binoculars_features(texts, tok, observer, performer, max_len, bs, device):
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])
    out = np.empty((len(texts), 3), dtype=np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        obs_logits = observer(**enc).logits[:, :-1]
        perf_logits = performer(**enc).logits[:, :-1]
        labels = enc.input_ids[:, 1:]
        lengths = enc.attention_mask[:, 1:].sum(1)
        for k in range(len(idx)):
            n = int(lengths[k])
            o = obs_logits[k, :n].float()
            p = perf_logits[k, :n].float()
            ppl = F.cross_entropy(p, labels[k, :n])
            x_ppl = -(F.softmax(o, dim=-1) * F.log_softmax(p, dim=-1)).sum(-1).mean()
            out[idx[k]] = ((ppl / x_ppl).item(), ppl.item(), x_ppl.item())
        del obs_logits, perf_logits
    return out


def cache_path(tag, quant, corpus, split):
    return ROOT / "outputs" / "scores" / f"{tag}_{quant}" / f"{corpus}__{split}.parquet"


def load_features(tag, quant, corpus, split) -> pd.DataFrame:
    p = cache_path(tag, quant, corpus, split)
    if not p.exists():
        raise FileNotFoundError(f"no cached Binoculars features: {p}. Run scripts/score_binoculars.py first.")
    return pd.read_parquet(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="coat | llmtrace | ainl | gen_<tag>")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--cap", type=int, default=0, help="score at most N (balanced) texts of this split")
    ap.add_argument("--observer", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--performer", default="Qwen/Qwen3-4B")
    ap.add_argument("--quant", choices=["nf4", "bf16"], default="nf4")
    ap.add_argument("--tag", default="qwen3_4b")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    df = load_corpus(args.corpus).query("split == @args.split").reset_index(drop=True)
    if args.cap and len(df) > args.cap:
        # train/dev: same rows as src.corpora.train_mix / dev_mix; test: same rows as train_mix --test_cap
        df = balance(df).head(args.cap) if args.split in ("train", "dev") else df.sample(args.cap, random_state=0)
    path = cache_path(args.tag, args.quant, args.corpus, args.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["id", "score", "ppl", "xppl"])
    todo = df[~df.id.isin(done.id)]
    print(f"{args.corpus}/{args.split}: {len(df)} requested, {len(done)} cached, {len(todo)} to score", flush=True)
    if len(todo) == 0:
        return

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.observer, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    observer = load_model(args.observer, args.quant, device).eval()
    performer = load_model(args.performer, args.quant, device).eval()

    t0 = time.time()
    chunk = 1000
    for i in range(0, len(todo), chunk):
        part = todo.iloc[i:i + chunk]
        feats = binoculars_features(part.text.values, tok, observer, performer, args.max_len, args.bs, device)
        new = pd.DataFrame({"id": part.id.values, "score": feats[:, 0], "ppl": feats[:, 1], "xppl": feats[:, 2]})
        done = pd.concat([done, new], ignore_index=True)
        done.to_parquet(path, index=False)
        print(f"  {min(i + chunk, len(todo))}/{len(todo)} scored, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
