"""Compute and cache Binoculars features (score, perplexity, cross-perplexity) for any corpus split.

Binoculars (Hans et al., 2024): score = PPL_performer(text) / X-PPL(observer, performer)(text).
A lower score means the text is more likely machine-generated. The observer is the base model and the
performer the instruct model of the same family, so both share one tokenizer.

Cache: outputs/scores/<tag>_<quant>/<corpus>__<split>.parquet with columns id, score, ppl, xppl.
Re-running only scores ids that are not cached yet.

Usage (defaults = the main pair used in the reported results, needs ~60 GB of GPU memory):
    python scripts/score_binoculars.py --corpus llmtrace --split test
    python scripts/score_binoculars.py --corpus coat --split train --cap 16000
    python scripts/score_binoculars.py --corpus coat --split test --observer Qwen/Qwen3-4B-Base \
        --performer Qwen/Qwen3-4B --tag qwen3_4b --quant nf4 --bs 16      # fits a 12 GB GPU
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.corpora import balance, load_corpus

ROOT = Path(__file__).resolve().parents[1]


def load_model(name: str, quant: str, device):
    if quant == "nf4":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        return AutoModelForCausalLM.from_pretrained(name, quantization_config=cfg, device_map={"": 0})
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device)


@torch.no_grad()
def binoculars_features(texts, tok, observer, performer, max_len, bs, device):
    """Per-text (score, ppl, xppl), processed in length-sorted batches; returned in the original order."""
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
    ap.add_argument("--observer", default="Qwen/Qwen3-14B-Base")
    ap.add_argument("--performer", default="Qwen/Qwen3-14B")
    ap.add_argument("--quant", choices=["nf4", "bf16"], default="bf16")
    ap.add_argument("--tag", default="qwen3_14b")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    df = load_corpus(args.corpus).query("split == @args.split").reset_index(drop=True)
    if args.cap and len(df) > args.cap:
        # train/dev: same rows as src.corpora.train_mix / dev_mix
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
