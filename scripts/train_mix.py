"""Mixed-corpus training with leave-one-corpus-out evaluation, optionally with Binoculars features.

Our method = pretrained encoder + Binoculars features (score, log PPL, log X-PPL) concatenated
to the [CLS] vector before the classification head, trained on a mix of corpora.

Usage (defaults reproduce the reported runs):
    python scripts/train_mix.py --train coat,llmtrace --run mix_coat+llmtrace
    python scripts/train_mix.py --train coat,llmtrace --bino --run mix_coat+llmtrace_bino
Evaluation runs on the test split of every corpus and of every generated set in data/gen.
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from scripts.score_binoculars import load_features
from scripts.train_encoder import predict, train_step
from src.corpora import dev_mix, test_sets, train_mix
from src.evaluate import report

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["score", "log_ppl", "log_xppl"]


class MixDataset(Dataset):
    def __init__(self, df: pd.DataFrame, use_feats: bool):
        self.texts = df.text.tolist()
        self.labels = df.label.tolist()
        self.feats = df[FEATS].values.astype(np.float32) if use_feats else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        f = self.feats[i] if self.feats is not None else None
        return self.texts[i], self.labels[i], f


class MixCollator:
    def __init__(self, tokenizer, max_len):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts, labels, feats = zip(*batch)
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        if feats[0] is not None:
            enc["feats"] = torch.tensor(np.stack(feats), dtype=torch.float32)
        return dict(enc)


class EncoderWithFeats(nn.Module):
    """[CLS] vector of a pretrained encoder (+ projected extra features) -> 2-way head."""

    def __init__(self, model_name: str, n_feats: int, feat_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.n_feats = n_feats
        self.feat_proj = nn.Sequential(nn.Linear(n_feats, feat_dim), nn.GELU()) if n_feats else None
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + (feat_dim if n_feats else 0), 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, feats=None, **kw):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kw).last_hidden_state[:, 0]
        if self.n_feats:
            h = torch.cat([h, self.feat_proj(feats)], dim=-1)
        logits = self.head(self.dropout(h))
        loss = self.loss_fn(logits.float(), labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)


def attach_feats(df: pd.DataFrame, tag: str, quant: str, allow_partial: bool) -> pd.DataFrame:
    """Merge cached Binoculars features by (corpus, split, id). Rows without features are dropped."""
    parts = []
    for (corpus, split), g in df.groupby(["corpus", "split"]):
        f = load_features(tag, quant, corpus, split)
        f["log_ppl"] = np.log(f.ppl)
        f["log_xppl"] = np.log(f.xppl)
        merged = g.merge(f[["id", "score", "log_ppl", "log_xppl"]], on="id", how="inner")
        if len(merged) < len(g) and not allow_partial:
            raise RuntimeError(f"{corpus}/{split}: features for {len(merged)}/{len(g)} rows only")
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ai-forever/ruRoberta-large")
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", required=True, help="comma-separated corpora, e.g. coat,llmtrace")
    ap.add_argument("--cap", type=int, default=16000, help="balanced rows per corpus in the train mix")
    ap.add_argument("--bino", action="store_true", help="add Binoculars features")
    ap.add_argument("--bino_tag", default="qwen3_14b")
    ap.add_argument("--bino_quant", default="bf16")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--test_cap", type=int, default=0, help="smoke test: evaluate on at most N texts per test set")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda")
    out_dir = ROOT / "outputs" / "checkpoints" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    corpora = args.train.split(",")

    train_df = train_mix(corpora, args.cap)
    dev_df = dev_mix(corpora, cap=1500)
    tests = test_sets()
    if args.test_cap:
        tests = {k: v.sample(min(len(v), args.test_cap), random_state=0).reset_index(drop=True) for k, v in tests.items()}
    if args.bino:
        train_df = attach_feats(train_df, args.bino_tag, args.bino_quant, allow_partial=True)
        dev_df = attach_feats(dev_df, args.bino_tag, args.bino_quant, allow_partial=True)
        tests = {k: attach_feats(v, args.bino_tag, args.bino_quant, allow_partial=True) for k, v in tests.items()}
        mu, sd = train_df[FEATS].mean(), train_df[FEATS].std() + 1e-6
        for d in [train_df, dev_df, *tests.values()]:
            d[FEATS] = (d[FEATS] - mu) / sd
    print(f"train {len(train_df):,} from {corpora} | dev {len(dev_df):,} | tests "
          + ", ".join(f"{k}={len(v):,}" for k, v in tests.items()), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    collate = MixCollator(tok, args.max_len)
    mk = lambda df, shuffle: DataLoader(MixDataset(df, args.bino), batch_size=args.bs if shuffle else 2 * args.bs,
                                        shuffle=shuffle, collate_fn=collate)
    train_loader, dev_loader = mk(train_df, True), mk(dev_df, False)

    model = EncoderWithFeats(args.model, n_feats=len(FEATS) if args.bino else 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    best_acc, step, t0 = 0.0, 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            loss = train_step(model, batch, optimizer, scheduler, device)
            step += 1
            if step % 100 == 0:
                print(f"ep {epoch} step {step}/{total_steps} loss {loss:.4f} {time.time()-t0:.0f}s", flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                pred, prob = predict(model, dev_loader, device)
                acc = float((pred == dev_df.label.values).mean())
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), out_dir / "best.pt")
                print(f"== step {step} dev acc {acc:.4f} (best {best_acc:.4f})", flush=True)

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    for name, df in tests.items():
        pred, prob = predict(model, mk(df, False), device)
        report(args.run, name, df, pred, prob, save=not args.test_cap)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
