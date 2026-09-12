"""Fine-tune a pretrained Russian encoder (ruBERT / ruRoBERTa) for human-vs-machine detection.

Usage:
    python scripts/train_encoder.py --model ai-forever/ruBert-base --run rubert
    python scripts/train_encoder.py --model ai-forever/ruRoberta-large --run ruroberta --bs 16 --lr 1e-5
    python scripts/train_encoder.py --limit 2000 --eval_every 20 --epochs 1   # smoke test
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import get_splits
from src.evaluate import report

ROOT = Path(__file__).resolve().parents[1]


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


class Collator:
    """Tokenizes a list of (text, label) pairs into one padded batch."""

    def __init__(self, tokenizer, max_len):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts, labels = zip(*batch)
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return dict(enc)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_step(model, batch, optimizer, scheduler, device):
    batch = to_device(batch, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


@torch.no_grad()
def predict(model, loader, device):
    """Returns (predicted labels, P(machine)) as numpy arrays."""
    model.eval()
    preds, probs = [], []
    for batch in loader:
        batch = to_device(batch, device)
        batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(-1).cpu().numpy())
    model.train()
    return np.concatenate(preds), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ai-forever/ruBert-base")
    ap.add_argument("--run", default="rubert")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N examples per split")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda")
    out_dir = ROOT / "outputs" / "checkpoints" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, test_df = get_splits()
    if args.limit:
        train_df = train_df.sample(args.limit, random_state=0)
        dev_df = dev_df.sample(min(len(dev_df), args.limit), random_state=0).reset_index(drop=True)
        test_df = test_df.sample(min(len(test_df), args.limit), random_state=0).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collate = Collator(tokenizer, args.max_len)
    train_loader = DataLoader(TextDataset(train_df.text, train_df.label), batch_size=args.bs,
                              shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(TextDataset(dev_df.text, dev_df.label), batch_size=2 * args.bs, collate_fn=collate)
    test_loader = DataLoader(TextDataset(test_df.text, test_df.label), batch_size=2 * args.bs, collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    print(f"train {len(train_df):,} | dev {len(dev_df):,} | test {len(test_df):,} | steps {total_steps:,}", flush=True)

    best_acc, step, t0 = 0.0, 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            loss = train_step(model, batch, optimizer, scheduler, device)
            step += 1
            if step % 100 == 0:
                print(f"ep {epoch} step {step}/{total_steps} loss {loss:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                pred, prob = predict(model, dev_loader, device)
                acc = report(args.run, "dev", dev_df, pred, prob, save=False)["accuracy"]
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), out_dir / "best.pt")
                print(f"== step {step} dev acc {acc:.4f} (best {best_acc:.4f})", flush=True)

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    for name, df, loader in [("dev", dev_df, dev_loader), ("test", test_df, test_loader)]:
        pred, prob = predict(model, loader, device)
        report(args.run, name, df, pred, prob, save=not args.limit)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
