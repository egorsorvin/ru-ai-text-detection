"""Evaluate detectors trained on CoAT on external Russian corpora (cross-corpus transfer).

Usage:
    python scripts/eval_transfer.py --run rubert --model ai-forever/ruBert-base
    python scripts/eval_transfer.py --run ruroberta --model ai-forever/ruRoberta-large --bs 32
    python scripts/eval_transfer.py --run tfidf_lr            # sklearn pipeline saved by baseline_tfidf.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.corpora import test_sets
from src.evaluate import report

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=None, help="HF model name for encoder runs; omit for tfidf")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--only", default="", help="comma-separated test names to evaluate, e.g. gen_qwen38_27b_test")
    args = ap.parse_args()

    sets = test_sets()
    if args.only:
        sets = {k: v for k, v in sets.items() if k in args.only.split(",")}
    if args.model is None:
        pipe = joblib.load(ROOT / "outputs" / "checkpoints" / f"{args.run}.joblib")
        for name, df in sets.items():
            score = pipe.predict_proba(df.text)[:, 1]
            report(args.run, name, df, (score >= 0.5).astype(int), score)
        return

    from scripts.train_encoder import Collator, TextDataset, predict

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    state = torch.load(ROOT / "outputs" / "checkpoints" / args.run / "best.pt", map_location=device)
    model.load_state_dict(state)
    collate = Collator(tok, args.max_len)
    for name, df in sets.items():
        loader = DataLoader(TextDataset(df.text, df.label), batch_size=args.bs, collate_fn=collate)
        pred, prob = predict(model, loader, device)
        report(args.run, name, df, pred, prob)


if __name__ == "__main__":
    main()
