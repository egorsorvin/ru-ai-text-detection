# Cross-corpus generalization of Russian AI-generated text detectors

Detectors trained on the 2022-era CoAT corpus are evaluated on newer Russian
corpora (LLMTrace-ru, AINL-Eval 2025) and on a small held-out set generated with 2026 models; a
zero-shot Binoculars detector with Russian-capable model pairs and an encoder + Binoculars-features
model trained on corpus mixes are compared under a leave-one-corpus-out protocol.

## Layout

| Path | What |
|---|---|
| `src/data.py` | CoAT loader; official test labels are closed, so HF `validation` is used as test and 10% of train as dev |
| `src/external_data.py` | LLMTrace (Russian, binary) and AINL-Eval 2025 loaders in the same frame format |
| `src/corpora.py` | Balanced corpus mixes, leave-one-corpus-out splits, generated test sets |
| `scripts/baseline_tfidf.py` | TF-IDF (word + char n-grams) + logistic regression |
| `scripts/train_encoder.py` | Custom PyTorch fine-tuning loop for ruBERT / ruRoBERTa on CoAT |
| `scripts/eval_transfer.py` | Evaluate CoAT-trained detectors on every test set |
| `scripts/binoculars.py`, `scripts/score_binoculars.py` | Zero-shot Binoculars with a base/instruct pair; cached per-text features |
| `scripts/train_mix.py` | Encoder (+ Binoculars features) trained on corpus mixes |
| `scripts/generate_set.py` | Held-out test set from unseen generators (continue / paraphrase / simplify) |
| `run_all.sh`, `docs/vast_ai.md` | Full pipeline for one 80 GB GPU and how to run it on a rented instance |
| `outputs/results/*.json` | Metrics of every run (accuracy, macro-F1, AUROC, per-generator accuracy) |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
export HF_HOME=/path/to/hf_cache
```

Datasets download automatically from HuggingFace (`RussianNLP/coat`, `iitolstykh/LLMTrace_detection`,
`iis-research-team/AINL-Eval-2025`).

## Reproduce

```bash
python scripts/baseline_tfidf.py
python scripts/train_encoder.py --model ai-forever/ruRoberta-large --run ruroberta --bs 16 --lr 1e-5
python scripts/eval_transfer.py --run ruroberta --model ai-forever/ruRoberta-large
bash run_all.sh            # generation, Binoculars features, mixed training (80 GB GPU)
python scripts/show_results.py
```
