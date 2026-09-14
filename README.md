# Cross-era generalization of Russian AI-generated text detectors

Detectors of machine-generated Russian text are usually trained and tested on one corpus. This repository
measures what happens when the generators change. Detectors trained on the 2022-era CoAT corpus are evaluated
on two newer corpora, LLMTrace-ru and AINL-Eval 2025, and on two held-out test sets generated with 2026 models
(T-pro 2.0 and Qwen3.8-27B). Three families of detectors are compared under a leave-one-corpus-out protocol:
fine-tuned encoders, zero-shot Binoculars with Russian-capable model pairs, and an encoder trained on corpus
mixtures with Binoculars features.

## Key findings

- **Detectors do not transfer across generator eras.** ruRoBERTa-large trained on CoAT reaches 0.95 AUROC on
  CoAT and 0.52-0.64 on the newer corpora. On our 2026 sets the human texts are the CoAT ones and only the
  generator changes: AUROC drops to 0.58-0.62, and the model flags 14-17% of machine texts.
- **The shift is symmetric.** Models trained only on modern corpora are inverted on CoAT (0.46-0.48 AUROC), and
  so is zero-shot Binoculars (0.40-0.44). Generators of 2020-2021 wrote less fluently than people; current
  generators write more predictably.
- **Corpus diversity drives transfer.** Adding LLMTrace (38 generators, 8 domains) to CoAT raises AUROC on the
  unseen AINL from 0.64 to 0.91 and reaches about 0.80 on the unseen 2026 sets. Adding the narrower AINL helps
  much less.
- **Binoculars works on Russian with a Russian-capable pair**, 0.61-0.80 AUROC on modern corpora, but only for
  free continuation. Paraphrase and simplification of human texts stay at 0.51-0.66. Pair size and
  native-Russian pretraining change little. A threshold calibrated on CoAT is useless on newer data, while a
  threshold calibrated on other modern corpora is within 0.04 balanced accuracy of in-domain calibration.
- **Binoculars features do not improve a multi-corpus encoder**: differences range from -0.03 to +0.03 AUROC
  (single seed).
- **The practical regime is weak.** At a 5% false-positive rate on human texts, the best models catch 36-44% of
  machine texts from the unseen 2026 generators.

## Results

AUROC on the test split of every corpus. Cells where the corpus or generator was not seen in training are the
transfer results.

| Detector | CoAT | LLMTrace-ru | AINL-Eval | Gen: Qwen3.8-27B | Gen: T-pro 2.0 |
|---|---|---|---|---|---|
| TF-IDF + LR, trained on CoAT | 0.85 | 0.43 | 0.71 | 0.57 | 0.55 |
| ruBERT-base, trained on CoAT | 0.94 | 0.44 | 0.63 | 0.60 | 0.60 |
| ruRoBERTa-large, trained on CoAT | 0.95 | 0.52 | 0.64 | 0.62 | 0.58 |
| Binoculars Qwen3-4B, zero-shot | 0.44 | 0.62 | 0.79 | 0.67 | 0.70 |
| Binoculars YandexGPT-5-Lite-8B, zero-shot | 0.40 | 0.62 | 0.80 | 0.57 | 0.60 |
| Binoculars Qwen3-14B, zero-shot | 0.43 | 0.61 | 0.78 | 0.68 | 0.70 |
| ruRoBERTa-large, CoAT + LLMTrace | 0.92 | 0.99 | 0.91 | 0.81 | 0.80 |
| &nbsp;&nbsp;+ Binoculars features | 0.92 | 0.99 | 0.90 | 0.81 | 0.79 |
| ruRoBERTa-large, CoAT + AINL | 0.93 | 0.67 | 0.99 | 0.68 | 0.64 |
| &nbsp;&nbsp;+ Binoculars features | 0.93 | 0.68 | 0.99 | 0.69 | 0.68 |
| ruRoBERTa-large, LLMTrace + AINL | 0.46 | 0.99 | 0.99 | 0.82 | 0.81 |
| &nbsp;&nbsp;+ Binoculars features | 0.48 | 0.99 | 0.99 | 0.80 | 0.81 |
| ruRoBERTa-large, all three corpora | 0.90 | 0.98 | 0.99 | 0.79 | 0.79 |
| &nbsp;&nbsp;+ Binoculars features | 0.92 | 0.99 | 0.99 | 0.80 | 0.81 |

Leave-one-corpus-out summary, AUROC on the test set that was not used for training:

| Trained on | Unseen test | ruRoBERTa on CoAT only | Mix | Mix + Binoculars | Binoculars Qwen3-14B |
|---|---|---|---|---|---|
| CoAT + LLMTrace | AINL-Eval | 0.64 | 0.91 | 0.90 | 0.78 |
| CoAT + AINL | LLMTrace-ru | 0.52 | 0.67 | 0.68 | 0.61 |
| LLMTrace + AINL | CoAT | n/a | 0.46 | 0.48 | 0.43 |
| all three | Gen: Qwen3.8-27B | 0.62 | 0.79 | 0.80 | 0.68 |
| all three | Gen: T-pro 2.0 | 0.58 | 0.79 | 0.81 | 0.70 |

On the 2026 test sets by generation task and text length. Each cell shows Qwen3.8 / T-pro.

| Detector | Continue | Paraphrase | Simplify | Texts up to 20 words | TPR at 5% FPR |
|---|---|---|---|---|---|
| ruRoBERTa-large, CoAT only | 0.58 / 0.52 | 0.62 / 0.60 | 0.66 / 0.62 | 0.66 / 0.62 | 0.14 / 0.11 |
| Binoculars Qwen3-14B | 0.77 / 0.84 | 0.63 / 0.63 | 0.66 / 0.62 | 0.61 / 0.63 | 0.14 / 0.21 |
| ruRoBERTa-large, all three | 0.84 / 0.85 | 0.74 / 0.71 | 0.79 / 0.82 | 0.68 / 0.65 | 0.38 / 0.36 |
| &nbsp;&nbsp;+ Binoculars features | 0.86 / 0.86 | 0.76 / 0.75 | 0.78 / 0.83 | 0.69 / 0.68 | 0.38 / 0.39 |

Full metrics, including accuracy, macro-F1 and per-generator accuracy, are in `outputs/results/`. Slice and
threshold tables are in `outputs/analysis/`.

## Data

| Corpus | Source | Split protocol |
|---|---|---|
| CoAT (Shamardina et al., 2025) | [`RussianNLP/coat`](https://huggingface.co/datasets/RussianNLP/coat), Apache-2.0 | Official test labels are closed. The HF `validation` split is used as test, 10% of train stratified by generator as dev. |
| LLMTrace, Russian part (Tolstykh et al., 2025) | [`iitolstykh/LLMTrace_detection`](https://huggingface.co/datasets/iitolstykh/LLMTrace_detection), see the dataset card | Binary task: texts labelled `mixed` are dropped. Official splits. |
| AINL-Eval 2025 (Batura et al., 2025) | [`iis-research-team/AINL-Eval-2025`](https://huggingface.co/datasets/iis-research-team/AINL-Eval-2025), Apache-2.0 | Official test labels are closed. `dev_full.csv` is used as test, 5% of train as dev. One empty text is dropped. |

Training mixtures use a balanced sample of 16,000 texts per corpus.

**Held-out 2026 test sets.** `scripts/generate_set.py` takes 1,000 human texts from the CoAT test split per task
and asks each generator to continue the first 10 words, paraphrase, or simplify the text. Each set has about
3,000 machine texts and the matching 2,600 human sources. Cleaning is rule-based and identical for all texts:
removal of reasoning blocks, chat preambles, markdown and prompt echoes; rejection of refusals, non-Russian
outputs, degenerate repetition and copies of the source; continuations are cut at sentence boundaries to at most
1.2 times the source length. Raw generations are stored next to the cleaned text.

## Methods

- **Baselines trained on CoAT**: TF-IDF over word and character n-grams with logistic regression; ruBERT-base and
  ruRoBERTa-large fine-tuned with a custom PyTorch loop.
- **Binoculars** (Hans et al., 2024): the ratio of perplexity to cross-perplexity under a base/instruct pair of the
  same model family, with no training. Pairs: Qwen3-4B, YandexGPT-5-Lite-8B, Qwen3-14B.
- **Multi-corpus training**: ruRoBERTa-large fine-tuned on two or three corpora, evaluated on the corpus left out.
  The `_bino` variant concatenates three Binoculars features of the Qwen3-14B pair to the `[CLS]` vector
  before the classification head.

## Repository layout

| Path | Contents |
|---|---|
| `src/data.py` | CoAT loader and the train/dev/test protocol |
| `src/external_data.py` | LLMTrace-ru and AINL-Eval loaders in the same format |
| `src/corpora.py` | Unified corpus access, balanced mixtures, generated test sets |
| `src/evaluate.py` | Metrics, per-generator breakdown, saving results and per-text predictions |
| `scripts/baseline_tfidf.py` | TF-IDF + logistic regression |
| `scripts/train_encoder.py` | Fine-tuning loop for encoders on CoAT |
| `scripts/eval_transfer.py` | CoAT-trained detectors on every test set |
| `scripts/generate_set.py` | Held-out test sets from unseen generators |
| `scripts/score_binoculars.py` | Binoculars features with a base/instruct pair, cached per text |
| `scripts/eval_binoculars_cache.py` | Zero-shot Binoculars metrics from cached features |
| `scripts/train_mix.py` | Encoder, optionally with Binoculars features, trained on corpus mixtures |
| `scripts/show_results.py` | Results matrix over all runs |
| `scripts/analyze_slices.py` | Metrics by task and length, TPR at fixed FPR |
| `scripts/analyze_thresholds.py` | Binoculars threshold calibration variants |
| `outputs/results/` | Metrics of every run |
| `outputs/analysis/` | Slice and threshold tables |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install bitsandbytes vllm                      # optional: 4-bit Binoculars pairs, generation
```

Datasets and models are downloaded from the Hugging Face Hub on first use. Set `HF_HOME` to choose the cache
location.

## Reproducing the results

The CoAT baselines fit a 12 GB GPU. Generation, Binoculars in bf16 and multi-corpus training were run on one
80 GB GPU, about three GPU hours in total.

```bash
# 1. Baselines trained on CoAT
python scripts/baseline_tfidf.py
python scripts/train_encoder.py --model ai-forever/ruBert-base --run rubert
python scripts/train_encoder.py --model ai-forever/ruRoberta-large --run ruroberta --bs 16 --lr 1e-5

# 2. Held-out 2026 test sets
python scripts/generate_set.py --model t-tech/T-pro-it-2.0 --tag tpro2_32b --n 1000 --backend vllm --bs 64
python scripts/generate_set.py --model Qwen/Qwen3.8-27B --tag qwen38_27b --n 1000 --backend vllm --bs 64

# 3. Baselines on every test set
python scripts/eval_transfer.py --run tfidf_lr
python scripts/eval_transfer.py --run rubert --model ai-forever/ruBert-base
python scripts/eval_transfer.py --run ruroberta --model ai-forever/ruRoberta-large --bs 32

# 4. Binoculars features and zero-shot metrics; defaults use the Qwen3-14B pair in bf16
for c in coat llmtrace ainl gen_tpro2_32b gen_qwen38_27b; do
  python scripts/score_binoculars.py --corpus $c --split test
done
for c in coat llmtrace ainl; do
  python scripts/score_binoculars.py --corpus $c --split dev
  python scripts/score_binoculars.py --corpus $c --split train --cap 16000
done
python scripts/eval_binoculars_cache.py --tag qwen3_14b
# other pairs: add --observer Qwen/Qwen3-4B-Base --performer Qwen/Qwen3-4B --tag qwen3_4b, and
# --observer yandex/YandexGPT-5-Lite-8B-pretrain --performer yandex/YandexGPT-5-Lite-8B-instruct --tag yagpt5_8b

# 5. Multi-corpus training, leave-one-corpus-out
for mix in coat,llmtrace coat,ainl llmtrace,ainl coat,llmtrace,ainl; do
  name=mix_${mix//,/+}; [ "$mix" = "coat,llmtrace,ainl" ] && name=mix_all
  python scripts/train_mix.py --train $mix --run $name
  python scripts/train_mix.py --train $mix --run ${name}_bino --bino
done

# 6. Tables
python scripts/show_results.py
python scripts/analyze_slices.py --tests gen_qwen38_27b_test,gen_tpro2_32b_test
python scripts/analyze_thresholds.py
```

## Limitations

- All numbers come from a single training seed. Differences of a few hundredths are within run-to-run noise.
- Both 2026 generators belong to the Qwen lineage, since T-pro 2.0 is built on Qwen3-32B.
- The three corpora differ in human sources and domains, so cross-corpus drops mix a generator shift with a
  domain shift. The generated sets keep CoAT human texts fixed to isolate the generator shift.
- CoAT and AINL-Eval results use the labelled validation data, not the closed official test sets, so they are
  not directly comparable with published leaderboard numbers.

## References

- T. Shamardina et al. CoAT: Corpus of artificial texts. *Natural Language Processing*, 31(1), 2025.
- T. Shamardina et al. Findings of the RuATD Shared Task 2022 on Artificial Text Detection in Russian. *Dialogue*, 2022. [arXiv:2206.01583](https://arxiv.org/abs/2206.01583)
- I. Tolstykh et al. LLMTrace: A Corpus for Classification and Fine-Grained Localization of AI-Written Text. 2025. [arXiv:2509.21269](https://arxiv.org/abs/2509.21269)
- T. Batura et al. AINL-Eval 2025 Shared Task: Detection of AI-Generated Scientific Abstracts in Russian. 2025. [arXiv:2508.09622](https://arxiv.org/abs/2508.09622)
- A. Hans et al. Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text. *ICML*, 2024. [arXiv:2401.12070](https://arxiv.org/abs/2401.12070)

## License

The code is released under the [MIT License](LICENSE). The corpora and models used here keep their own licenses,
see the links in the Data section.
