"""Generate the held-out test set with generators unseen in CoAT, LLMTrace and AINL.

Sources: human texts from the CoAT test split (same domains as the old test).
Tasks (mirroring CoAT's generation tasks):
  continue    - the model continues the first 10 words of a human text to a similar length
  paraphrase  - the model rewrites the human text in its own words
  simplify    - the model rewrites the human text in simpler language
Output: data/gen/<tag>.parquet with columns id, text, label, generator, domain, task, source_id.
Human source texts are included with label 0, so the file is a balanced, self-contained test set.
Resumable: results are flushed every --flush items and existing (source_id, task) pairs are skipped.

Usage:
    python scripts/generate_set.py --model Qwen/Qwen3-4B --tag qwen3_4b --n 20 --quant nf4      # local smoke
    python scripts/generate_set.py --model Qwen/Qwen3.8-27B --tag qwen38_27b --n 1000 --backend vllm
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"E:\hf_cache" if os.name == "nt" else os.path.expanduser("~/hf_cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data import get_splits

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "gen"
SEED = 42

PROMPTS = {
    "continue": "Продолжи этот текст на русском языке. Напиши не менее {words} слов, несколько предложений в том же стиле. Выведи только продолжение, без вступлений и пояснений.\n\nТекст: {text}",
    "paraphrase": "Перескажи этот текст своими словами на русском языке, сохранив смысл и примерно ту же длину. Выведи только пересказ, без вступлений и пояснений.\n\nТекст: {text}",
    "simplify": "Перепиши этот текст на русском языке более простым языком, короткими предложениями, сохранив смысл. Выведи только переписанный текст, без вступлений и пояснений.\n\nТекст: {text}",
}
PREAMBLE = re.compile(r"^\s*(вот|конечно|разумеется|хорошо)[^\n]{0,80}:\s*\n", re.I)
REFUSAL = re.compile(r"^\s*(извините|к сожалению|я не могу|как (языковая )?модель|i'm sorry|i cannot)", re.I)
LABEL = re.compile(r"^\s*(текст|продолжение|пересказ|упрощ[её]нный текст|ответ)\s*:\s*", re.I)


def _norm_words(s: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", s.lower()).split()


def latin_share(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(1, len(letters))


META = re.compile(r"(перепиш|перескаж|продолж)[а-я]*\s+(этот|текст)|как\s+(языковая\s+)?модель|исходн[ыоа][йе]\s+текст", re.I)


def repetition_loop(t: str, n: int = 4, thr: float = 0.2) -> bool:
    """True if the text degenerates into repeated n-grams (a known failure mode of open generation)."""
    w = t.lower().split()
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return bool(grams) and (len(grams) - len(set(grams))) / len(grams) > thr


def build_jobs(n_per_task: int, min_words: int, max_words: int, seed: int) -> pd.DataFrame:
    _, _, test = get_splits()
    human = test[test.label == 0].copy()
    human["words"] = human.text.str.split().str.len()
    human = human[(human.words >= min_words) & (human.words <= max_words)]
    rng = np.random.default_rng(seed)
    jobs = []
    for task in PROMPTS:
        src = human.sample(n_per_task, random_state=int(rng.integers(1 << 30)))
        for r in src.itertuples():
            if task == "continue":
                head = " ".join(r.text.split()[:10])
                prompt = PROMPTS[task].format(words=int(r.words), text=head)
            else:
                prompt = PROMPTS[task].format(text=r.text)
            jobs.append({"source_id": r.id, "task": task, "prompt": prompt, "source_text": r.text,
                         "words": int(r.words), "head": head if task == "continue" else ""})
    return pd.DataFrame(jobs)


def clean(raw: str, job) -> str | None:
    t = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    t = PREAMBLE.sub("", t).strip().strip('"«»').strip()
    t = re.sub(r"\*\*|__|(?<!\w)\*(?!\w)|^#+\s*", "", t, flags=re.M).strip()  # markdown emphasis / headers
    if REFUSAL.match(t) or latin_share(t) > 0.3:  # off-task: refusal or non-Russian output
        return None
    t = LABEL.sub("", t).strip()
    if job.task == "continue":
        head_words = _norm_words(job.head)
        t_words = t.split()
        if _norm_words(" ".join(t_words[: len(head_words) + 3]))[: len(head_words)] == head_words:
            consumed, k = 0, 0  # drop raw tokens until the normalized head is consumed (punctuation-insensitive)
            while k < len(t_words) and consumed < len(head_words):
                consumed += len(_norm_words(t_words[k]))
                k += 1
            t = " ".join(t_words[k:])
        elif t_words and _norm_words(t_words[0]) == head_words[-1:]:
            t = " ".join(t_words[1:])  # model echoed the last prompt word
        t = LABEL.sub("", t).strip()
        if len(t.split()) < max(8, int(0.4 * (job.words - 10))):
            return None
        t = job.head + " " + t
    words = t.split()
    if len(words) < 5 or t.strip().lower() == job.source_text.strip().lower():
        return None
    if repetition_loop(t) or META.search(t):  # degenerate output or the model talking about the task
        return None
    cap = int(job.words * 1.2) + 5  # keep generated length close to the human source (drop whole sentences)
    if len(words) > cap:
        t = trim_to_sentence(t, max_words=cap, min_keep=int(0.6 * cap))
    return t


def trim_to_sentence(t: str, max_words: int, min_keep: int) -> str:
    """Drop trailing sentences so that the text has <= max_words words, keeping >= min_keep.
    If no sentence boundary satisfies both, return the text untrimmed (never cut mid-sentence)."""
    ends = [m.end() for m in re.finditer(r"[.!?…»\"]+(?=\s|$)", t)]
    for e in reversed(ends):
        n = len(t[:e].split())
        if min_keep <= n <= max_words:
            return t[:e].strip()
    return t


class HFBackend:
    def __init__(self, model, quant, max_new):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"device_map": {"": 0}}
        if quant == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kw["dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
        self.max_new = max_new

    def chat_prompts(self, prompts):
        out = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            try:
                s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            out.append(s)
        return out

    def generate(self, prompts, max_new_list):
        enc = self.tok(self.chat_prompts(prompts), return_tensors="pt", padding=True).to("cuda")
        with self.torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=max(max_new_list), do_sample=True,
                                      temperature=0.7, top_p=0.9, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True) for g in gen]


class VLLMBackend:
    def __init__(self, model, quant, max_new, gpu_util=0.85, eager=False):
        from vllm import LLM, SamplingParams
        kw = dict(model=model, dtype="bfloat16", max_model_len=1536, gpu_memory_utilization=gpu_util,
                  max_num_seqs=64, enforce_eager=eager)
        try:
            self.llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})  # text-only use of multimodal models
        except (TypeError, ValueError):
            self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.SP = SamplingParams
        self.max_new = max_new

    def generate(self, prompts, max_new_list):
        texts = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            try:
                texts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                texts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        sp = self.SP(temperature=0.7, top_p=0.9, max_tokens=max(max_new_list), seed=SEED)
        return [o.outputs[0].text for o in self.llm.generate(texts, sp)]


def write_testset(machine: pd.DataFrame, tag: str) -> Path:
    _, _, test = get_splits()
    human = test[test.id.isin(machine.source_id.unique())]
    task_code = {t: i + 1 for i, t in enumerate(PROMPTS)}
    final = pd.concat([
        pd.DataFrame({"id": machine.source_id.values * 10 + machine.task.map(task_code).values,  # stable ids
                      "text": machine.text, "label": 1, "generator": tag, "task": machine.task, "source_id": machine.source_id}),
        pd.DataFrame({"id": human.id.values * 10, "text": human.text.values, "label": 0, "generator": "Human",
                      "task": "human", "source_id": human.id.values}),
    ], ignore_index=True)
    final["domain"] = "coat"
    final["corpus"] = f"gen_{tag}"
    final["split"] = "test"
    out = OUT_DIR / f"{tag}_testset.parquet"
    final.to_parquet(out, index=False)
    print(f"test set: {len(machine)} machine + {len(human)} human -> {out}", flush=True)
    return out


def reclean(tag: str, n: int, min_words: int, max_words: int):
    """Re-derive cleaned texts from stored raw outputs with the current clean() and rebuild the test set."""
    path = OUT_DIR / f"{tag}.parquet"
    gen = pd.read_parquet(path)
    jobs = build_jobs(n, min_words, max_words, SEED)
    merged = gen.merge(jobs[["source_id", "task", "source_text", "words", "head"]], on=["source_id", "task"], how="left")
    assert merged.source_text.notna().all(), "job table does not match the stored generations (different --n?)"
    texts = [clean(r.raw, r) for r in merged.itertuples()]
    merged["text"] = texts
    kept = merged[merged.text.notna()]
    print(f"{tag}: {len(merged)} raw outputs, {len(merged) - len(kept)} dropped by clean()", flush=True)
    kept[["source_id", "task", "text", "raw"]].to_parquet(path, index=False)
    write_testset(kept, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reclean", action="store_true", help="re-apply clean() to stored raw outputs and rebuild the test set")
    ap.add_argument("--n", type=int, default=1000, help="sources per task")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--quant", choices=["nf4", "bf16"], default="bf16")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--min_words", type=int, default=15)
    ap.add_argument("--max_words", type=int, default=200)
    ap.add_argument("--flush", type=int, default=64)
    ap.add_argument("--gpu_util", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    ap.add_argument("--eager", action="store_true", help="vLLM enforce_eager (no CUDA graphs; use if graph capture OOMs)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.reclean:
        reclean(args.tag, args.n, args.min_words, args.max_words)
        return
    if not args.model:
        ap.error("--model is required unless --reclean")
    path = OUT_DIR / f"{args.tag}.parquet"
    jobs = build_jobs(args.n, args.min_words, args.max_words, SEED)
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["source_id", "task"])
    done_keys = set(zip(done.source_id, done.task)) if len(done) else set()
    todo = jobs[[k not in done_keys for k in zip(jobs.source_id, jobs.task)]].reset_index(drop=True)
    print(f"{args.tag}: {len(jobs)} jobs, {len(done)} done, {len(todo)} to generate", flush=True)
    if len(todo) == 0:
        return

    backend = (VLLMBackend(args.model, args.quant, 0, args.gpu_util, args.eager) if args.backend == "vllm"
               else HFBackend(args.model, args.quant, 0))
    rows = [] if len(done) == 0 else done.to_dict("records")
    t0, dropped = time.time(), 0
    for i in range(0, len(todo), args.bs):
        part = todo.iloc[i:i + args.bs]
        max_new = [min(int(w * 2.2) + 20, 400) for w in part.words]
        raws = backend.generate(part.prompt.tolist(), max_new)
        for job, raw in zip(part.itertuples(), raws):
            text = clean(raw, job)
            if text is None:
                dropped += 1
                continue
            rows.append({"source_id": job.source_id, "task": job.task, "text": text, "raw": raw})
        if (i // args.bs) % max(1, args.flush // args.bs) == 0 or i + args.bs >= len(todo):
            pd.DataFrame(rows).to_parquet(path, index=False)
            print(f"  {min(i + args.bs, len(todo))}/{len(todo)} generated, {dropped} dropped, {time.time()-t0:.0f}s", flush=True)

    machine = pd.DataFrame(rows)
    write_testset(machine, args.tag)
    for task in PROMPTS:
        s = machine[machine.task == task].iloc[0]
        print(f"\n[{task}] {s.text[:300]}")


if __name__ == "__main__":
    main()
