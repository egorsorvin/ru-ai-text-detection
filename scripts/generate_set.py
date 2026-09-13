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
    "continue": "Продолжи этот текст на русском языке примерно до {words} слов. Выведи только продолжение, без вступлений и пояснений.\n\nТекст: {text}",
    "paraphrase": "Перескажи этот текст своими словами на русском языке, сохранив смысл и примерно ту же длину. Выведи только пересказ, без вступлений и пояснений.\n\nТекст: {text}",
    "simplify": "Перепиши этот текст на русском языке более простым языком, короткими предложениями, сохранив смысл. Выведи только переписанный текст, без вступлений и пояснений.\n\nТекст: {text}",
}
PREAMBLE = re.compile(r"^\s*(вот|конечно|разумеется|хорошо)[^\n]{0,80}:\s*\n", re.I)


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
    if job.task == "continue":
        t = job.head + " " + t
    words = t.split()
    if len(words) < 5 or t.strip() == job.source_text.strip():
        return None
    cap = int(job.words * 1.5) + 10
    if len(words) > cap:
        t = " ".join(words[:cap])
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
    def __init__(self, model, quant, max_new):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.9)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=1000, help="sources per task")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--quant", choices=["nf4", "bf16"], default="bf16")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--min_words", type=int, default=15)
    ap.add_argument("--max_words", type=int, default=200)
    ap.add_argument("--flush", type=int, default=64)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.tag}.parquet"
    jobs = build_jobs(args.n, args.min_words, args.max_words, SEED)
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["source_id", "task"])
    done_keys = set(zip(done.source_id, done.task)) if len(done) else set()
    todo = jobs[[k not in done_keys for k in zip(jobs.source_id, jobs.task)]].reset_index(drop=True)
    print(f"{args.tag}: {len(jobs)} jobs, {len(done)} done, {len(todo)} to generate", flush=True)
    if len(todo) == 0:
        return

    backend = (VLLMBackend if args.backend == "vllm" else HFBackend)(args.model, args.quant, 0)
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
    _, _, test = get_splits()
    human = test[test.id.isin(machine.source_id.unique())]
    final = pd.concat([
        pd.DataFrame({"text": machine.text, "label": 1, "generator": args.tag, "task": machine.task, "source_id": machine.source_id}),
        pd.DataFrame({"text": human.text.values, "label": 0, "generator": "Human", "task": "human", "source_id": human.id.values}),
    ], ignore_index=True)
    final.insert(0, "id", range(len(final)))
    final["domain"] = "coat"
    final["corpus"] = f"gen_{args.tag}"
    final["split"] = "test"
    final.to_parquet(OUT_DIR / f"{args.tag}_testset.parquet", index=False)
    print(f"test set: {len(machine)} machine + {len(human)} human -> {OUT_DIR / (args.tag + '_testset.parquet')}", flush=True)
    for task in PROMPTS:
        s = machine[machine.task == task].iloc[0]
        print(f"\n[{task}] {s.text[:300]}")


if __name__ == "__main__":
    main()
