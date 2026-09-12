"""Print a results matrix (runs x test sets) and per-generator breakdowns from outputs/results/*.json."""
import json
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "outputs" / "results"


def main(detail: bool):
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"run": d["model"], "test": d["split"], "acc": d["accuracy"],
                     "f1": d["f1_macro"], "auroc": d.get("auroc", float("nan")), "pg": d.get("per_generator", {})})
    df = pd.DataFrame(rows)
    df = df[df.test.str.endswith("_test")]
    for metric in ["auroc", "acc", "f1"]:
        print(f"--- {metric}")
        print(df.pivot(index="run", columns="test", values=metric).round(3).to_string())
    if detail:
        for _, r in df.iterrows():
            if r.test == "coat_test":
                continue
            items = sorted(r.pg.items(), key=lambda kv: (kv[0] != "Human", kv[0]))
            print(f"\n== {r.run} / {r.test}")
            for k, v in items:
                print(f"   {k.split('/')[-1][:40]:40s} {v:.2f}")


if __name__ == "__main__":
    main(detail="--detail" in sys.argv)
