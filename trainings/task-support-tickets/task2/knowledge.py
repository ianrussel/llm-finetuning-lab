"""Knowledge-absorption axis: the third axis next to task and regression (a touch, not a hard gate).

Scores the base model and the fine-tuned model CLOSED-BOOK (no retrieval) on the held-out corpus
Q&A set built by build_knowledge.py ("which DAOS component does this issue affect?"), and reports
the knowledge gain = fine-tuned - base. Positive means the fine-tune absorbed some domain knowledge
beyond the task label; near-zero is the expected, acceptable outcome for a small classification
fine-tune. It is reported alongside the gate, not used to accept or reject.

Reuses the eval load/gen helpers, so it is model-agnostic like the rest of the pipeline.

Run from this folder (GPU), after a pipeline run, pointing at the accepted adapter:
    ../../../.venv/bin/python knowledge.py --adapter runs/jira-issue-type-replay-s1
"""

import argparse
import json
import os

import torch

import common_p2 as c
import evaluate_from_config as ef


def score(cfg, adapter, probes, batch, max_new=64):
    model, tok = ef._load(cfg["base_model"], adapter)
    try:
        raw = ef._gen(model, tok, [[{"role": "user", "content": p["question"]}] for p in probes],
                      max_new, batch)
        return c.score_probes(raw, probes, "any")   # any correct component named = a hit
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="the accepted adapter dir (from runs/)")
    ap.add_argument("--probes", default="data/knowledge_probes.jsonl")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    cfg = c.load_config()
    path = c.resolve(args.probes)
    if not os.path.exists(path):
        raise SystemExit(f"no knowledge set at {path}; run build_knowledge.py first")
    probes = c.read_jsonl(path)
    print(f"[knowledge] {len(probes)} held-out closed-book probes (DAOS components)")

    base_score = score(cfg, None, probes, args.batch)
    ft_score = score(cfg, args.adapter, probes, args.batch)
    gain = ft_score - base_score

    out = {"n": len(probes), "base": round(base_score, 4), "fine_tuned": round(ft_score, 4),
           "knowledge_gain": round(gain, 4), "adapter": args.adapter}
    with open(os.path.join(c.out_dir(cfg), "knowledge.json"), "w") as f:
        json.dump(out, f, indent=2)

    verdict = ("some domain knowledge stuck" if gain >= 0.02
               else "little/no domain knowledge shift (expected for a small fine-tune)")
    print(f"[knowledge] base={base_score:.3f}  fine-tuned={ft_score:.3f}  "
          f"gain={gain:+.3f}  -> {verdict}")
    print("[knowledge] third axis, reported not gated; wrote runs/knowledge.json")


if __name__ == "__main__":
    main()
