"""Generative evaluation: produce the model's answers on the held-out inputs (for the judge to
compare) and score the same three regression probes as the classification pipeline.

Unlike the classification evaluator there is no macro-F1 here. The task axis is computed later by
judge.py from the base-vs-candidate answers; this module just generates and stores each model's
answers, plus the regression-probe scores, reusing evaluate_from_config's load/gen/probe helpers.
"""

import argparse
import json
import os

import common_p2 as c
import evaluate_from_config as ef


def evaluate_gen(cfg, name, adapter=None, batch=None, max_new=384):
    if batch is None:
        batch = cfg.get("train", {}).get("early_stopping", {}).get("eval_batch", 8)
    gold = c.read_jsonl(c.data_path(cfg, "gold"))
    model, tok = ef._load(cfg["base_model"], adapter)
    print(f"[eval-gen] {name}: base={cfg['base_model']}" + (f" + {adapter}" if adapter else ""))

    answers = ef._gen(model, tok, [r["messages"] for r in gold], max_new, batch)
    sentinel = ef._probe(model, tok, c.probe_path(cfg, "sentinel"), "any", 128, batch)
    reasoning = ef._probe(model, tok, c.probe_path(cfg, "reasoning_probes"), "any", 256, batch)
    tools = ef._probe(model, tok, c.probe_path(cfg, "tool_probes"), "all", 128, batch)
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    res = {"name": name, "adapter": adapter,
           "tickets": [r["messages"][-1]["content"] for r in gold],   # the user turn = the ticket
           "answers": answers,
           "sentinel": sentinel, "reasoning": reasoning, "tools": tools}
    path = os.path.join(c.out_dir(cfg), f"result_{name}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[eval-gen] {name}: wrote {len(answers)} answers + probes "
          f"(sentinel={sentinel and round(sentinel['score'],3)} "
          f"reasoning={reasoning and round(reasoning['score'],3)} "
          f"tools={tools and round(tools['score'],3)}) -> {path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()
    cfg = c.load_config()
    evaluate_gen(cfg, args.name, adapter=args.adapter, batch=args.batch)


if __name__ == "__main__":
    main()
