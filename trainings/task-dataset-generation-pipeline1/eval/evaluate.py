"""Measure the resolution model on the two axes, on the SAME sets every time.

  axis 1 (task)       : accuracy + macro-F1 on data/gold.jsonl (binary Done/Won't Do).
                        The model emits a <think> trace then the label; c_predict_label
                        reads the label from the post-</think> tail.
  axis 2 (regression) : three fixed probe sets, to catch loss of ability after training:
                          sentinel.jsonl        general knowledge (same 12 as track A)
                          reasoning_probes.jsonl step-by-step arithmetic/logic
                          tool_probes.jsonl      function-calling (must include name + args)

Run from the track root.
    ../../.venv/bin/python eval/evaluate.py --name base
    ../../.venv/bin/python eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth

Writes data/result_<name>.json and per-row preds to data/preds_<name>.jsonl.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import common_c as common

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SENTINEL = os.path.join(_HERE, "sentinel.jsonl")
REASONING = os.path.join(_HERE, "reasoning_probes.jsonl")
TOOLS = os.path.join(_HERE, "tool_probes.jsonl")


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load(adapter):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=pick_dtype())
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tok


def generate(model, tok, prompts, max_new_tokens, batch):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
        print(f"  {min(i + batch, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return outs


def eval_task(model, tok, rows, labels, batch):
    prompts = [r["messages"] for r in rows]   # gold rows are system+user only
    raw = generate(model, tok, prompts, max_new_tokens=384, batch=batch)
    gold = [r["label"] for r in rows]
    pred = [common.c_predict_label(o, labels) for o in raw]
    n = len(rows)
    acc = sum(1 for g, p in zip(gold, pred) if g == p) / n
    valid = sum(1 for p in pred if p is not None) / n
    f1 = common.macro_f1(gold, pred, labels)
    per_row = [{"id": r["id"], "gold": g, "raw": o, "pred": p, "correct": g == p}
               for r, g, o, p in zip(rows, gold, raw, pred)]
    return {"n": n, "accuracy": acc, "macro_f1": f1, "valid_label_rate": valid,
            "rows": per_row}


def eval_probes(model, tok, path, batch, mode="any", max_new=48):
    """Score a probe set. mode='any': output contains any expected answer (sentinel,
    reasoning). mode='all': output contains every expected substring (tool calls)."""
    if not os.path.exists(path):
        return None
    probes = common.read_jsonl(path)
    prompts = [[{"role": "user", "content": p["question"]}] for p in probes]
    raw = generate(model, tok, prompts, max_new_tokens=max_new, batch=batch)
    rows, ok = [], 0
    for p, o in zip(probes, raw):
        low = o.lower()
        ans = [a.lower() for a in p["answers"]]
        hit = all(a in low for a in ans) if mode == "all" else any(a in low for a in ans)
        ok += hit
        rows.append({"q": p["question"], "want": p["answers"], "got": o[:160], "hit": hit})
    return {"n": len(probes), "score": ok / len(probes), "probes": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    labels = common.load_labels()
    gold = common.read_jsonl(f"{common.DATA}/gold.jsonl")
    if args.limit:
        gold = gold[:args.limit]

    model, tok = load(args.adapter)
    print(f"model: {BASE_MODEL}" + (f" + adapter {args.adapter}" if args.adapter else "")
          + f"  dtype={pick_dtype()}  device={model.device}")

    print(f"\naxis 1, task on {len(gold)} gold rows:")
    task = eval_task(model, tok, gold, labels, args.batch)
    print(f"  accuracy={task['accuracy']:.3f}  macro_f1={task['macro_f1']:.3f}  "
          f"valid_label_rate={task['valid_label_rate']:.3f}")
    per_row = task.pop("rows")
    common.write_jsonl(f"{common.DATA}/preds_{args.name}.jsonl", per_row)

    print("\naxis 2, regression probes:")
    sent = eval_probes(model, tok, SENTINEL, args.batch, "any", 32)
    reason = eval_probes(model, tok, REASONING, args.batch, "any", 256)
    tools = eval_probes(model, tok, TOOLS, args.batch, "all", 128)
    for label, r in [("sentinel", sent), ("reasoning", reason), ("tools", tools)]:
        if r:
            print(f"  {label:<10} {r['score']:.3f} ({int(r['score']*r['n'])}/{r['n']})")

    out = {"name": args.name, "base_model": BASE_MODEL, "adapter": args.adapter,
           "task": task, "sentinel": sent, "reasoning": reason, "tools": tools}
    path = f"{common.DATA}/result_{args.name}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
