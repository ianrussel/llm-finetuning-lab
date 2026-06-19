"""Measure a model on the two axes, on the SAME sets every time.

  axis 1 (task)       : accuracy + macro-F1 on data/gold.jsonl
  axis 2 (regression) : score on sentinel.jsonl, a small fixed set of general
                        known-answer probes, to catch forgetting after training

Used by both the Phase 1 baseline and the Phase 4 re-measure, so it lives in its
own eval/ folder. Run from the track_a_banking77 folder.

Run the base model now to get the pre-training baseline (Phase 1's last step):
    ../../../.venv/bin/python eval/evaluate.py --name base

Later, re-run the identical sets with an adapter to get the real delta:
    ../../../.venv/bin/python eval/evaluate.py --name seed-ft --adapter ./lora-out

Results are written to data/result_<name>.json so Phase 4 can line them up.
Use --limit to smoke-test on a slice before committing to the full 1540 rows.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# common.py lives one level up (shared by every phase); make it importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # same base as modules 1-6
SENTINEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentinel.jsonl")


def pick_dtype():
    """bf16 on Ampere+, fp16 on older CUDA, fp32 on CPU (the module-3 rule)."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load(adapter):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"  # decoder-only: pad on the left so generation aligns
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=pick_dtype())
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tok


def generate(model, tok, prompts, max_new_tokens, batch):
    """Greedy-decode a list of [messages] prompts, batched. Returns texts."""
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                 for m in chunk]
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


def eval_task(model, tok, labels, rows, batch):
    prompts = [[{"role": "system", "content": common.system_prompt(labels)},
                {"role": "user", "content": common.user_of(r)}] for r in rows]
    raw = generate(model, tok, prompts, max_new_tokens=24, batch=batch)
    gold = [common.assistant_of(r) for r in rows]
    pred = [common.predict_label(o, labels) for o in raw]
    n = len(rows)
    acc = sum(1 for g, p in zip(gold, pred) if g == p) / n
    valid = sum(1 for p in pred if p is not None) / n
    f1 = common.macro_f1(gold, pred, labels)
    per_row = [{"query": common.user_of(r), "gold": g, "raw": o, "pred": p,
                "correct": g == p}
               for r, g, o, p in zip(rows, gold, raw, pred)]
    return {"n": n, "accuracy": acc, "macro_f1": f1, "valid_label_rate": valid,
            "rows": per_row}


def sample_examples(rows, k=10):
    """A readable sample that leads with misses, so the file shows real errors
    instead of whatever intent happens to sort first."""
    misses = [r for r in rows if not r["correct"]]
    hits = [r for r in rows if r["correct"]]
    half = k // 2
    return misses[:k - min(half, len(hits))] + hits[:half]


def eval_sentinel(model, tok, batch):
    probes = common.read_jsonl(SENTINEL_PATH)
    prompts = [[{"role": "user", "content": p["question"]}] for p in probes]
    raw = generate(model, tok, prompts, max_new_tokens=32, batch=batch)
    rows, ok = [], 0
    for p, o in zip(probes, raw):
        hit = any(a in o.lower() for a in p["answers"])
        ok += hit
        rows.append({"q": p["question"], "want": p["answers"], "got": o, "hit": hit})
    return {"n": len(probes), "score": ok / len(probes), "probes": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base", help="label for the results file")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter (optional)")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N gold rows")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    labels = common.load_labels()
    gold = common.read_jsonl(f"{common.DATA}/gold.jsonl")
    if args.limit:
        gold = gold[:args.limit]

    model, tok = load(args.adapter)
    print(f"model: {BASE_MODEL}" + (f" + adapter {args.adapter}" if args.adapter else "")
          + f"  dtype={pick_dtype()}  device={model.device}")

    print(f"\naxis 1, task on {len(gold)} gold rows:")
    task = eval_task(model, tok, labels, gold, args.batch)
    print(f"  accuracy={task['accuracy']:.3f}  macro_f1={task['macro_f1']:.3f}  "
          f"valid_label_rate={task['valid_label_rate']:.3f}")

    # full per-row predictions go to their own file; keep a miss-led sample inline
    per_row = task.pop("rows")
    preds_path = f"{common.DATA}/preds_{args.name}.jsonl"
    common.write_jsonl(preds_path, per_row)
    task["examples"] = sample_examples(per_row)
    print(f"  wrote {sum(1 for r in per_row if not r['correct'])} misses / "
          f"{len(per_row)} rows -> {preds_path}")

    print(f"\naxis 2, regression sentinel:")
    sent = eval_sentinel(model, tok, args.batch)
    print(f"  score={sent['score']:.3f} ({int(sent['score']*sent['n'])}/{sent['n']})")

    out = {"name": args.name, "base_model": BASE_MODEL, "adapter": args.adapter,
           "task": task, "sentinel": sent}
    path = f"{common.DATA}/result_{args.name}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
