"""
Module 5, step 3: observe the effect of merging.

Scores each model on the support-ticket task (the module 2/3 eval) and also shows its
answer to a couple of off-task general prompts. Run this after merge.py to see the whole
curve: pure base, the blends, and the pure fine-tune.

What to watch: as the fine-tune weight rises, the task scores should climb, while the
general answers may drift. The merge is a dial between "general model" and "task model".

Run from this folder:
    aipy evaluate.py
"""

import gc
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import SYSTEM, GENERAL_PROMPTS, read_jsonl, user_of, assistant_of, parse_label

# (label, path) in increasing fine-tune weight. Skip ones that do not exist yet.
CANDIDATES = [
    ("base",      "Qwen/Qwen2.5-0.5B-Instruct"),
    ("merge-25",  "./merged-25"),
    ("merge-50",  "./merged-50"),
    ("merge-75",  "./merged-75"),
    ("finetuned", "./finetuned-full"),
]
MODELS = [(name, p) for name, p in CANDIDATES
          if p.startswith("Qwen/") or os.path.isdir(p)]

eval_rows = read_jsonl("data/eval.jsonl")

# Fallback chat template: merged models may not carry one, so borrow the base's.
_BASE_CHAT_TEMPLATE = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct").chat_template


def generate(model, tok, messages, max_new_tokens):
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def score(name, path):
    tok = AutoTokenizer.from_pretrained(path)
    if not tok.chat_template:
        tok.chat_template = _BASE_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    model.eval()

    valid = cat_ok = exact = 0
    for row in eval_rows:
        ans = generate(model, tok,
                       [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_of(row)}], 40)
        c, p = parse_label(ans)
        gc_, gp_ = parse_label(assistant_of(row))
        valid += c is not None
        cat_ok += c == gc_
        exact += (c, p) == (gc_, gp_)

    # one general prompt, to eyeball whether off-task ability survives
    general = generate(model, tok, [{"role": "user", "content": GENERAL_PROMPTS[0]}], 30)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(eval_rows)
    return {"name": name, "valid": valid, "cat": cat_ok, "exact": exact, "n": n,
            "general": general}


print(f"Evaluating {len(MODELS)} models on {len(eval_rows)} eval cases...\n")
results = [score(name, path) for name, path in MODELS]

print(f"{'model':<11}{'valid':>7}{'category':>10}{'exact':>7}   general-prompt answer")
print("-" * 80)
for r in results:
    g = (r["general"][:34] + "...") if len(r["general"]) > 34 else r["general"]
    print(f"{r['name']:<11}{r['valid']:>5}/{r['n']}{r['cat']:>7}/{r['n']}"
          f"{r['exact']:>5}/{r['n']}   {g!r}")
print(f"\nGeneral prompt was: {GENERAL_PROMPTS[0]!r}")
