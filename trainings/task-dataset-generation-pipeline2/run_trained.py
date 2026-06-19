"""Interactively TEST the adapter the Part 2 gate accepted (the final output).

Loads the base model + the accepted adapter (from runs/report.json, or pass --adapter),
then predicts a held-out gold ticket's resolution with a <think> reasoning trace. At the
prompt, enter a gold index, or press Enter for the next one. Type 'q' or Ctrl-D to quit.

Run from this folder, after pipeline.py has accepted an adapter:
    ../../.venv/bin/python run_trained.py                       # interactive
    ../../.venv/bin/python run_trained.py --adapter runs/helpdesk-resolution-replay
    ../../.venv/bin/python run_trained.py --index 2             # one-shot
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p2 as c


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def accepted_adapter(cfg):
    path = os.path.join(c.out_dir(cfg), "report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("accepted_adapter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="adapter path; defaults to the gate-accepted one in runs/report.json")
    ap.add_argument("--index", type=int, default=None, help="run one gold index and exit")
    args = ap.parse_args()

    cfg = c.load_config()
    adapter = args.adapter or accepted_adapter(cfg)
    if not adapter:
        raise SystemExit("no adapter: run pipeline.py first (so the gate records an accepted "
                         "adapter), or pass --adapter <path>.")

    labels = c.load_labels(c.data_path(cfg, "labels"))
    gold = c.read_jsonl(c.data_path(cfg, "gold"))

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=pick_dtype())
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def show(row):
        text = tok.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=384, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\ngold id  : {row.get('id')}")
        print(f"--- model output ---\n{out}")
        print(f"predicted: {c.c_predict_label(out, labels)}    true: {row.get('label')}")

    print(f"adapter: {adapter}")
    if args.index is not None:                          # one-shot
        show(gold[args.index % len(gold)])
        return

    print(f"{len(gold)} gold tickets. Enter an index, or press Enter for the next. "
          "'q' or Ctrl-D to quit.")
    gi = 0
    while True:
        try:
            s = input("\ngold index (blank = next)> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if s.lower() == "q":
            break
        idx = int(s) if s.isdigit() else gi
        gi = idx + 1
        show(gold[idx % len(gold)])


if __name__ == "__main__":
    main()
