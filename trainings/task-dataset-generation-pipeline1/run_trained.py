"""Interactively TEST the trained help-desk resolution specialist (the final output).

Loads the base model + a trained LoRA adapter once, then predicts a ticket's resolution
(Done vs Won't Do) with a <think> reasoning trace. At the prompt, enter a ticket id (it
pulls the ticket's linked context via link + serialize), or press Enter to run the next
held-out gold ticket. Type 'q' or Ctrl-D to quit.

Run from this folder, after training an adapter:
    ../../.venv/bin/python run_trained.py                       # interactive
    ../../.venv/bin/python run_trained.py --adapter phase3_train/lora-seed
    ../../.venv/bin/python run_trained.py --issue-id 1004364    # one-shot
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_c as common

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="phase3_train/lora-seed-synth",
                    help="trained LoRA adapter ('' for the bare base model)")
    ap.add_argument("--issue-id", default=None, help="predict one ticket and exit (non-interactive)")
    args = ap.parse_args()

    labels = common.load_labels()
    gold = common.read_jsonl(f"{common.DATA}/gold.jsonl")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=pick_dtype())
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    _db = {"db": None, "survey": None}

    def from_issue(issue_id):
        import link
        import serialize
        if _db["db"] is None:
            _db["survey"] = serialize.require_confirmed_survey()
            _db["db"] = link.HelpDeskDB()
        linked = _db["db"].get_issue(issue_id)
        if linked["issue"] is None:
            return None, None
        ctx = serialize.serialize_issue(linked, _db["survey"])
        msgs = [{"role": "system", "content": common.system_prompt(labels)},
                {"role": "user", "content": common.user_content(ctx)}]
        return msgs, (linked["issue"].get("issue_resolution") or "").strip()

    def predict(msgs):
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=384, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return out, common.c_predict_label(out, labels)

    def show(msgs, true_label, src):
        out, pred = predict(msgs)
        print(f"\nsource   : {src}")
        print(f"--- model output ---\n{out}")
        print(f"predicted: {pred}    true: {true_label}")

    print(f"adapter: {args.adapter or '(base model only)'}")
    if args.issue_id is not None:                        # one-shot
        msgs, true_label = from_issue(args.issue_id)
        if msgs is None:
            print(f"no ticket with id {args.issue_id}")
        else:
            show(msgs, true_label, f"issue {args.issue_id}")
        return

    print("Enter a ticket id, or press Enter for the next gold ticket. 'q' or Ctrl-D to quit.")
    gi = 0
    while True:
        try:
            s = input("\nticket id (blank = next gold)> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if s.lower() == "q":
            break
        if s:
            msgs, true_label = from_issue(s)
            if msgs is None:
                print(f"  no ticket with id {s}")
                continue
            show(msgs, true_label, f"issue {s}")
        else:
            row = gold[gi % len(gold)]
            gi += 1
            show(row["messages"], row.get("label"), f"gold ticket (id {row.get('id')})")


if __name__ == "__main__":
    main()
