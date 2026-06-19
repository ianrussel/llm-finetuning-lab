"""Interactively TEST the trained banking77 intent classifier (the final output).

Loads the base model + a trained LoRA adapter once, then lets you type customer messages
and see the predicted intent. Type a message at the prompt; blank line or Ctrl-D to quit.

Run from the track_a_banking77 folder, after training an adapter:
    ../../../.venv/bin/python run_trained.py                       # interactive
    ../../../.venv/bin/python run_trained.py --adapter phase3_train/lora-seed
    ../../../.venv/bin/python run_trained.py --message "I lost my card"   # one-shot
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="phase3_train/lora-seed-synth",
                    help="trained LoRA adapter ('' for the bare base model)")
    ap.add_argument("--message", default=None, help="ask one message and exit (non-interactive)")
    args = ap.parse_args()

    labels = common.load_labels()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=pick_dtype())
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def classify(message):
        msgs = [{"role": "system", "content": common.system_prompt(labels)},
                {"role": "user", "content": message}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return raw, common.predict_label(raw, labels)

    print(f"adapter: {args.adapter or '(base model only)'} | {len(labels)} intents loaded")
    if args.message is not None:                       # one-shot
        raw, intent = classify(args.message)
        print(f"message: {args.message}\nraw    : {raw!r}\nintent : {intent}")
        return

    print("Type a customer message to classify. Blank line or Ctrl-D to quit.")
    while True:
        try:
            message = input("\nmessage> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not message:
            break
        raw, intent = classify(message)
        print(f"  intent: {intent}    (raw: {raw!r})")


if __name__ == "__main__":
    main()
