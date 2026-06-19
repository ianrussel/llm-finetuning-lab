"""Interactively TEST the trained grounded-QA specialist (the final output).

Loads the base model + a trained LoRA adapter once. You give it a context passage and
then ask questions about it; the model answers ONLY from that context and abstains
("not in the context") when the answer is not there. At the prompt, type a question;
type 'ctx: <new passage>' to change the context; blank line or Ctrl-D to quit.

Run from the track_b_squad_grounded folder, after training an adapter:
    ../../../.venv/bin/python run_trained.py                      # interactive
    ../../../.venv/bin/python run_trained.py --adapter phase3_train/lora-seed
    ../../../.venv/bin/python run_trained.py --context "..." --question "..."   # one-shot
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CONTEXT = ("The Great Barrier Reef is the world's largest coral reef system, off the "
                   "coast of Queensland, Australia. It stretches for over 2,300 kilometres.")


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="phase3_train/lora-seed-synth",
                    help="trained LoRA adapter ('' for the bare base model)")
    ap.add_argument("--context", default=None, help="context passage (defaults to a built-in one)")
    ap.add_argument("--question", default=None, help="ask one question and exit (non-interactive)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=pick_dtype())
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def ask(context, question):
        msgs = [{"role": "system", "content": common.SYSTEM_PROMPT},
                {"role": "user", "content": common.user_content([context], question)}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        ans = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return ans, common.is_abstention(ans)

    context = args.context or DEFAULT_CONTEXT
    print(f"adapter: {args.adapter or '(base model only)'}")
    if args.question is not None:                       # one-shot
        ans, abstained = ask(context, args.question)
        print(f"context : {context}\nquestion: {args.question}\nanswer  : {ans!r}  "
              f"(abstained={abstained})")
        return

    print(f"\ncontext: {context}")
    print("Ask a question about the context. 'ctx: <text>' changes the context. "
          "Blank line or Ctrl-D to quit.")
    while True:
        try:
            line = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            break
        if line.lower().startswith("ctx:"):
            context = line[4:].strip()
            print(f"  context set to: {context}")
            continue
        ans, abstained = ask(context, line)
        print(f"  answer: {ans!r}    (abstained={abstained})")


if __name__ == "__main__":
    main()
