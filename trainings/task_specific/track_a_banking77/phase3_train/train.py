"""Phase 3: train the LoRA/QLoRA specialist on the constructed set.

The mechanics are the familiar SFT-with-LoRA recipe from modules 1-3; what
matters in this phase is applying it cleanly to the data built in phases 1-2.
Two disciplines from the PDF are baked in here:

  1. Assistant-only loss. We train on the label tokens only, never on the long
     system prompt + user query. This is the fix for the "free-completion bug"
     from module 1 (where the model was trained on the whole text and learned to
     parrot the prompt). TRL masks everything except the assistant turn because
     the Qwen2.5 chat template marks it with a {% generation %} block. For a
     classifier whose target is a 1-5 token label, this is the difference between
     learning the task and drowning the signal in 400+ prompt tokens per row.

  2. No silent truncation of the label. The label is the LAST thing in each row,
     so a max_length set too low would cut exactly the token we are trying to
     learn. MAX_LEN is chosen above the longest row (max ~637 tokens); the script
     still counts and prints any row that would be truncated so it never happens
     unnoticed.

Base model is Qwen2.5-0.5B-Instruct, the same base the evaluator scores, and it
already emits the chat format, so no template surgery is needed. QLoRA (4-bit)
kicks in automatically on CUDA; plain LoRA on CPU.

Reads  : data/train_synth.jsonl (default) or any messages-format JSONL
Writes : phase3_train/lora-<name>/  (adapter + tokenizer + train_config.json)

Run from the track_a_banking77 folder. The planned three-condition comparison:

    # seed-only fine-tune (the honest "what does synthetic data add?" control)
    ../../../.venv/bin/python phase3_train/train.py --data data/seed.jsonl --name seed
    # seed + synthetic fine-tune (the real run)
    ../../../.venv/bin/python phase3_train/train.py --data data/train_synth.jsonl --name seed-synth

Then re-measure each against the recorded base baseline (Phase 4):

    ../../../.venv/bin/python eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# common.py lives one level up; reuse the exact same contract/helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import common

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # same base the evaluator scores


def pick_dtype():
    """bf16 on Ampere+, fp16 on older CUDA, fp32 on CPU (the module-3 rule)."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def count_truncated(tok, rows, max_len):
    """How many rows are longer than max_len, i.e. would lose their trailing
    label. We want this to be zero before training a classifier."""
    over = 0
    for r in rows:
        text = tok.apply_chat_template(r["messages"], tokenize=False,
                                       add_generation_prompt=False)
        if len(tok(text, add_special_tokens=False)["input_ids"]) > max_len:
            over += 1
    return over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{common.DATA}/train_synth.jsonl",
                    help="training JSONL in the messages format")
    ap.add_argument("--name", default="seed-synth",
                    help="label for the output adapter dir (lora-<name>)")
    ap.add_argument("--out", default=None, help="override the output dir")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=768,
                    help="token cap; above the longest row so the label is never cut")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="train on first N rows (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Resolve --data so the script works whether you run it from the track root
    # or from phase3_train/: absolute paths as-is, otherwise try the current dir
    # first, then fall back to the track root (where data/ actually lives).
    data_path = args.data
    if not os.path.isabs(data_path) and not os.path.exists(data_path):
        cand = os.path.join(os.path.dirname(_HERE), data_path)
        if os.path.exists(cand):
            data_path = cand
    if not os.path.exists(data_path):
        raise SystemExit(f"--data not found: {args.data} (looked in cwd and the "
                         f"track root {os.path.dirname(_HERE)})")

    out_dir = args.out or os.path.join(_HERE, f"lora-{args.name}")
    on_gpu = torch.cuda.is_available()
    dtype = pick_dtype()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-flight: load, optionally slice, and confirm nothing truncates.
    rows = common.read_jsonl(data_path)
    if args.limit:
        rows = rows[:args.limit]
    over = count_truncated(tokenizer, rows, args.max_len)
    print(f"data: {data_path}  rows={len(rows)}  max_len={args.max_len}  "
          f"rows_over_max_len={over}")
    if over:
        print(f"  WARNING: {over} rows exceed max_len and would lose their trailing "
              f"label. Raise --max-len.")

    quant_cfg = None
    if on_gpu:  # QLoRA: 4-bit base, same recipe as module 3
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_cfg,
        torch_dtype=dtype,
        device_map="auto" if on_gpu else None,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
    )

    dataset = load_dataset("json", data_files=data_path, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    sft_cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=on_gpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        packing=False,                 # packing would cross example boundaries
        assistant_only_loss=True,      # train on the label tokens only
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    result = trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Reproducibility: dump the exact recipe next to the adapter (PDF section 7).
    cfg = {
        "base_model": BASE_MODEL, "data": data_path, "name": args.name,
        "rows": len(rows), "epochs": args.epochs, "lr": args.lr,
        "batch": args.batch, "grad_accum": args.grad_accum, "max_len": args.max_len,
        "lora_rank": args.rank, "lora_alpha": args.alpha, "lora_dropout": args.dropout,
        "assistant_only_loss": True, "qlora_4bit": on_gpu, "dtype": str(dtype),
        "seed": args.seed, "final_train_loss": result.training_loss,
    }
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nDone. final_train_loss={result.training_loss:.4f}")
    print(f"Adapter + config saved to {out_dir}")
    print(f"Next: ../../../.venv/bin/python eval/evaluate.py --name {args.name} "
          f"--adapter {os.path.relpath(out_dir, os.path.dirname(_HERE))}")


if __name__ == "__main__":
    main()
