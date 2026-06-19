"""Phase 3: train the grounded-QA LoRA/QLoRA specialist on the constructed set.

Same SFT-with-LoRA recipe as Track A; the two disciplines that matter carry over:

  1. Assistant-only loss. Loss is taken on the answer tokens only, never on the
     long RAFT context (oracle + distractors) or the question. TRL masks every
     turn except the assistant one (the Qwen2.5 template marks it with a
     {% generation %} block). For a grounded answer that is a short span or the
     fixed abstention string, this keeps the signal from drowning in the context.
  2. No silent truncation. The context here is long (multiple passages), so a
     too-low max_length would cut the oracle passage and turn an answerable row
     into a misleading one. MAX_LEN is set high and the script counts and prints
     any row that would still overflow, so it never happens unnoticed.

Base model is Qwen2.5-0.5B-Instruct, the same base the evaluator scores. QLoRA
(4-bit) on CUDA, plain LoRA on CPU.

Reads  : data/train_synth.jsonl (default) or any messages-format JSONL
Writes : phase3_train/lora-<name>/  (adapter + tokenizer + train_config.json)

Run from the track_b_squad_grounded folder. The three-condition comparison:

    ../../../.venv/bin/python phase3_train/train.py --data data/seed.jsonl --name seed
    ../../../.venv/bin/python phase3_train/train.py --data data/train_synth.jsonl --name seed-synth

Then re-measure each against the recorded base baseline (Phase 4):

    ../../../.venv/bin/python eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth
"""

import argparse
import json
import os
import sys

# Track B's RAFT contexts are long, so on a small GPU the loss-step logits tensor
# (sequence x 152k vocab) fragments VRAM and OOMs. Opt into expandable segments
# before torch initializes CUDA; harmless on big GPUs and on CPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import common

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # same base the evaluator scores


def pick_dtype():
    """bf16 on Ampere+, fp16 on older CUDA, fp32 on CPU (the module-3 rule)."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def count_truncated(tok, rows, max_len):
    """How many rows are longer than max_len. For RAFT rows that means a passage
    (maybe the oracle) gets cut, so we want this at or near zero."""
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
    ap.add_argument("--batch", type=int, default=1,
                    help="per-device batch; 1 keeps long RAFT contexts inside a 4 GB GPU")
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="effective batch = batch x grad_accum (default 16)")
    ap.add_argument("--max-len", type=int, default=1536,
                    help="token cap; high enough that the RAFT context is not cut")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="train on first N rows (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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

    rows = common.read_jsonl(data_path)
    if args.limit:
        rows = rows[:args.limit]
    over = count_truncated(tokenizer, rows, args.max_len)
    print(f"data: {data_path}  rows={len(rows)}  max_len={args.max_len}  "
          f"rows_over_max_len={over}")
    if over:
        print(f"  WARNING: {over} rows exceed max_len and would lose context. "
              f"Raise --max-len.")

    quant_cfg = None
    if on_gpu:
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
    model.config.use_cache = False

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
        packing=False,
        assistant_only_loss=True,      # train on the answer tokens only
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
