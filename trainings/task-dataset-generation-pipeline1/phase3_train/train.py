"""Phase 3: train the help-desk resolution specialist (LoRA/QLoRA) on the built set.

Same recipe as the earlier tracks. Two disciplines carry over:
  1. Assistant-only loss: loss falls only on the "<think> ... </think> The answer is X."
     turn, never on the long serialized ticket context, so the signal is the reasoning
     + label, not the prompt.
  2. No silent truncation: the serialized context plus a long reasoning trace is far
     longer than a bare label, so MAX_LEN is high and the script counts/prints any row
     that would still overflow.

Base is Qwen2.5-0.5B-Instruct (same base the evaluator scores). QLoRA 4-bit on CUDA.

Reads  : data/train_mix.jsonl (default) or any messages-format JSONL
Writes : phase3_train/lora-<name>/ (adapter + tokenizer + train_config.json)

Run from the track root. The control comparison:
    ../../.venv/bin/python phase3_train/train.py --data data/seed.jsonl --name seed
    ../../.venv/bin/python phase3_train/train.py --data data/train_mix.jsonl --name seed-synth
"""

import argparse
import json
import os
import sys

# Long serialized context + reasoning trace fragments VRAM on a small GPU; opt into
# expandable segments before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import common_c as common

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def count_truncated(tok, rows, max_len):
    over = 0
    for r in rows:
        text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        if len(tok(text, add_special_tokens=False)["input_ids"]) > max_len:
            over += 1
    return over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{common.DATA}/train_mix.jsonl")
    ap.add_argument("--name", default="seed-synth")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1,
                    help="per-device batch; 1 keeps long context+trace inside a 4 GB GPU")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1408,
                    help="token cap; high enough for serialized context + reasoning trace")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_path = args.data
    if not os.path.isabs(data_path) and not os.path.exists(data_path):
        cand = os.path.join(os.path.dirname(_HERE), data_path)
        if os.path.exists(cand):
            data_path = cand
    if not os.path.exists(data_path):
        raise SystemExit(f"--data not found: {args.data}")

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
    print(f"data: {data_path}  rows={len(rows)}  max_len={args.max_len}  rows_over_max_len={over}")
    if over:
        print(f"  WARNING: {over} rows exceed max_len and would lose the trace/label. Raise --max-len.")

    quant_cfg = None
    if on_gpu:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_cfg, torch_dtype=dtype,
        device_map="auto" if on_gpu else None)
    model.config.use_cache = False

    lora_cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                          bias="none", task_type="CAUSAL_LM", target_modules="all-linear")

    dataset = load_dataset("json", data_files=data_path, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    sft_cfg = SFTConfig(
        output_dir=out_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=5, save_strategy="epoch",
        bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
        gradient_checkpointing=on_gpu, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len, packing=False, assistant_only_loss=True,
        seed=args.seed, report_to="none")

    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=dataset,
                         peft_config=lora_cfg, processing_class=tokenizer)
    result = trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    cfg = {"base_model": BASE_MODEL, "data": data_path, "name": args.name, "rows": len(rows),
           "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "grad_accum": args.grad_accum,
           "max_len": args.max_len, "lora_rank": args.rank, "lora_alpha": args.alpha,
           "assistant_only_loss": True, "qlora_4bit": on_gpu, "dtype": str(dtype),
           "seed": args.seed, "final_train_loss": result.training_loss}
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nDone. final_train_loss={result.training_loss:.4f}")
    print(f"Adapter + config saved to {out_dir}")
    print(f"Next: ../../.venv/bin/python eval/evaluate.py --name {args.name} "
          f"--adapter {os.path.relpath(out_dir, os.path.dirname(_HERE))}")


if __name__ == "__main__":
    main()
