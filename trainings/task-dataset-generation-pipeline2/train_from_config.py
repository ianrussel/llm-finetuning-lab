"""Stage 3.2 (Train, config-driven): run a LoRA/QLoRA training defined entirely by the
config. The config is the single source of truth; this file just executes it.

Importable: run_training(cfg, run_name, data_path, lr=None, epochs=None) -> adapter_dir,
so pipeline.py can drive training and the adjust-on-fail re-run. Also runnable standalone
to train one variant.

Run from this folder (GPU):
    ../../.venv/bin/python train_from_config.py --variant replay
    ../../.venv/bin/python train_from_config.py --variant noreplay   # guardrail control
"""

import argparse
import json
import os

# Long context + reasoning trace fragments VRAM on a small GPU; opt into expandable
# segments before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import common_p2 as c


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


def run_training(cfg, run_name, data_path, lr=None, epochs=None):
    """Train one adapter from the config. Returns the adapter directory.

    `epochs` is an upper bound on the budget. If `train.early_stopping.enabled` is set, the run
    evaluates both axes after each epoch, keeps the best adapter, and stops as soon as a
    regression axis starts to slip or the task metric stops improving (see early_stopping.py)."""
    base = cfg["base_model"]
    lora = cfg["lora"]
    tr = cfg["train"]
    lr = tr["learning_rate"] if lr is None else lr
    epochs = tr["epochs"] if epochs is None else epochs
    max_len = tr["max_seq_len"]
    es = tr.get("early_stopping", {})
    es_enabled = bool(es.get("enabled", False))
    out = os.path.join(c.out_dir(cfg), run_name)
    on_gpu = torch.cuda.is_available()
    dtype = pick_dtype()

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = c.read_jsonl(data_path)
    over = count_truncated(tok, rows, max_len)
    print(f"[train] {run_name}: data={os.path.basename(data_path)} rows={len(rows)} "
          f"epochs={epochs} lr={lr} max_len={max_len} rows_over_max_len={over}")
    if over:
        print(f"  WARNING {over} rows exceed max_len and would lose the trace/label.")

    quant = None
    if on_gpu and tr.get("use_qlora", True):
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=quant, torch_dtype=dtype,
        device_map="auto" if on_gpu else None)
    model.config.use_cache = False

    lora_cfg = LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=lora["target_modules"])

    dataset = load_dataset("json", data_files=data_path, split="train")
    # When early stopping owns checkpoint selection it saves the best adapter into `out` itself,
    # so the trainer does not need its own per-epoch checkpoints.
    sft = SFTConfig(
        output_dir=out, num_train_epochs=epochs,
        per_device_train_batch_size=tr["batch_size"], gradient_accumulation_steps=tr["grad_accum"],
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=5, save_strategy=("no" if es_enabled else "epoch"),
        bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
        gradient_checkpointing=on_gpu, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_len, packing=False, assistant_only_loss=True, seed=0, report_to="none")

    stopper = None
    callbacks = []
    if es_enabled:
        from early_stopping import TwoAxisEarlyStopping
        stopper = TwoAxisEarlyStopping(cfg, tok, out, es)
        callbacks = [stopper]
        print(f"[train] {run_name}: adaptive length on (max epochs={epochs}, "
              f"patience={stopper.patience}, reg_tol={stopper.tol}, min_task_delta={stopper.min_delta})")

    trainer = SFTTrainer(model=model, args=sft, train_dataset=dataset,
                         peft_config=lora_cfg, processing_class=tok, callbacks=callbacks)
    result = trainer.train()

    # Keep the best adapter early stopping already wrote to `out`; otherwise save the final model.
    kept_best = stopper is not None and stopper.best_epoch is not None
    if not kept_best:
        trainer.save_model(out)
        tok.save_pretrained(out)

    run_meta = {"run_name": run_name, "base_model": base, "data": data_path,
                "max_epochs": epochs, "lr": lr, "max_len": max_len, "lora": lora,
                "final_train_loss": result.training_loss,
                "early_stopping": es_enabled}
    if stopper is not None:
        run_meta.update({"best_epoch": stopper.best_epoch, "stop_reason": stopper.stop_reason,
                         "epoch_history": stopper.history})
    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    tail = (f" best_epoch={stopper.best_epoch}/{epochs}"
            f" stop={stopper.stop_reason or 'reached max epochs'}") if stopper else ""
    print(f"[train] {run_name}: final_train_loss={result.training_loss:.4f}{tail} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["replay", "noreplay"], default="replay",
                    help="replay -> train_mix (task+rehearsal); noreplay -> train_synth (task only)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    args = ap.parse_args()

    cfg = c.load_config()
    if not cfg.get("confirmed"):
        raise SystemExit("config not confirmed. Review config.yaml, set confirmed: true.")
    key = "task_train" if args.variant == "replay" else "task_train_noreplay"
    data = c.data_path(cfg, key)
    run_name = f"{cfg['name']}-{args.variant}"
    run_training(cfg, run_name, data, lr=args.lr, epochs=args.epochs)


if __name__ == "__main__":
    main()
