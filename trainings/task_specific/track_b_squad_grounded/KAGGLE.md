# Training Track B on a free Kaggle GPU

The hybrid plan: generate the data locally (Phase 2, needs Ollama), then train on a
free Kaggle T4 (16 GB) where the long RAFT contexts are no longer cramped. Only two
small artifacts cross the boundary: the training JSONL goes up, the LoRA adapter
comes back down.

## 0. Get the data onto Kaggle (once)

Locally you already have `data/seed.jsonl`, `data/train_synth.jsonl` and
`data/gold.jsonl`. Make a Kaggle Dataset from them:

1. kaggle.com -> Datasets -> New Dataset.
2. Upload `seed.jsonl`, `train_synth.jsonl`, `gold.jsonl` (and `common.py` if you
   want the eval helpers later).
3. Name it `track-b-data`. It will mount at `/kaggle/input/track-b-data/`.

Then: New Notebook -> Settings -> Accelerator -> **GPU T4 x2** (or P100), and add
the `track-b-data` dataset to the notebook.

## 1. Install deps

```python
!pip -q install -U "transformers>=4.44" "trl>=0.9" peft bitsandbytes datasets accelerate
```

## 2. Config

```python
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # same base the local evaluator scores
DATA_DIR   = "/kaggle/input/track-b-data"
OUT_DIR    = "/kaggle/working"
MAX_LEN    = 1536        # RAFT context fits comfortably on 16 GB
BATCH      = 8           # a T4 handles this at 0.5B; effective batch = BATCH*GRAD_ACCUM
GRAD_ACCUM = 2
EPOCHS     = 3
LR         = 2e-4
```

## 3. Train one adapter (mirrors phase3_train/train.py)

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

def train_one(data_file, name):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map="auto")
    model.config.use_cache = False

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    ds = load_dataset("json", data_files=f"{DATA_DIR}/{data_file}", split="train")

    out = f"{OUT_DIR}/lora-{name}"
    cfg = SFTConfig(
        output_dir=out, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, save_strategy="epoch", bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_LEN, packing=False, assistant_only_loss=True,
        seed=0, report_to="none")

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora,
                         processing_class=tok)
    res = trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"{name}: final_train_loss={res.training_loss:.4f} -> {out}")
    return out
```

## 4. Run both arms of the comparison

```python
train_one("seed.jsonl", "seed")              # control
train_one("train_synth.jsonl", "seed-synth") # real run
```

## 5. Download the adapters

```python
import shutil
for name in ["seed", "seed-synth"]:
    shutil.make_archive(f"{OUT_DIR}/lora-{name}", "zip", f"{OUT_DIR}/lora-{name}")
print("download lora-seed.zip / lora-seed-synth.zip from the Output tab")
```

Bring those zips back, unzip into `phase3_train/lora-<name>/`, and run the local
evaluator exactly as before:

```
../../../.venv/bin/python eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth
```

## Notes

- Eval and generation stay local; only training moves here, so the numbers remain
  comparable (same base model, same recipe).
- With 16 GB you can push harder for a stronger result: raise `BATCH`, or swap
  `BASE_MODEL` to `Qwen/Qwen2.5-1.5B-Instruct` or `-3B-Instruct`. If you change the
  base, re-measure the base baseline on the same machine so the before/after stays
  honest.
- Kaggle sessions are time-limited and storage is ephemeral; persist anything you
  care about to the Dataset or download it before the session ends.
