"""
Module 2: train a small model on a hand-built dataset.

Task: classify a one-line support message into strict JSON:
    {"category": one of [billing, technical, account, general],
     "priority": one of [low, medium, high]}

The data lives in data/train.jsonl, one JSON object per line in the
conversational "messages" format (system + user + assistant). TRL applies the
model's chat template for us, so we never format prompt strings by hand.

Defaults assume a CUDA GPU (Kaggle T4 etc.) using fp16. Run from this folder:
    ../.venv/bin/python train.py
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "data/train.jsonl"
OUTPUT_DIR = "./lora-out"

ON_GPU     = torch.cuda.is_available()
USE_4BIT   = ON_GPU                       # QLoRA on GPU, plain LoRA on CPU
# Pick precision by what the device supports. bf16 needs no gradient scaler, so it
# avoids the fp16-scaler-vs-bf16-grad crash on Ampere cards (e.g. RTX 3050). Older
# GPUs without bf16 (T4) fall back to fp16; CPU uses fp32.
USE_BF16   = ON_GPU and torch.cuda.is_bf16_supported()
USE_FP16   = ON_GPU and not USE_BF16
if not ON_GPU:
    DTYPE = torch.float32
elif USE_BF16:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float16

EPOCHS     = 8        # tiny dataset, so we make several passes over it
LR         = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_LEN    = 512

# --- tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- model (optionally 4-bit) ---
quant_cfg = None
if USE_4BIT:
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,           # match the training precision
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_cfg,
    torch_dtype=DTYPE,
    device_map="auto" if ON_GPU else None,
)
model.config.use_cache = False

# --- LoRA config ---
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules="all-linear",
)

# --- data ---
# Each line has a "messages" list; TRL detects this and applies the chat template.
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
print(f"Loaded {len(dataset)} training examples from {TRAIN_FILE}")

# --- trainer ---
sft_cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy="epoch",
    bf16=USE_BF16,                     # bf16 on Ampere+ (no grad scaler needed)
    fp16=USE_FP16,                     # fp16 on older GPUs without bf16 (T4)
    gradient_checkpointing=ON_GPU,     # saves VRAM on GPU; pure slowdown on CPU
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=MAX_LEN,
    packing=False,        # keep each example separate; the dataset is tiny
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=dataset,
    peft_config=lora_cfg,
    processing_class=tokenizer,   # newer TRL; older versions use tokenizer=tokenizer
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Adapter saved to {OUTPUT_DIR}")
