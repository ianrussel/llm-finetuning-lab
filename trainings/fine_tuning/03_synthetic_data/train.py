"""
Module 3: train on the synthetic dataset (seeds + filtered paraphrases).

Same trainer as module 2, just pointed at data/train_synth.jsonl and with fewer
epochs because there are more examples now. Precision is device-aware: bf16 on
Ampere+ (no grad scaler), fp16 on older GPUs (T4), fp32 on CPU.

Run from this folder:
    aipy train.py
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "data/train_synth.jsonl"
OUTPUT_DIR = "./lora-out"

ON_GPU     = torch.cuda.is_available()
USE_4BIT   = ON_GPU
USE_BF16   = ON_GPU and torch.cuda.is_bf16_supported()
USE_FP16   = ON_GPU and not USE_BF16
if not ON_GPU:
    DTYPE = torch.float32
elif USE_BF16:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float16

EPOCHS     = 5            # more data than module 2, so fewer passes
LR         = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_LEN    = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

quant_cfg = None
if USE_4BIT:
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_cfg,
    torch_dtype=DTYPE,
    device_map="auto" if ON_GPU else None,
)
model.config.use_cache = False

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules="all-linear",
)

dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
print(f"Loaded {len(dataset)} training examples from {TRAIN_FILE}")

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
    bf16=USE_BF16,
    fp16=USE_FP16,
    gradient_checkpointing=ON_GPU,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=MAX_LEN,
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=dataset,
    peft_config=lora_cfg,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Adapter saved to {OUTPUT_DIR}")
