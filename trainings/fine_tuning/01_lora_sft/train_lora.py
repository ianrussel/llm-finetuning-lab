"""
Quick LoRA / QLoRA fine-tuning for a causal language model.

Install (GPU):
    pip install -U transformers peft datasets trl accelerate bitsandbytes
Install (CPU / Apple Silicon — skip bitsandbytes):
    pip install -U transformers peft datasets trl accelerate

Run:
    python train_lora.py

Load the trained adapter afterwards:
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained("./lora-out")
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# ----------------------------- Config -----------------------------
MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"   # any HF causal LM (swap for Llama-3, Mistral, etc.)
DATASET     = "mlabonne/guanaco-llama2-1k"   # demo set with a ready-made "text" column
TEXT_FIELD  = "text"                          # column holding the training string
OUTPUT_DIR  = "./lora-out"
USE_4BIT    = torch.cuda.is_available()       # QLoRA on a CUDA GPU; plain LoRA otherwise

MAX_SEQ_LEN = 512                            # lowered for 4 GB GPUs (raise to 1024+ if you have headroom)
EPOCHS      = 1
LR          = 2e-4
BATCH_SIZE  = 1                              # 4 GB VRAM: keep batch tiny...
GRAD_ACCUM  = 8                              # ...and recover effective batch size here (1 x 8 = 8)
# ------------------------------------------------------------------

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
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_cfg,
    torch_dtype=torch.bfloat16,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.config.use_cache = False  # required when gradient checkpointing is on

# --- LoRA config ---
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",  # auto-target every linear layer; or list e.g. ["q_proj", "v_proj"]
)

# --- data ---
dataset = load_dataset(DATASET, split="train")
# To use your own data instead, comment the line above and use:
#   dataset = load_dataset("json", data_files="data.jsonl", split="train")
# where each line looks like: {"text": "....your fully formatted training example...."}

# --- trainer ---
sft_cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_available(),   # set bf16=False, fp16=True on older GPUs (T4 / V100)
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field=TEXT_FIELD,
    #max_seq_length=MAX_SEQ_LEN,
    max_length=MAX_SEQ_LEN,
    packing=True,                     # pack short samples together for throughput
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=dataset,
    peft_config=lora_cfg,
    processing_class=tokenizer,       # newer TRL; on older versions use tokenizer=tokenizer
)

trainer.train()
trainer.save_model(OUTPUT_DIR)        # saves the LoRA adapter (small, not the full model)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Adapter saved to {OUTPUT_DIR}")
