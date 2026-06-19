"""
Module 5, step 1: turn a LoRA adapter into a standalone fine-tuned model.

MergeKit merges full model weights, not LoRA adapters. Our fine-tunes from modules 1-3
are adapters, so first we bake one into the base with peft's merge_and_unload(), which
folds the adapter math into the base weights and gives a normal full model. That full
model is what we then merge back with the base in merge.py.

Default adapter is module 3's (the synthetic-data classifier). Run from this folder:
    aipy prepare_finetuned.py
"""

import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "../03_synthetic_data/lora-out"   # swap to 02_.../lora-out if you prefer
OUT_DIR = "./finetuned-full"

if not os.path.isdir(ADAPTER):
    raise SystemExit(f"Adapter not found at {ADAPTER}. Train module 3 first, or point "
                     f"ADAPTER at another lora-out folder.")

# float32 on CPU avoids half-precision matmul issues during the merge; MergeKit will
# cast to float16 later anyway.
tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, ADAPTER)

print("Folding the adapter into the base weights...")
merged = model.merge_and_unload()      # base + adapter -> plain full model
merged.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"Done. Standalone fine-tuned model saved to {OUT_DIR}. Next: aipy merge.py")
