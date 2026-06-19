"""
Module 6: LoRA fine-tune a small VLM on the synthetic image-QA set.

The multimodal flow: an image goes through a vision encoder, the resulting visual
features are projected into the language model's token space, and the language model
generates the answer conditioned on both the image features and the question text.

The freeze-versus-train choice (the point of this module):
  - We add LoRA adapters ONLY to the language model's attention/MLP projections.
  - The vision encoder is left frozen (no adapters on its layers, base weights frozen).
  - Rationale: the pretrained vision encoder already sees shapes and colors fine; what
    we are teaching is how the language model answers about them. Freezing the vision
    tower is the cheap, stable default. Training it too costs much more memory and
    risks wrecking good visual features on a tiny dataset.

GPU recommended. Run from this folder (after make_data.py):
    aipy train.py
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

from common import MODEL_ID, ADAPTER_DIR, read_jsonl

ON_GPU   = torch.cuda.is_available()
USE_BF16 = ON_GPU and torch.cuda.is_bf16_supported()
DTYPE    = torch.bfloat16 if USE_BF16 else (torch.float16 if ON_GPU else torch.float32)

processor = AutoProcessor.from_pretrained(MODEL_ID)
# Our images are one simple shape each, so we do NOT tile them into sub-images. Turning
# off splitting cuts the image-token count (and so the lm_head memory) dramatically,
# which is what makes this fit a 4 GB GPU.
processor.image_processor.do_image_splitting = False

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=DTYPE, device_map="auto" if ON_GPU else None)

# LoRA on the language model only -> the vision encoder stays frozen.
# Important: the vision tower and the text model SHARE projection names (both have
# q_proj/k_proj/v_proj), so a plain name list would also adapt the vision encoder. We
# scope target_modules with a regex that matches only model.text_model.* layers, which
# is what makes this a true "freeze the vision, train the language model" run.
TEXT_LORA_TARGETS = (r"model\.text_model\.layers\.\d+\."
                     r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=TEXT_LORA_TARGETS,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# image placeholder token, masked out of the loss labels
try:
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")]
except Exception:
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")


def load_split(name):
    rows = read_jsonl(f"data/{name}.jsonl")
    for r in rows:
        r["pil"] = Image.open(f"data/{r['image']}").convert("RGB")
    return rows


train_rows = load_split("train")


def collate(examples):
    texts, images = [], []
    for ex in examples:
        msgs = [
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": ex["question"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]},
        ]
        texts.append(processor.apply_chat_template(msgs, add_generation_prompt=False).strip())
        images.append([ex["pil"]])
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100          # do not predict image tokens
    batch["labels"] = labels
    return batch


args = TrainingArguments(
    output_dir=ADAPTER_DIR,
    num_train_epochs=8,
    per_device_train_batch_size=1,    # 4 GB GPU; keep batch tiny
    gradient_accumulation_steps=8,    # recover effective batch size (1 x 8)
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy="no",
    bf16=USE_BF16,
    fp16=ON_GPU and not USE_BF16,
    remove_unused_columns=False,    # keep our custom columns for the collator
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=train_rows, data_collator=collate)
trainer.train()
model.save_pretrained(ADAPTER_DIR)
processor.save_pretrained(ADAPTER_DIR)
print(f"Done. Adapter saved to {ADAPTER_DIR}. Next: aipy evaluate.py")
