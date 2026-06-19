"""
Module 6: before/after on the held-out images.

Scores the base VLM and the LoRA-tuned VLM on data/eval.jsonl. An answer counts as
correct if the expected word (the color or shape) appears in the model's output.

Run from this folder:
    aipy evaluate.py
"""

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from common import MODEL_ID, ADAPTER_DIR, read_jsonl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.image_processor.do_image_splitting = False   # match training
rows = read_jsonl("data/eval.jsonl")


def load(adapter=None):
    m = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.float32).to(DEVICE)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    return m.eval()


def answer(model, img, question):
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def score(model):
    contains = 0   # lenient: the right word appears anywhere in the output
    exact = 0      # strict: the output IS just the word (the trained terse format)
    outs = []
    for r in rows:
        img = Image.open(f"data/{r['image']}").convert("RGB")
        got = answer(model, img, r["question"])
        want = r["answer"].lower()
        norm = got.lower().strip().strip(".").strip()
        contains += want in got.lower()
        exact += norm == want
        outs.append(got)
    return contains, exact, outs


print("Scoring base model...")
base = load()
base_contains, base_exact, base_outs = score(base)
del base
if torch.cuda.is_available():
    torch.cuda.empty_cache()

adapter_outs = None
adapter_contains = adapter_exact = None
if os.path.isdir(ADAPTER_DIR):
    print("Scoring fine-tuned model...")
    adapter_contains, adapter_exact, adapter_outs = score(load(ADAPTER_DIR))

n = len(rows)
print(f"\n{'want':<10}{'base':<22}{'adapter':<14}{'question'}")
print("-" * 78)
for i, r in enumerate(rows):
    a = adapter_outs[i] if adapter_outs else "-"
    print(f"{r['answer']:<10}{base_outs[i][:20]:<22}{a[:12]:<14}{r['question']}")

print(f"\n{'':12}{'contains (lenient)':<22}{'exact word (strict)'}")
print(f"{'base':<12}{f'{base_contains}/{n}':<22}{base_exact}/{n}")
if adapter_exact is not None:
    print(f"{'fine-tuned':<12}{f'{adapter_contains}/{n}':<22}{adapter_exact}/{n}")
else:
    print("(no adapter yet; run train.py to compare)")
