"""Shared bits for module 6 (the synthetic image-QA task)."""

import json

# A genuinely small VLM that fits a 4 GB GPU. On a bigger GPU (Kaggle T4) you can step
# up to HuggingFaceTB/SmolVLM-500M-Instruct or SmolVLM-Instruct for better quality.
MODEL_ID    = "HuggingFaceTB/SmolVLM-256M-Instruct"
ADAPTER_DIR = "./vlm-lora"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
