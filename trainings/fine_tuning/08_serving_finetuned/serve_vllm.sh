#!/usr/bin/env bash
# Module 8: serve one base model with multiple LoRA adapters via vLLM's OpenAI server.
#
# TEMPLATE: needs a CUDA GPU and `pip install vllm`. This is the multi-LoRA demo, ONE
# base model in GPU memory, several small adapters loaded on top, each selectable per
# request by name. Clients pick the specialist via the OpenAI `model` field (see client.py).
#
# Each --lora-modules entry is name=path. Paths point at the adapters this repo produced.
set -euo pipefail

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules \
    ticket-classifier=../03_synthetic_data/lora-out \
    guanaco-style=../01_lora_sft/lora-out \
  --port 8000

# OpenAI-compatible API is now at http://localhost:8000/v1
# Available "models": Qwen/Qwen2.5-0.5B-Instruct (base), ticket-classifier, guanaco-style.
