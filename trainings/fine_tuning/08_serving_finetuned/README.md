# Module 8: Serving fine-tuned models

Goal: a lighter-touch module, understand how a trained model actually gets used in
production. Serving with an inference engine (vLLM), multi-adapter serving (many LoRA
adapters on one base model, switched per request), and exporting to GGUF for local
runtimes (Ollama, llama.cpp).

This is more reference than heavy hands-on: vLLM wants a real GPU, and the point is to
understand the moving parts, not to stand up a production cluster.

## The two questions, answered

### 1. What does an inference engine like vLLM do, and why use it over plain transformers?

Plain `transformers` (`model.generate`) is great for experiments and the before/after
tests in this repo, but it is a poor way to *serve* a model under load:

- it handles one or a few requests at a time, with no smart batching across users,
- it allocates the KV cache naively (reserve max length per sequence), which wastes GPU
  memory and limits how many requests fit at once,
- it leaves the GPU idle between steps.

An inference engine like vLLM is purpose-built for high-throughput, low-latency serving:

- PagedAttention: manages the KV cache like virtual memory (in fixed-size pages), so
  many sequences pack into GPU memory without fragmentation or over-allocation. This is
  the big one, it dramatically raises how many concurrent requests fit.
- Continuous (in-flight) batching: requests are batched token by token, sequences are
  added and removed as they arrive and finish, so the GPU stays saturated instead of
  waiting for a fixed batch to complete.
- Optimized CUDA kernels, tensor parallelism for big models, quantization support, token
  streaming, and an OpenAI-compatible HTTP server out of the box.

Net effect: far higher tokens/second and GPU utilization than looping `model.generate`,
which is what you need when many users hit the model at once. Rule of thumb: plain
transformers for training and experimentation; vLLM (or TGI, TensorRT-LLM, etc.) for
production serving.

### 2. How does multi-LoRA serving work, and why is it efficient for several specialists?

A LoRA adapter is a small set of low-rank weight deltas that sit on top of a frozen base
model. Crucially, many task-specific adapters all share the SAME base weights.

Multi-LoRA serving exploits that:

- Load ONE copy of the base model in GPU memory, plus many small adapters (each tens of
  MB, not GBs).
- Each request names which adapter it wants. The engine applies that adapter's delta for
  that request, and specialized kernels (vLLM multi-LoRA, S-LoRA / Punica) can serve a
  batch where different requests use different adapters at the same time.

Why it is efficient for hosting several specialists:

- Memory: you do not load N full models (N x several GB). You load 1 base + N tiny
  adapters. Ten specialists cost roughly one base model of VRAM, not ten.
- Density and cost: one container / one GPU serves many task specialists, switched per
  request. Much cheaper than a separate deployment per fine-tune.
- Flexibility: add or remove adapters without reloading the base, and route each request
  to the right specialist by name.

This is the standard efficient pattern for serving a fleet of fine-tunes: one base, many
adapters, picked per request.

## Three serving paths (from this repo's artifacts)

You already produced the artifacts these use: LoRA adapters in 01/02/03 (`lora-out/`) and
a merged full model in 05 (`finetuned-full/`).

### A. vLLM, single model (high-throughput GPU serving)

```bash
pip install vllm                     # needs a CUDA GPU; heavy install
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
# OpenAI-compatible API at http://localhost:8000/v1
```

### B. vLLM, multi-LoRA (several specialists on one base)

See serve_vllm.sh, it loads the module 1 and module 3 adapters on one base, and
client.py picks the adapter per request by the `model` field. This is the multi-LoRA
demo: same base in memory, different specialist per call.

### C. GGUF / Ollama (local, CPU-friendly serving)

GGUF is the quantized file format used by llama.cpp and Ollama, the opposite end from
vLLM: optimized for running locally on CPU or a small GPU, offline. To serve the module 5
merged classifier in Ollama, use the provided Modelfile:

```bash
ollama create ticket-classifier -f Modelfile
ollama run ticket-classifier "My invoice is higher than last month."
```

(Ollama imports the Qwen2 safetensors directly. To go through a real .gguf file instead,
convert with llama.cpp's convert_hf_to_gguf.py and optionally quantize to q4_K_M, then
point the Modelfile FROM at the .gguf.)

## Files

- serve_vllm.sh: launch vLLM's OpenAI server with two LoRA adapters (template; needs GPU + vllm).
- client.py: OpenAI-compatible client that selects an adapter per request (needs `pip install openai`).
- Modelfile: Ollama import of the module 5 merged model for local serving.

## When to use which

- vLLM (or TGI): production, many concurrent users, throughput matters, you have GPUs.
- vLLM multi-LoRA: you have several fine-tuned specialists of one base and want to serve
  them all cheaply from one container.
- GGUF + Ollama / llama.cpp: local, offline, CPU or small-GPU, single-user or edge.

## Make it my own

- Add the module 2 adapter to serve_vllm.sh as a third specialist and route between all three.
- Quantize for serving (vLLM supports AWQ/GPTQ; GGUF supports q4_K_M etc.) and compare
  speed/quality.
- Put a tiny router in front (pick the adapter by request type) to mimic a real
  multi-tenant setup.
