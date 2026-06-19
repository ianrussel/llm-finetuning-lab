# Module 1: LoRA/QLoRA SFT

Goal: fine-tune a small Qwen model on a small instruction dataset, start to finish.
Read the loss curve. Test before and after.

## Files
- train_lora.py: the fine-tune. QLoRA (4-bit) on a CUDA GPU, plain LoRA otherwise.
  Defaults to Qwen/Qwen2.5-0.5B-Instruct on the guanaco-llama2-1k demo set, 1 epoch.
  Writes the adapter to ./lora-out.
- test_base.py: run the untuned base model. This is the "before".
- test_adapter.py: load base + the trained adapter from ./lora-out. This is the "after".
- lora-out/: the trained LoRA adapter (small, adapter only, not the full model).

## How I run it
Run everything from inside this folder so the ./lora-out paths match up.

```bash
cd 01_lora_sft

# 1. before: see how the raw base model answers
aipy test_base.py

# 2. train
aipy train_lora.py

# 3. after: same kind of prompt, now through the tuned adapter
aipy test_adapter.py
```

Then read the loss (printed every 10 steps): trending down, flat, or overfitting?
Write a concrete before/after difference in ../PROGRESS.md.

## Notes to self
- The milestone says to use Unsloth. This script uses trl + peft + bitsandbytes
  instead, same LoRA/QLoRA idea. Trying the Unsloth version later (faster, less VRAM)
  is worth doing.
- On Kaggle/Colab T4 GPUs, switch bf16 to fp16 (T4 is bad at bf16):
  bnb_4bit_compute_dtype=torch.float16, model torch_dtype=torch.float16, and in
  SFTConfig set bf16=False, fp16=True.
- Out of memory? Lower MAX_SEQ_LEN or raise GRAD_ACCUM while keeping BATCH_SIZE=1.
