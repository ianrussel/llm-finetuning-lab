# Module 3: Synthetic data

Goal: use an LLM to paraphrase a handful of examples into a bigger set (more variety,
not new facts), then quality-filter, and see if the bigger set beats module 2.

This reuses module 2's task (support ticket -> strict JSON {category, priority}) and
the same 6 held-out eval cases, so the scores are directly comparable. Module 2 trained
on 24 hand-written seeds. Here I grow those 24 into 100+ by paraphrasing, then retrain.

## Why a local model (not a cloud API)

The generator runs an instruct model on my own machine (served by Ollama), so the seeds
only ever go to localhost, never to a cloud API. In a real data-protection-sensitive
setting (PII, PHI, confidential data) that matters: a cloud API would send the seeds to
a third party, which regulation or contracts may forbid. Here the seeds are toy data, so
this is just practicing the privacy-preserving workflow. The trade-off is lower quality
than a frontier API.

## Setup (once)

The generator talks to a local Ollama server. Pull a model first:

```bash
ollama pull qwen2.5:3b-instruct      # ~2 GB, fits a 4 GB GPU
# better paraphrases if you have the room:
# ollama pull qwen2.5:7b-instruct
```

## Pipeline

1. paraphrase.py: for each seed, ask the local Ollama model for 5 reworded versions that
   keep the same meaning and the same label. Sends only to localhost. Writes
   data/generated.jsonl. (Set GEN_MODEL in the script to whatever you pulled.)
2. filter.py: clean the paraphrases. Drop empties, bad labels, exact and near-duplicates,
   and anything that collides with the eval set (no leakage). Writes data/train_synth.jsonl
   = seeds + kept paraphrases.
3. train.py: LoRA fine-tune on data/train_synth.jsonl (5 epochs). Writes ./lora-out.
4. test_base.py / test_adapter.py: score before/after on data/eval.jsonl
   (valid JSON, category, exact).

## Files

- data/seeds.jsonl: the 24 module 2 examples, the starting point.
- data/eval.jsonl: the 6 held-out cases (same as module 2).
- data/generated.jsonl: raw paraphrases (created by paraphrase.py).
- data/train_synth.jsonl: cleaned seeds + paraphrases (created by filter.py).
- common.py: shared task constants and helpers (SYSTEM prompt, label format, dedup).
- paraphrase.py, filter.py, train.py, test_base.py, test_adapter.py.

## How to run

Make sure Ollama is running and the model is pulled (see Setup). From this folder:

```bash
aipy paraphrase.py     # seeds -> data/generated.jsonl (calls local Ollama)
aipy filter.py         # -> data/train_synth.jsonl (read the dropped counts)
aipy train.py          # -> ./lora-out
aipy test_base.py      # before
aipy test_adapter.py   # after
```

paraphrase.py checks Ollama is up and the model is pulled before starting, and prints a
fix-it message if not. Ollama manages GPU/CPU offload, so the generator runs even on a
4 GB card. For better paraphrases, pull and switch GEN_MODEL to qwen2.5:7b-instruct.

## Sharing one small GPU between the generator and the trainer

On a 4 GB GPU the generator and the trainer cannot both sit in VRAM at once. If Ollama
is still holding the paraphrase model when train.py starts, training dies with a CUDA
out-of-memory error. Two ways this is handled:

- paraphrase.py sends keep_alive: 0, so Ollama unloads the model from VRAM right after
  the last paraphrase call. This is automatic, no action needed.
- If a model is still loaded for any reason (a manual run, a different keep_alive),
  unload it by hand before training:

  ```bash
  ollama stop qwen2.5:3b-instruct     # or whatever GEN_MODEL is
  nvidia-smi                          # confirm VRAM is freed
  ```

  `ollama ps` shows what Ollama currently has loaded.

## What to look for

- After filter.py, eyeball data/train_synth.jsonl. Are the paraphrases natural, varied,
  and still correctly labeled? Bad paraphrases are the main failure mode of SDG, garbage
  in, garbage out.
- Compare module 3's adapter score to module 2's adapter (category 6/6, exact 4/6). The
  hope is the extra variety helps the model generalize, especially on the priority calls
  module 2 missed.
- If the synthetic set does not help, the usual cause is paraphrases that are too similar
  to each other (no real variety) or mislabeled. Tighten filter.py or the paraphrase
  prompt.

## Make it my own

- Change N_PER_SEED in paraphrase.py to generate more or fewer per seed.
- Tighten or loosen SIM_THRESHOLD in filter.py to control how aggressive dedup is.
- Add a judge step: a second LLM pass that scores each paraphrase for "same meaning and
  label as the seed" and drops the low-scoring ones.
