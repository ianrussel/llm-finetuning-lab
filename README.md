# LLM Fine-Tuning Lab

A hands-on, self-paced curriculum for fine-tuning LLMs end to end: prompt engineering,
LoRA/QLoRA, synthetic dataset generation, MCP, and an automated training pipeline with a
two-axis quality gate. Each track pairs a short concept with runnable code, so you learn the
technique by watching it change the model's output, not by reading about it.

The throughline is a real engineering question: how do you take a model from generic to
genuinely better at a task **without losing its general ability**, and how do you prove it.
That is what the dataset pipelines and the two-axis gate are built to answer.

## What's inside

| Track | What it covers |
|-------|----------------|
| [prompt_engineering/](trainings/prompt_engineering/) | 8 modules, weak vs strong prompt side by side: clarity, few-shot, structured output, roles, chain-of-thought, grounding, evaluation, pitfalls. Runs on a local model via Ollama, no GPU. |
| [fine_tuning/](trainings/fine_tuning/) | 8 milestones: LoRA/QLoRA SFT, a dataset built from scratch, synthetic data, embedding fine-tune with hard-negative mining, model merging, a small multimodal (VLM) tune, a managed Vertex AI tune, and serving (vLLM multi-LoRA + GGUF/Ollama). |
| [mcp/](trainings/mcp/) | 5 modules on the Model Context Protocol: concepts, your first server, resources and prompts, connecting a server to Claude Code, and building a help-desk server. |
| [task-dataset-generation-pipeline1/](trainings/task-dataset-generation-pipeline1/) | Part 1: turn a linked knowledge base into a training set. Field survey and confirm, a verified seed and a sacred gold set, synthetic expansion (paraphrase, evolve, LLM-as-judge, decontaminate), and a replay/rehearsal mix. |
| [task-dataset-generation-pipeline2/](trainings/task-dataset-generation-pipeline2/) | Part 2: a config-driven training pipeline. Config in, trained-and-gated adapter out, with a two-axis acceptance gate (task gain vs no regression), adaptive training length, and one automatic adjusted re-run on failure. |
| [task_specific/](trainings/task_specific/) | Applied tracks that run the corpus-to-specialist flow on real datasets: banking77 intent classification (Track A) and SQuAD-grounded QA (Track B). |

## The idea worth a closer look: the two-axis gate

Fine-tuning a small model on a narrow task is easy to do badly: the task score goes up while
the model quietly forgets how to reason or call tools. The Part 2 pipeline accepts an adapter
only if it clears **both** axes at once:

- **task** improves by at least a set margin (macro-F1 on a held-out gold set), and
- **no regression**: a sentinel, a reasoning probe set, and a tool-calling probe set each stay
  within a small tolerance of the base model.

Training length is not hand-tuned either. A rule proposes a starting budget from the dataset
size, early stopping adapts the real length per task on the same signal the gate uses, and the
gate plus one adjusted re-run is the backstop when a run still misses. The write-up in
[pipeline2/WRITEUP.md](trainings/task-dataset-generation-pipeline2/WRITEUP.md) walks through a
real verdict.

## Run on Kaggle

The heavier training runs are set up to run on a free Kaggle T4. Live notebooks:

- **Dataset generation (Part 1):** [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ianrusseladem/notebook-datasetgenerationpipeline)
- **Pipeline 2 (the gated training pipeline):** [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ianrusseladem/notebook-pipeline2)
- **Reasoning-chain SFT (Part B):** [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ianrusseladem/notebook-reasoning-partb)
- More notebooks coming soon.

## Running it locally

Most of the prompt-engineering and MCP work runs without a GPU; the fine-tuning and dataset
pipelines need one (or use the Kaggle notebooks above). Setup and environment notes, including
the venv and GPU gotchas, are in [development.md](development.md).

One thing to know up front: the scripts expect the project virtual environment, so call it
explicitly (for example `../../.venv/bin/python train_from_config.py`) rather than a bare
`python`, which is often the wrong interpreter on this setup.

## Status

This is a learning and experimentation repo, research-grade rather than a polished product.
Each track has its own README and a PROGRESS file tracking what is done and what is next.
