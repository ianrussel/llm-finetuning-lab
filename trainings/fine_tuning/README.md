# ai_training

My hands-on fine-tuning practice, split into 8 milestones (7 core + a lighter-touch
serving module). I track progress in PROGRESS.md and edit it by hand as I go.

```
ai_training/
├── PROGRESS.md                 # progress tracker (start here)
├── development.md              # how to run + environment gotchas (venv, GPU, pyenv)
├── requirements.txt            # shared base deps; each module adds its own extras
├── 01_lora_sft/                # has code: LoRA/QLoRA SFT
├── 02_dataset_from_scratch/    # has code: hand-built JSONL dataset, trained from scratch
├── 03_synthetic_data/          # has code: local-LLM paraphrase + quality filter
├── 04_embedding_finetune/      # has code: embedding model + hard-negative mining
├── 05_model_merge/             # has code: MergeKit merge of fine-tune + base
├── 06_multimodal_finetune/     # has code: small VLM (SmolVLM) on synthetic image-QA
├── 07_vertex_ai_managed/       # has code: Vertex AI Gemini managed tune (optional)
└── 08_serving_finetuned/       # has code: serving (vLLM multi-LoRA) + GGUF/Ollama (lighter touch)
```

All 8 modules now have code. Modules 1-6 run locally; module 7 (Vertex AI) is the managed
cloud path (data prep runs locally, the tune needs a GCP account); module 8 (serving) is a
lighter-touch reference with runnable vLLM and Ollama templates.

For how to run scripts and the environment gotchas on this machine, see development.md.

Throughout: small models, small datasets. Reps over scores.
