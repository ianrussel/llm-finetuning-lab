# Diffusion LoRA training + ComfyUI workflow engineering

A self-paced, self-contained module to build the exact skills the **AI Training & Workflow Engineer
(ComfyUI / LoRA Trainer Optimization, Image & Video)** role asks for, and to produce the portfolio
evidence the application requires. It is deliberately separate from the LLM fine-tuning work in this
repo and brings **its own dataset** (generated locally, nothing imported from other folders).

This is a *training-optimization* role, not a run-the-trainer operator role. The whole point is to
get better quality, faster speed, and lower cost out of the trainer and the ComfyUI workflows, and
to **measure** the improvement. This module is organised around that: tune configs, run sweeps,
compare results systematically, document before/after.

## What the job needs vs. what this module builds

| Job requirement | What you build here |
|-----------------|---------------------|
| Deep LoRA-training optimization with **before/after** evidence | A baseline LoRA config + a hyperparameter **sweep** (rank, LR, scheduler, regularisation) + a comparison harness that produces before/after contact sheets and a config-diff table (`configs/`, `eval/`) |
| Expert use of **AI-Toolkit** and **Musubi Tuner** | Annotated, runnable config templates for both, with every tunable knob explained (`configs/aitoolkit_lora.yaml`, `configs/musubi_video_lora.toml`) |
| **Full fine-tuning** of base models, not just LoRA | A milestone + config notes contrasting LoRA vs full FT (memory, when each wins) |
| Build/optimize complex **ComfyUI workflows** | A workflow build + a VRAM/offload/batching optimization checklist, and a starter graph (`comfyui/`) |
| **Diffusion training mechanics** (LR, rank, regularisation, overfitting, schedulers) | Each config knob is documented with its quality/speed/cost effect; the sweep makes the trade-offs concrete |
| **Pipeline optimization** for speed/cost/quality; multi-GPU | Caching, quantization, gradient checkpointing, batch/accumulation, multi-GPU notes in the configs + README |
| **Systematic evaluation** of configs/checkpoints | `eval/compare.py`: fixed prompts + fixed seeds across runs → side-by-side grids + metrics, the core "measure the improvement" skill |
| Following the latest open-source models | A living `comfyui/models.md` to track LTX 2.3, Wan 2.2, Qwen-Image-Edit, Z-Image and whatever drops next |

## The models the role names

- **ComfyUI** — the workflow environment everything runs in.
- **LTX 2.3** (video), **Wan 2.2** (video / image-to-video) — video generation + motion.
- **Qwen-Image-Edit** — instruction-based image editing.
- **Z-Image** (Alibaba Tongyi 6B; Turbo/Base/Edit; LoRA via DiffSynth-Studio) — image gen + editing.
- Trainers: **AI-Toolkit**, **Musubi Tuner** (and DiffSynth-Studio for Z-Image).

This space moves monthly, so the module teaches the *method* (config, sweep, measure) that transfers
to whatever model is current, and keeps a model-tracking note rather than hard-coding one model.

## Milestones (do them in order)

1. **Toolchain up** — install ComfyUI + AI-Toolkit + Musubi Tuner; render one image; train one toy
   LoRA end to end on the dataset below. Goal: a working pipeline you fully control.
2. **Baseline LoRA** — train the toy concept with `configs/aitoolkit_lora.yaml` as-is. Save fixed
   sample prompts + seeds. This is your "before".
3. **Sweep + measure** — vary one knob at a time (rank, LR, scheduler, steps, caption dropout) per
   `configs/sweep.yaml`; render the same prompts/seeds; run `eval/compare.py`. Produce a before/after
   grid and a one-paragraph finding per knob. **This is the deliverable the job asks for.**
4. **Speed/cost pass** — re-hit the best-quality config and cut time/VRAM (8-bit optimizer, latent
   caching, gradient checkpointing, batch vs accumulation, quantized base). Record steps/sec, peak
   VRAM, and quality delta.
5. **Video LoRA** — repeat 2-4 with Musubi Tuner on a short Wan 2.2 / LTX clip set (`configs/musubi_video_lora.toml`).
6. **ComfyUI workflow** — build a production graph (txt2img or i2v + LoRA + upscale/refine), then
   optimize node graph, offloading, and batching (`comfyui/README.md`).
7. **Full fine-tune (stretch)** — one small base full FT to contrast with LoRA on config + eval.

## What to send when you apply

The posting asks for three things; produce them here:

1. **LoRA optimization examples** — from milestone 3-4: the config diffs, the before/after grids from
   `eval/compare.py`, and the steps/sec + VRAM numbers. Concrete, measured, reproducible.
2. **ComfyUI workflows** — the graphs from milestone 6 (export the `.json`), with a note on what you
   optimized and the VRAM/throughput effect.
3. **Recent models + trainers note** — from `comfyui/models.md`: which of LTX 2.3 / Wan 2.2 /
   Qwen-Image-Edit / Z-Image you trained or built workflows for, most recently.

## How to use this folder

```
datasets/   make_dataset.py builds a small synthetic "concept" set (images + captions) so you can
            train a real toy LoRA today; drop your own images in datasets/raw/ for the real thing
configs/    annotated AI-Toolkit (image) + Musubi (video) configs, and a sweep definition
eval/       compare.py: same prompts/seeds across runs -> contact sheets + a config-diff table
comfyui/    workflow build + optimization checklist + a model-tracking note
```

## Honest scope

This module gives you a **runnable skeleton and a method**, not a one-click result. The training
runs and ComfyUI itself need a GPU and the trainers installed (AI-Toolkit, Musubi Tuner, ComfyUI) —
follow each tool's own install. The dataset generator, the configs, the sweep definition, and the
evaluation harness all run without a GPU, so you can set everything up and understand every knob
before you spend GPU time. The goal is that when you train, you train *deliberately* and can show the
before/after, which is exactly what the role is hiring for.
