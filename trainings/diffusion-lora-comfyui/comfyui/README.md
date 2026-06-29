# ComfyUI workflow engineering + optimization

The other half of the role: design and continuously optimize ComfyUI generation workflows. The bar
is "build and optimize complex workflows," not "use prebuilt ones," so the deliverable is a graph
you built plus a measured optimization (VRAM, throughput, repeatability).

## Build it (milestone 6)

1. **Base graph:** Load Checkpoint -> Load LoRA (your trained `.safetensors`) -> CLIP Text Encode
   (pos/neg) -> Empty Latent -> KSampler -> VAE Decode -> Save Image. Confirm the LoRA trigger token
   (`tk-emblem` for the toy set) actually fires.
2. **Production graph:** add the pieces real output needs - an upscale/refine pass (e.g. a second
   low-denoise KSampler or an upscaler), a face/detail fix, and consistent seeds. For video, swap in
   the model's i2v/t2v nodes (Wan 2.2 / LTX 2.3) and a frame-count/length control.
3. **Export** the `.json` (Workflow -> Export) - that file is your portfolio artifact.

## Optimize it (the part that gets hired)

Change one thing, measure VRAM (nvidia-smi) and wall-clock for a fixed prompt/seed, keep what wins:

- **VRAM / model offload:** enable model offloading and `--lowvram`/`--medvram`-style settings; use
  tiled VAE decode for big images; quantize/`fp8` the base where supported. Goal: fit the model and
  leave headroom.
- **Block swapping (video):** swap N transformer blocks to CPU to fit big video models - trades speed
  for VRAM. Find the minimum swap that fits.
- **Batching:** batch the latent for throughput when VRAM allows; otherwise queue. Measure images/min.
- **Node-graph efficiency:** cache/Reroute shared nodes, avoid recomputing CLIP/VAE, reuse loaded
  models across prompts, prune dead branches. ComfyUI only re-runs changed nodes - exploit that.
- **Repeatability:** fix seeds, pin model/LoRA versions, and save the workflow with the output so a
  run is reproducible. "Production-grade" means same input -> same output.

## What to record per optimization

| change | peak VRAM (GB) | time / image | quality delta | keep? |
|--------|----------------|--------------|---------------|-------|
| baseline | | | - | - |
| + tiled VAE | | | | |
| + fp8 base | | | | |
| + block swap N | | | | |

That table, plus the exported `.json`, is the ComfyUI evidence the application asks for.

## Install note

ComfyUI, the trainers (AI-Toolkit, Musubi Tuner, DiffSynth-Studio for Z-Image), and the model
weights are large and GPU-bound - follow each project's own install. This folder gives you the
method, the configs, the dataset, and the evaluation harness so every GPU hour is spent deliberately.
