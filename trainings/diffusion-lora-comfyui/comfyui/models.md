# Model + trainer tracker (keep this current)

The role wants someone "actively following the latest open-source image/video releases and building
workflows around them quickly." This is the living note you cite in the application's third ask
("which open-source models and trainers you've worked with most recently"). Update it as you go.

## The stack named in the posting

| model / tool | use | trainer / integration | status (fill in) |
|--------------|-----|------------------------|------------------|
| **ComfyUI** | workflow environment | - | |
| **LTX 2.3** | video generation | ComfyUI nodes | |
| **Wan 2.2** | video / image-to-video | Musubi Tuner LoRA | |
| **Qwen-Image-Edit** | instruction image editing | ComfyUI | |
| **Z-Image** (Tongyi 6B) | image gen / edit (Turbo/Base/Edit) | DiffSynth-Studio LoRA + ComfyUI | |
| **AI-Toolkit** | image LoRA / FT trainer | - | |
| **Musubi Tuner** | video/image LoRA trainer | - | |
| **DiffSynth-Studio** | Z-Image LoRA training | - | |

For each, record: did you build a workflow, train a LoRA, or just evaluate it; the date; one concrete
result (a sample, a VRAM/speed number, a config that worked).

## How to keep up (the actual skill)

- Watch the trainer repos (AI-Toolkit, Musubi Tuner, DiffSynth-Studio) and ComfyUI + popular custom
  nodes for releases; new base models usually land with a reference workflow within days.
- When a model drops: get it running in ComfyUI first (prove the workflow), then train a small LoRA
  on the toy `datasets/concept` set to learn its trainer quirks (rank ranges, schedulers, VRAM).
- Note quantization options (fp8, GGUF, nf4) per model - they decide what fits a given card and are a
  recurring optimization lever.

## Log

<!-- date - model/tool - what you did - result. Newest first. -->
- (start logging here)
