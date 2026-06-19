# Module 6: Multimodal fine-tune

Goal: run a small VLM (vision-language model) fine-tune on an image-QA dataset to feel
the multimodal flow and the freeze-versus-train choice.

## The multimodal flow

An image goes through a **vision encoder** into visual features, those features are
**projected** into the language model's token space, and the **language model** generates
an answer conditioned on both the image tokens and the question text. Fine-tuning can
touch any of those three parts; the choice of which to train is the main decision.

## The freeze-versus-train choice (the point of this module)

We add LoRA adapters to the **language model only** and leave the **vision encoder
frozen**.

- Why freeze the vision tower: it already sees shapes and colors fine. Training it costs
  far more memory and can wreck good visual features on a tiny dataset. Freezing it is
  the cheap, stable default.
- What that means for results: the fine-tune can change *how the model answers*, not
  *how it sees*. (We saw exactly this, more below.)
- Watch out: the vision encoder and the text model share projection names
  (both have q_proj/k_proj/v_proj). A plain target-modules list would adapt the vision
  tower too. train.py scopes LoRA with a regex matching only `model.text_model.*`, which
  is what actually keeps the vision encoder frozen.

## The task and data

`make_data.py` draws tiny synthetic images: one colored shape (circle/square/triangle in
red/green/blue/yellow) on white, with a little jitter so no two are identical. Each gets
a question (color or shape) and a one-word answer. No downloads, fully reproducible, and
the answer depends only on the pixels, so it is a real vision task.

## Files

- make_data.py: builds data/images/*.png and data/train.jsonl (36) + data/eval.jsonl (12).
- train.py: LoRA fine-tune of the small VLM, vision frozen. Saves the adapter to ./vlm-lora.
- evaluate.py: before/after on the held-out images, with two metrics (see below).
- common.py: MODEL_ID and a jsonl reader.

## How to run

```bash
aipy make_data.py     # build the synthetic image-QA set
aipy train.py         # LoRA fine-tune (vision frozen) -> ./vlm-lora
aipy evaluate.py      # base vs fine-tuned on the held-out images
```

Model is HuggingFaceTB/SmolVLM-256M-Instruct (tiny, fits a 4 GB GPU). On a bigger GPU
bump common.py to SmolVLM-500M-Instruct or SmolVLM-Instruct.

## Making it fit a 4 GB GPU

VLMs are memory-hungry because images become many tokens and the lm_head is large. Three
things make this run on 4 GB:

- `processor.image_processor.do_image_splitting = False`: our images are one simple shape,
  so we do not tile them into sub-images. This is the big lever (far fewer image tokens).
- batch size 1 with gradient accumulation 8.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

## Result and the lesson

Training ran in ~90s on a 4 GB GPU, loss 7.97 -> 2.94, only ~2.4M LoRA params (0.94%) and
the vision encoder frozen. Before/after on the 12 held-out images:

| metric | base | fine-tuned |
|--------|------|-----------|
| contains the right word (lenient) | 12/12 | 11/12 |
| exact one-word answer (strict)    | 4/12  | 11/12 |

What this says:

- The fine-tune did exactly what fine-tuning does best here: it changed the **output
  format**. The base answers verbosely ("The shape in the image is a square"); the adapter
  answers in one word ("square"). Strict exact-match jumped 4/12 -> 11/12.
- It did **not** improve vision. The base already saw the shapes (the lenient metric was
  already 12/12), and because we froze the vision encoder, the one genuine visual mistake
  (an image called circle instead of square) stayed wrong. That is the freeze choice made
  concrete: train the language side, leave the seeing alone.
- Metric design matters. The lenient substring metric hid the change (and even looked
  like a regression); the strict metric revealed it. Always measure the thing you
  actually changed.

## Make it my own

- To try improving the seeing, not just the answering, also LoRA the vision encoder
  (widen the regex) or unfreeze the connector, and watch memory and stability change.
- Make the task harder (smaller shapes, two shapes per image, counting) so the base model
  actually fails and there is vision headroom to recover.
- Add an exact-format metric from the start; it is what shows SFT format learning.
