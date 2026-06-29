# raw/ — your own images go here

For a real LoRA, put 15-30 images of ONE subject here and a matching `<name>.txt` caption
per image (same convention as ../concept/). Tips that matter for quality:

- One clear subject, varied pose/background/lighting; avoid duplicates and watermarks.
- Caption with a rare trigger token + a short, true description; vary the wording.
- Keep resolution >= the training resolution (e.g. 1024) and crop to the subject.
- Point the trainer config at this folder instead of ../concept when you are ready.
