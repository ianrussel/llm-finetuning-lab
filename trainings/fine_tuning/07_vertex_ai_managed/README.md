# Module 7: Vertex AI managed tune (optional)

Goal: run one Vertex AI Gemini supervised tune to feel the managed workflow, and contrast
it with the local training from modules 1-6.

This one is different from the rest. There is no local GPU, no training loop, no weights
to inspect. You hand Google a dataset and a model name, it provisions hardware, tunes a
LoRA adapter on a Gemini model, and serves the result behind an endpoint. The lesson is
the contrast, not the code.

It needs a Google Cloud account with billing, so it is optional and costs a few dollars.
Only make_dataset.py runs locally; tune.py and predict.py are templates you fill in.

## Files

- make_dataset.py: converts the module 2 support-ticket data into the Vertex Gemini
  tuning JSONL format. Runs locally, no cloud needed. (Already run: see data/vertex_*.jsonl.)
- tune.py: template that launches the managed tuning job (fill in project/bucket).
- predict.py: template that calls the tuned model and scores it on the eval set, with the
  same valid/category/exact metric as modules 2/3.
- data/source_*.jsonl: the local-format source (copied from module 2).
- data/vertex_*.jsonl: the converted Vertex-format files to upload.

## The dataset format is different

Local (TRL) format, what modules 2/3 used:

```json
{"messages": [
  {"role": "system",    "content": "..."},
  {"role": "user",      "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

Vertex Gemini tuning format, what this module needs:

```json
{
  "systemInstruction": {"role": "system", "parts": [{"text": "..."}]},
  "contents": [
    {"role": "user",  "parts": [{"text": "..."}]},
    {"role": "model", "parts": [{"text": "..."}]}
  ]
}
```

Differences: roles are user/model (not assistant), text sits under parts: [{"text": ...}],
and the system prompt is a separate systemInstruction. make_dataset.py does this rewrite.

## Walkthrough

1. Convert the data (local):
   ```bash
   aipy make_dataset.py
   ```
2. Set up Google Cloud (once):
   ```bash
   pip install google-cloud-aiplatform
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable aiplatform.googleapis.com
   gsutil mb gs://YOUR_BUCKET
   gsutil cp data/vertex_train.jsonl data/vertex_eval.jsonl gs://YOUR_BUCKET/
   ```
3. Fill in PROJECT_ID / BUCKET in tune.py, then launch and wait:
   ```bash
   aipy tune.py
   ```
   Watch the job and its loss curve in the Vertex AI console under Tuning.
4. Copy the printed tuned endpoint name into predict.py, then score it:
   ```bash
   aipy predict.py
   ```

The official codelab walks the same path with screenshots: search "Google Cloud tune
Gemini Vertex AI supervised fine-tuning codelab".

## Local vs managed: the actual lesson

| | Local (modules 1-6) | Managed (Vertex AI) |
|--|--|--|
| What you write | the training loop, LoRA config, collator | a dataset + one train() call |
| Hardware | your GPU (4 GB), you fight OOM | Google's, invisible to you |
| Model | open weights (Qwen, SmolVLM) you can hold | Gemini, closed, you never see the weights |
| The adapter | a file on disk you can load/merge/inspect | lives behind an endpoint, not downloadable |
| Visibility | full: loss every step, every tensor | a loss curve in a console, little else |
| Control | total: precision, freezing, merge, hack internals | a few knobs (epochs, adapter size, LR multiplier) |
| Cost | free (your hardware) | per-token training + hosting, real money |
| Effort | high: env, drivers, memory, versions | low: upload and wait |
| Data privacy | stays on your machine | uploaded to Google Cloud |
| Iteration speed | as fast as your GPU | queue + provision, but scales to big models |

The takeaway: managed tuning trades **control and visibility** for **convenience and scale**.
Locally you own every detail and pay with your time and a 4 GB memory budget. On Vertex you
give up the internals and the weights, and pay with money, but you can tune a frontier-scale
model from a laptop with one call. Same core idea (supervised LoRA fine-tuning on a small
dataset), opposite ends of the effort/control spectrum.

## Status

make_dataset.py has been run (data/vertex_*.jsonl exist and are valid). Running the actual
tune needs a GCP account, so the cloud steps are left for when I want to spend the few
dollars. The point, feeling the managed workflow versus local, is captured above.
