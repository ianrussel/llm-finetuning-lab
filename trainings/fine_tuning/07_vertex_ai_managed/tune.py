"""
Module 7, step 2: launch a Vertex AI supervised tuning job for Gemini.

TEMPLATE: this does not run locally. It needs a Google Cloud project with billing and the
Vertex AI API enabled, the SDK installed, and the data uploaded to a GCS bucket. Fill in
the CONFIG below, then run.

Setup (once):
    pip install google-cloud-aiplatform
    gcloud auth application-default login
    gcloud config set project YOUR_PROJECT_ID
    gsutil mb gs://YOUR_BUCKET                       # create a bucket
    gsutil cp data/vertex_train.jsonl data/vertex_eval.jsonl gs://YOUR_BUCKET/

Run:
    aipy tune.py
"""

import time

import vertexai
from vertexai.tuning import sft

# --- CONFIG: fill these in ---
PROJECT_ID   = "your-gcp-project-id"
LOCATION     = "us-central1"
BUCKET       = "your-bucket-name"
SOURCE_MODEL = "gemini-2.0-flash-001"   # check Vertex docs for currently tunable models
# -----------------------------

vertexai.init(project=PROJECT_ID, location=LOCATION)

# This is the whole "training" step in the managed world: one call. Google provisions the
# hardware, runs the (LoRA) tuning, and serves the result. You never see the GPUs, the
# loss loop, or the weights.
job = sft.train(
    source_model=SOURCE_MODEL,
    train_dataset=f"gs://{BUCKET}/vertex_train.jsonl",
    validation_dataset=f"gs://{BUCKET}/vertex_eval.jsonl",
    tuned_model_display_name="support-ticket-classifier",
    epochs=3,
    adapter_size=4,                 # LoRA rank; managed, so this is the only knob exposed
    learning_rate_multiplier=1.0,
)

print("Tuning job started:", job.resource_name)
print("Watch progress + the loss curve in the Vertex AI console under 'Tuning'.")
while not job.refresh().has_ended:
    time.sleep(60)
    print("  ...still tuning")

print("\nDone.")
print("tuned model endpoint:", job.tuned_model_endpoint_name)
print("Save that endpoint name; predict.py needs it.")
