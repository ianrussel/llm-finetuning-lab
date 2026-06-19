"""
Module 7, step 1: convert our support-ticket data into Vertex AI Gemini tuning format.

This is the only part of module 7 that runs locally, no cloud needed. It takes the same
classifier data from module 2 (messages format) and rewrites it into the JSONL shape
Vertex AI expects for Gemini supervised tuning:

    {
      "systemInstruction": {"role": "system", "parts": [{"text": "..."}]},
      "contents": [
        {"role": "user",  "parts": [{"text": "..."}]},
        {"role": "model", "parts": [{"text": "..."}]}
      ]
    }

Note the differences from the local (TRL) format we used before:
  - roles are "user" and "model" (not "assistant")
  - text lives under parts: [{"text": ...}], not "content": "..."
  - the system prompt is a separate top-level systemInstruction

Run from this folder:
    aipy make_dataset.py
"""

import json

SYSTEM = ("You are a support ticket classifier. Reply with ONLY a JSON object "
          "of the form {\"category\": one of [billing, technical, account, general], "
          "\"priority\": one of [low, medium, high]} and nothing else.")


def messages_of(row):
    """Pull (system, user, model) text out of a module-2 style messages row."""
    sys_text, user_text, model_text = SYSTEM, None, None
    for m in row["messages"]:
        if m["role"] == "system":
            sys_text = m["content"]
        elif m["role"] == "user":
            user_text = m["content"]
        elif m["role"] == "assistant":
            model_text = m["content"]
    return sys_text, user_text, model_text


def to_vertex(row):
    sys_text, user_text, model_text = messages_of(row)
    return {
        "systemInstruction": {"role": "system", "parts": [{"text": sys_text}]},
        "contents": [
            {"role": "user",  "parts": [{"text": user_text}]},
            {"role": "model", "parts": [{"text": model_text}]},
        ],
    }


def convert(src, dst):
    n = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            if not line.strip():
                continue
            out.write(json.dumps(to_vertex(json.loads(line))) + "\n")
            n += 1
    print(f"{src} -> {dst}  ({n} examples)")


convert("data/source_train.jsonl", "data/vertex_train.jsonl")
convert("data/source_eval.jsonl",  "data/vertex_eval.jsonl")
print("Done. Upload these to a GCS bucket, then run tune.py (see README).")
