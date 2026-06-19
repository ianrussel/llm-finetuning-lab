"""Shared bits for module 5 (reuses the module 2/3 classifier task for scoring)."""

import json

SYSTEM = ("You are a support ticket classifier. Reply with ONLY a JSON object "
          "of the form {\"category\": one of [billing, technical, account, general], "
          "\"priority\": one of [low, medium, high]} and nothing else.")

# A couple of off-task prompts, to eyeball whether general ability survives the merge.
GENERAL_PROMPTS = [
    "What is the capital of France?",
    "Give me one short tip for saving money.",
]


def parse_label(text):
    try:
        obj = json.loads(text)
        return obj.get("category"), obj.get("priority")
    except Exception:
        return None, None


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def user_of(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def assistant_of(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "assistant")
