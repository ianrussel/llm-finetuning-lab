"""
Module 7, step 3: call the tuned Gemini model and score it on the eval set.

TEMPLATE: needs the tuned endpoint name that tune.py printed, plus the same GCP setup.
Uses the same scoring as modules 2/3 (valid JSON / category / exact) so the managed
result is directly comparable to the local fine-tunes.

Run:
    aipy predict.py
"""

import json

import vertexai
from vertexai.generative_models import GenerativeModel

# --- CONFIG: fill these in ---
PROJECT_ID     = "your-gcp-project-id"
LOCATION       = "us-central1"
TUNED_ENDPOINT = "projects/.../locations/.../endpoints/..."   # printed by tune.py
# -----------------------------

SYSTEM = ("You are a support ticket classifier. Reply with ONLY a JSON object "
          "of the form {\"category\": one of [billing, technical, account, general], "
          "\"priority\": one of [low, medium, high]} and nothing else.")

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(TUNED_ENDPOINT, system_instruction=SYSTEM)


def parse(text):
    try:
        obj = json.loads(text)
        return obj.get("category"), obj.get("priority")
    except Exception:
        return None, None


rows = [json.loads(l) for l in open("data/source_eval.jsonl") if l.strip()]
valid = category = exact = 0
for r in rows:
    user = next(m["content"] for m in r["messages"] if m["role"] == "user")
    gold = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
    ans = model.generate_content(user).text.strip()
    c, p = parse(ans)
    g_c, g_p = parse(gold)
    valid += c is not None
    category += c == g_c
    exact += (c, p) == (g_c, g_p)
    print(f"WANT {gold}   GOT {ans}")

n = len(rows)
print(f"\nvalid JSON {valid}/{n}   category {category}/{n}   exact {exact}/{n}")
