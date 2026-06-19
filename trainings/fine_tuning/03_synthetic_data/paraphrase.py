"""
Module 3: generate synthetic training data by paraphrasing the seed examples
with a LOCAL LLM served by Ollama.

Why this is still the "local" path: Ollama runs the model on this machine and serves
it at http://localhost:11434. The seeds are sent only to localhost, never to a cloud
API, so the privacy-preserving workflow holds. The upside over loading the model
in-process is that Ollama manages GPU/CPU offload, so it runs fine on a 4 GB card.

For each seed we ask the model for several reworded versions of the customer message
that keep the SAME meaning, topic, and urgency, and the SAME label. Variety, not new
facts. Output goes to data/generated.jsonl for filter.py to clean up.

Setup (once):
    ollama pull qwen2.5:3b-instruct      # ~2 GB, fits a 4 GB GPU
    # for better paraphrases if you have the room:
    # ollama pull qwen2.5:7b-instruct

Run from this folder:
    aipy paraphrase.py
"""

import json
import re
import urllib.error
import urllib.request

from common import read_jsonl, user_of, parse_label, assistant_of, looks_like_junk

OLLAMA_URL = "http://localhost:11434/api/chat"
GEN_MODEL  = "qwen2.5:3b-instruct"     # must be pulled in Ollama first
SEEDS_FILE = "data/seeds.jsonl"
OUT_FILE   = "data/generated.jsonl"
N_PER_SEED = 5

GEN_SYSTEM = ("You rewrite customer support messages. You keep the original meaning, "
              "topic, and urgency, and you never invent new facts or change what the "
              "customer wants. You never add greetings, sign-offs, or signatures, you "
              "output only the rewritten message text.")

PREAMBLE = ("here are", "sure", "certainly", "rewrite", "rewrites", "message:",
            "here's", "below are", "versions:")


def clean(line):
    """Strip bullets, numbering, and quotes a model tends to add."""
    s = line.strip()
    s = re.sub(r'^[\-\*•\d\.\)\(\s]+', '', s)
    return s.strip().strip('"').strip("'").strip()


def chat(messages):
    """One non-streaming chat call to the local Ollama server."""
    payload = {
        "model": GEN_MODEL,
        "messages": messages,
        "stream": False,
        # keep_alive 0 makes Ollama unload the model from VRAM right after this call,
        # so it does not squat on a small GPU while train.py needs the memory.
        "keep_alive": 0,
        "options": {"temperature": 0.9, "top_p": 0.95, "seed": 0, "num_predict": 240},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["message"]["content"]


def paraphrase(message, k):
    text = chat([
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content":
            f"Rewrite this customer message in {k} different ways. Keep the same "
            f"meaning, topic, and urgency. Vary the wording, length, and tone (some "
            f"terse, some polite). Each line must be one complete standalone version of "
            f"the message. Do NOT include greetings (Hi, Hello, Dear), sign-offs, "
            f"signatures, numbering, or any extra text.\n\nMessage: \"{message}\""},
    ])
    results = []
    for raw in text.splitlines():
        s = clean(raw)
        if len(s) < 3 or s.lower().startswith(PREAMBLE) or looks_like_junk(s):
            continue
        results.append(s)
        if len(results) >= k:
            break
    return results


def preflight():
    """Fail early with a helpful message if Ollama or the model is not ready."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except urllib.error.URLError:
        raise SystemExit("Ollama is not reachable on :11434. Start it with `ollama serve`.")
    if not any(n == GEN_MODEL or n.startswith(GEN_MODEL) for n in names):
        raise SystemExit(f"Model '{GEN_MODEL}' is not pulled. Run: ollama pull {GEN_MODEL}\n"
                         f"Available: {names or '(none)'}")


preflight()
seeds = read_jsonl(SEEDS_FILE)
written = 0
with open(OUT_FILE, "w") as f:
    for i, row in enumerate(seeds, 1):
        message = user_of(row)
        category, priority = parse_label(assistant_of(row))
        variants = paraphrase(message, N_PER_SEED)
        for v in variants:
            f.write(json.dumps({"user": v,
                                "category": category,
                                "priority": priority}) + "\n")
            written += 1
        print(f"[{i}/{len(seeds)}] {message[:50]!r} -> {len(variants)} paraphrases")

print(f"Done. Wrote {written} paraphrases to {OUT_FILE}. Next: aipy filter.py")
