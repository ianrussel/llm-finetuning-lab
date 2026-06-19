"""Shared synthetic-data-generation helpers: talk to a LOCAL Ollama model.

Same privacy-preserving choice as Track A and module 3: the model runs on this
machine, served at http://localhost:11434, so passages never leave localhost.
qgen.py and judge.py both import from here so the connection logic lives in one
place.

Setup (once):
    ollama pull qwen2.5:3b-instruct      # ~2 GB, fits a 4 GB GPU
    ollama serve                         # if it is not already running
"""

import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
GEN_MODEL = "qwen2.5:3b-instruct"   # must be pulled in Ollama first


def chat(messages, temperature=0.7, num_predict=512, seed=0):
    """One non-streaming chat call to the local Ollama server."""
    payload = {
        "model": GEN_MODEL,
        "messages": messages,
        "stream": False,
        # keep_alive 0 unloads the model from VRAM after the call so it does not
        # squat on a small GPU while a later step needs the memory.
        "keep_alive": 0,
        "options": {"temperature": temperature, "top_p": 0.95,
                    "seed": seed, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.loads(resp.read())["message"]["content"]


def parse_json(text):
    """Pull the first JSON value (array or object) out of a model reply, tolerant
    of the prose and code fences a model tends to wrap it in. Returns None on
    failure so callers can skip a bad generation rather than crash."""
    text = text.strip()
    if "```" in text:
        # drop the first fenced block's fence lines
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("[") or p.startswith("{"):
                text = p
                break
    for opn, cls in (("[", "]"), ("{", "}")):
        i, j = text.find(opn), text.rfind(cls)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                pass
    return None


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
