"""Shared synthetic-data-generation helpers: talk to a LOCAL Ollama model.

Same privacy-preserving choice as module 3: the model runs on this machine and
is served at http://localhost:11434, so seeds never leave localhost. Ollama also
manages GPU/CPU offload, so a small card is fine. paraphrase.py, evolve.py and
judge.py all import from here so the connection logic lives in one place.

Setup (once):
    ollama pull qwen2.5:3b-instruct      # ~2 GB, fits a 4 GB GPU
    ollama serve                         # if it is not already running
"""

import json
import re
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
GEN_MODEL = "qwen2.5:3b-instruct"   # must be pulled in Ollama first


def chat(messages, temperature=0.9, num_predict=240, seed=0):
    """One non-streaming chat call to the local Ollama server."""
    payload = {
        "model": GEN_MODEL,
        "messages": messages,
        "stream": False,
        # keep_alive 0 unloads the model from VRAM right after the call, so it
        # does not squat on a small GPU while a later step needs the memory.
        "keep_alive": 0,
        "options": {"temperature": temperature, "top_p": 0.95,
                    "seed": seed, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["message"]["content"]


def clean_line(line):
    """Strip bullets, numbering, and surrounding quotes a model tends to add."""
    s = line.strip()
    s = re.sub(r'^[\-\*•\d\.\)\(\s]+', '', s)
    return s.strip().strip('"').strip("'").strip()


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
