"""Shared synthetic-data-generation helper: talk to a LOCAL Ollama model.

Same privacy-preserving choice as the earlier tracks: the model runs on this machine,
served at http://localhost:11434, so the knowledge base never leaves localhost. The
survey, generation, and judge steps all import from here so the connection logic lives
in one place. Lives at the track root because the Phase 1 survey also uses it.

Setup (once):
    ollama pull qwen2.5:3b-instruct
    ollama serve     # if it is not already running
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
        "keep_alive": 0,   # unload from VRAM after the call so it frees the small GPU
        "options": {"temperature": temperature, "top_p": 0.95,
                    "seed": seed, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.loads(resp.read())["message"]["content"]


def parse_json(text):
    """Pull the first JSON value (array or object) out of a model reply, tolerant of
    prose and code fences. Returns None on failure so callers can skip a bad row."""
    text = text.strip()
    if "```" in text:
        for p in text.split("```"):
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("[") or p.startswith("{"):
                text = p
                break
    for opn, cls in (("{", "}"), ("[", "]")):
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
