"""Shared helper for the prompt-engineering demos: call a LOCAL Ollama model.

Every demo imports this so the technique is what changes, not the plumbing. Uses only
the Python standard library, so any python works as long as Ollama is running.

Setup (once):
    ollama pull qwen2.5:3b-instruct
    ollama serve
"""

import json
import textwrap
import urllib.error
import urllib.request

URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b-instruct"


def ask(user, system=None, temperature=0.3, num_predict=512, seed=0):
    """Send one prompt to the local model and return its text reply.

    temperature defaults low (0.3) so the demos are fairly reproducible; raise it to
    see how much the wording wanders.
    """
    messages = ([{"role": "system", "content": system}] if system else [])
    messages += [{"role": "user", "content": user}]
    payload = {
        "model": MODEL, "messages": messages, "stream": False, "keep_alive": 0,
        "options": {"temperature": temperature, "top_p": 0.95, "seed": seed,
                    "num_predict": num_predict},
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"].strip()


def preflight():
    """Fail early with a clear message if Ollama or the model is not ready."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except urllib.error.URLError:
        raise SystemExit("Ollama is not reachable on :11434. Start it with `ollama serve`.")
    if not any(n == MODEL or n.startswith(MODEL) for n in names):
        raise SystemExit(f"Model '{MODEL}' is not pulled. Run: ollama pull {MODEL}")


def show(title, text):
    """Print a labelled block so weak-vs-strong comparisons are easy to read."""
    print("\n" + "=" * 6 + f" {title} " + "=" * 6)
    print(textwrap.indent(text, "  "))


def compare(task, weak, strong, system_weak=None, system_strong=None, **kw):
    """Run a weak prompt and a strong prompt for the same task and print both."""
    preflight()
    print(f"\n### TASK: {task}")
    show("WEAK prompt", weak)
    show("WEAK response", ask(weak, system=system_weak, **kw))
    show("STRONG prompt", strong)
    show("STRONG response", ask(strong, system=system_strong, **kw))
