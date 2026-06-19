"""Phase 2, rehearsal mix: keep general reasoning + tool-calling ability while
specializing on the ticket task (the brief's requirement 3, the Part C rehearsal idea).

Builds train_mix.jsonl = ~75% task rows + ~25% rehearsal, where the rehearsal is
~60% general reasoning (GSM8K, wrapped in the same <think> format) and ~40% tool
calling (a small slice of an open function-calling set). Mixing these in stops the
fine-tune from forgetting how to reason generally or call tools.

Reads  : data/train_synth.jsonl
Writes : data/train_mix.jsonl

Run from the track root (downloads the rehearsal sets from HF on first run):
    ../../.venv/bin/python phase2_synthetic/mix_rehearsal.py
"""

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import common_c as common

OUT = f"{common.DATA}/train_mix.jsonl"


def gsm8k_to_messages(ex):
    reasoning, _, final = ex["answer"].partition("####")
    content = f"<think>\n{reasoning.strip()}\n</think>\n\nThe answer is {final.strip()}."
    return {"messages": [{"role": "user", "content": ex["question"]},
                         {"role": "assistant", "content": content}]}


def xlam_to_messages(ex):
    tools = str(ex.get("tools", "")).strip()
    query = str(ex.get("query", "")).strip()
    answers = str(ex.get("answers", "")).strip()
    sysmsg = ("You are a function-calling assistant. Call the available tools when needed "
              "and return the calls as JSON.\nAvailable tools: " + tools)
    return {"messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": query},
                         {"role": "assistant", "content": answers}]}


# ShareGPT -> role/content. Hermes (the open fallback for tool-calling) stores turns as
# {"from","value"}; we map roles and keep a clean system+user+assistant triple (dropping
# tool-result/follow-up turns) so it trains cleanly under the chat template.
_ROLE = {"system": "system", "human": "user", "user": "user",
         "gpt": "assistant", "assistant": "assistant"}


def sharegpt_to_messages(ex):
    conv = ex.get("conversations") or ex.get("messages") or []
    sys_m = user_m = asst_m = None
    for t in conv:
        role = t.get("role") or _ROLE.get(str(t.get("from", "")).lower())
        content = t.get("content") or t.get("value") or ""
        if not (role and content):
            continue
        if role == "system" and sys_m is None:
            sys_m = content
        elif role == "user" and user_m is None:
            user_m = content
        elif role == "assistant" and asst_m is None and user_m is not None:
            asst_m = content
    if not (user_m and asst_m):
        return {"messages": []}
    msgs = ([{"role": "system", "content": str(sys_m)}] if sys_m else [])
    msgs += [{"role": "user", "content": str(user_m)},
             {"role": "assistant", "content": str(asst_m)}]
    return {"messages": msgs}


def _load(name, kw, n, conv):
    """Best-effort load + convert n rows; returns [] with a warning on failure.
    Rows that convert to empty messages are dropped (so request a little extra upstream)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(*name, split=f"{kw}[:{n}]")
        return [r for r in (conv(x) for x in ds) if r.get("messages")]
    except Exception as e:
        print(f"  WARNING could not load {name}: {type(e).__name__}: {str(e)[:100]}")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal-frac", type=float, default=0.25,
                    help="fraction of the final mix that is rehearsal (default 0.25)")
    args = ap.parse_args()

    rng = random.Random(0)
    task = common.read_jsonl(f"{common.DATA}/train_synth.jsonl")
    n_task = len(task)
    # solve n_reh / (n_task + n_reh) = frac
    frac = args.rehearsal_frac
    n_reh = round(n_task * frac / (1 - frac)) if frac < 1 else 0
    n_general = round(n_reh * 0.6)
    n_tool = n_reh - n_general

    print(f"task rows: {n_task}; targeting {n_reh} rehearsal "
          f"({n_general} general reasoning + {n_tool} tool-calling)")
    # GSM8K: use the canonical repo id; the bare "gsm8k" alias hits the legacy loader
    # script which the current datasets library rejects (HfUriError).
    general = _load(("openai/gsm8k", "main"), "train", n_general, gsm8k_to_messages)
    # Tool-calling: Salesforce/xlam is GATED (needs HF auth); fall back to the open
    # Hermes set, converting its ShareGPT turns to role/content.
    tool = _load(("Salesforce/xlam-function-calling-60k",), "train", n_tool, xlam_to_messages)
    if not tool:
        tool = _load(("NousResearch/hermes-function-calling-v1",), "train", n_tool,
                     sharegpt_to_messages)

    mix = [{"messages": r["messages"]} for r in task] + general + tool
    rng.shuffle(mix)
    common.write_jsonl(OUT, mix)
    print(f"wrote {len(mix)} rows -> {OUT} "
          f"(task {n_task} + general {len(general)} + tool {len(tool)})")
    if not general or not tool:
        print("NOTE: a rehearsal source failed to load; rerun with network for the full mix.")


if __name__ == "__main__":
    main()
