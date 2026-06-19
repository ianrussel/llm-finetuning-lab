"""Module 03: structured output. Make the model return machine-parseable JSON.

Run from the prompt_engineering folder (Ollama up):
    python module03_structured_output/demo.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm

SENTENCE = "On 2024-03-14, Maria Santos paid 1,250.50 USD for the annual plan."

# Loose: asks for json but doesn't fix the keys, types, or forbid prose/fences.
WEAK = f"Extract the info from this sentence as json:\n{SENTENCE}"

# Strict: exact keys, exact types, and "ONLY JSON, no prose, no code fences."
STRONG = (
    "Extract fields from the sentence below.\n"
    "Output ONLY a JSON object, no prose and no code fences, with exactly these keys:\n"
    '  "name": string, "date": string in YYYY-MM-DD, "amount_usd": number\n\n'
    f"Sentence: {SENTENCE}"
)


def try_parse(label, text):
    """Show whether the output is actually parseable JSON."""
    try:
        obj = json.loads(text)
        print(f"  [{label}] parsed OK -> {obj}")
    except Exception as e:
        print(f"  [{label}] NOT valid JSON ({e}); raw was: {text[:120]!r}")


if __name__ == "__main__":
    llm.preflight()
    weak_out = llm.ask(WEAK)
    strong_out = llm.ask(STRONG)
    llm.show("WEAK response", weak_out)
    llm.show("STRONG response", strong_out)
    print("\nparse test (can your code actually use it?):")
    try_parse("weak", weak_out)
    try_parse("strong", strong_out)
    print("\nNOTICE: the loose prompt often wraps JSON in prose or ```json fences, so json.loads\n"
          "fails. The strict prompt (exact keys, types, 'ONLY JSON, no fences') returns something\n"
          "your program can parse directly. For anything a script consumes, specify the schema.")
