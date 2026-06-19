"""Phase 2, generation pass 2: Evol-Instruct-style expansion.

Paraphrasing only varies surface wording. Evol-Instruct (WizardLM) instead
*evolves* each seed into a harder, richer, more realistic query, which teaches
the classifier to hold the intent steady under more varied and difficult input.
We apply a few distinct evolution operators per seed.

The hard rule is the same as paraphrasing: the customer's underlying request,
and therefore the intent label, must not change. An operator may add context,
detail, emotion, or a back-story, but it may not turn a `card_arrival` question
into a `card_not_working` one. The judge step checks that this held.

Reads  : data/seed.jsonl
Writes : data/gen_evol.jsonl  (one {user, intent, method, seed} per line)

Run from the track_a_banking77 folder:
    ../../../.venv/bin/python phase2_synthetic/evolve.py
"""

import json
import os
import sys

# common.py lives one level up; sdg.py sits next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # parent folder -> common.py
sys.path.insert(0, _HERE)                    # this folder -> sdg.py
import common
import sdg

OUT_FILE = f"{common.DATA}/gen_evol.jsonl"

# Each operator is one evolution direction. Keep the list short: more operators
# means more local model calls (one call per seed per operator).
OPERATORS = {
    "deepen": ("Rewrite it as a longer, more detailed message that adds realistic "
               "background or context, while asking for the exact same thing."),
    "constraint": ("Rewrite it with an extra specific condition or detail (an amount, "
                   "a date, a country, a device) that fits the same request."),
    "concretize": ("Rewrite it as a frustrated or confused real customer would type it, "
                   "with informal phrasing, while keeping the same underlying request."),
}

EVOL_SYSTEM = ("You rewrite a bank customer's message into a more complex but realistic "
               "version. The customer's underlying request must stay identical, so the "
               "support category never changes. You never invent a different problem. "
               "You output only the rewritten message, with no greetings or sign-offs.")


def evolve(message, instruction):
    text = sdg.chat([
        {"role": "system", "content": EVOL_SYSTEM},
        {"role": "user", "content":
            f"{instruction}\n\nKeep it to one message, no extra text.\n\n"
            f"Message: \"{message}\""},
    ], num_predict=200)
    # take the first non-empty cleaned line; evolved versions are single messages
    for raw in text.splitlines():
        s = sdg.clean_line(raw)
        if len(s.split()) >= 3:
            return s
    return None


def main():
    sdg.preflight()
    seeds = common.read_jsonl(f"{common.DATA}/seed.jsonl")
    written = 0
    with open(OUT_FILE, "w") as f:
        for i, row in enumerate(seeds, 1):
            message = common.user_of(row)
            intent = common.assistant_of(row)
            for name, instruction in OPERATORS.items():
                s = evolve(message, instruction)
                if not s:
                    continue
                f.write(json.dumps({"user": s, "intent": intent,
                                    "method": f"evol:{name}", "seed": message},
                                   ensure_ascii=False) + "\n")
                written += 1
            print(f"[{i}/{len(seeds)}] {intent:<28} {message[:40]!r}".ljust(90), end="\r")
    print(f"\nWrote {written} evolved rows to {OUT_FILE}. Next: judge.py")


if __name__ == "__main__":
    main()
