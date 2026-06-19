"""Module 07: iterate and evaluate. Compare two prompts on a small test set, by score.

The skill is not writing one clever prompt; it's measuring two prompts on real cases and
keeping the better one. This is a tiny A/B harness over a handful of labelled inputs.

Run from the prompt_engineering folder (Ollama up):
    python module07_iterate_and_evaluate/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm

LABELS = ["billing", "technical", "account"]

# A small fixed test set: (message, correct label). Real eval needs more, but this shows the method.
TESTS = [
    ("I was charged twice for last month.", "billing"),
    ("The mobile app crashes on startup.", "technical"),
    ("Please update the email on my profile.", "account"),
    ("My invoice total looks wrong.", "billing"),
    ("I can't log in even after resetting my password.", "technical"),
    ("How do I close my account?", "account"),
]


def variant_a(msg):  # loose zero-shot
    return f"What kind of support request is this: {msg}"


def variant_b(msg):  # constrained few-shot
    return (
        "Classify the message into exactly one label: billing, technical, account. "
        "Reply with only the label.\n\n"
        "Message: Why is there a late fee on my bill?\nLabel: billing\n\n"
        "Message: The dashboard won't load.\nLabel: technical\n\n"
        f"Message: {msg}\nLabel:"
    )


def predict(output):
    """Map free text to a label the way the eval scripts do: the label that appears, longest wins."""
    low = output.lower()
    hits = [l for l in LABELS if l in low]
    return max(hits, key=len) if hits else None


def score(name, build):
    correct = 0
    for msg, gold in TESTS:
        pred = predict(llm.ask(build(msg), temperature=0.0, num_predict=40))
        ok = pred == gold
        correct += ok
        print(f"  [{ 'OK ' if ok else 'XX ' }] gold={gold:<9} pred={pred}  <- {msg[:42]}")
    acc = correct / len(TESTS)
    print(f"  {name} accuracy: {correct}/{len(TESTS)} = {acc:.2f}\n")
    return acc


if __name__ == "__main__":
    llm.preflight()
    print("\n=== Variant A: loose zero-shot ===")
    a = score("A", variant_a)
    print("=== Variant B: constrained few-shot ===")
    b = score("B", variant_b)
    winner = "B" if b > a else ("A" if a > b else "tie")
    print(f"WINNER: {winner}  (A={a:.2f} vs B={b:.2f})")
    print("\nNOTICE: you picked a prompt by a number on several cases, not by eyeballing one output.\n"
          "That is the whole discipline, write variants, run them over a small fixed test set, keep\n"
          "the one that scores higher. It's the same two-axis-eval mindset used to judge fine-tunes.")
