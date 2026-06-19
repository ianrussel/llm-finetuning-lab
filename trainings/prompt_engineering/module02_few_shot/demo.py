"""Module 02: few-shot. Show the model examples instead of only describing the task.

Run from the prompt_engineering folder (Ollama up):
    python module02_few_shot/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm

QUERY = "Hi, the invoice I got charges me twice for the same month, can you check?"

# Zero-shot: describe the task. The model may explain, hedge, or vary the wording.
WEAK = (
    "Classify the customer message into one of: billing, technical, account.\n\n"
    f"Message: {QUERY}"
)

# Few-shot: a few labelled examples teach the exact output shape (one bare label).
STRONG = (
    "Classify each customer message into exactly one label: billing, technical, account.\n"
    "Reply with only the label.\n\n"
    "Message: My app keeps crashing when I upload a file.\nLabel: technical\n\n"
    "Message: I want to change the email on my profile.\nLabel: account\n\n"
    "Message: Why was I charged a late fee?\nLabel: billing\n\n"
    f"Message: {QUERY}\nLabel:"
)

if __name__ == "__main__":
    llm.compare("classify a support message into a closed label set", WEAK, STRONG)
    print("\nNOTICE: zero-shot often adds explanation or phrases the label loosely; the few-shot\n"
          "version returns a clean, single label because the examples SHOW the format and the\n"
          "decision boundary. Few-shot is how you get consistency without a long description.\n"
          "Try it: swap in a tricky message (e.g. 'I was double-charged AND the app crashed').")
