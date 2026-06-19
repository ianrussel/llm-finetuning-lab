"""Module 01: clarity beats cleverness. Same task, vague prompt vs specific prompt.

Run from the prompt_engineering folder (Ollama up):
    python module01_anatomy_and_clarity/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm

TEXT = (
    "The support team reviewed the outage on Tuesday. They found a misconfigured cache "
    "caused 40 minutes of downtime for about 2,000 users. They rolled back the config, "
    "added an alert for cache health, and scheduled a post-mortem for Friday. The customer "
    "was notified and offered a service credit."
)

# Vague: no audience, no length, no format, no focus. The model guesses all of it.
WEAK = f"Summarize this:\n\n{TEXT}"

# Specific: audience, exact format, length cap, and what to focus on.
STRONG = (
    "Summarize the incident report below for a busy manager.\n"
    "Rules:\n"
    "- exactly 3 bullet points\n"
    "- each bullet under 15 words\n"
    "- focus on impact and the actions taken, not background\n\n"
    f"REPORT:\n{TEXT}"
)

if __name__ == "__main__":
    llm.compare("summarize an incident report", WEAK, STRONG)
    print("\nNOTICE: the strong prompt pinned down audience, count, length, and focus, so the\n"
          "output is consistent and usable. The weak one leaves all of that to chance.\n"
          "Try it yourself: change '3 bullet points' to 'one sentence' and re-run.")
