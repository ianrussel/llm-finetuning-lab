"""Module 04: roles and constraints. Use the system prompt to control persona, tone, length.

Run from the prompt_engineering folder (Ollama up):
    python module04_roles_and_constraints/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm

QUESTION = "Explain what recursion is."

# Weak: no role, no constraints. You get a generic, possibly long, technical answer.
WEAK_SYSTEM = None
WEAK_USER = QUESTION

# Strong: a role (who the model is) + explicit constraints (audience, length, do/don't).
STRONG_SYSTEM = (
    "You are a patient tutor for absolute beginners with no coding background. "
    "You explain with one everyday analogy, in plain language. You never use code or jargon, "
    "and you keep answers under 80 words."
)
STRONG_USER = QUESTION

if __name__ == "__main__":
    llm.preflight()
    llm.show("WEAK (no role/constraints)", WEAK_USER)
    llm.show("WEAK response", llm.ask(WEAK_USER, system=WEAK_SYSTEM))
    llm.show("STRONG system prompt", STRONG_SYSTEM)
    llm.show("STRONG response", llm.ask(STRONG_USER, system=STRONG_SYSTEM))
    print("\nNOTICE: the same question yields a generic (often long, jargon-y) answer with no role,\n"
          "and a beginner-friendly, analogy-based, under-80-words answer once the system prompt sets\n"
          "the persona and constraints. The system prompt is where you set durable behavior; the\n"
          "user message carries the specific request.\n"
          "Try it: change the role to 'a terse senior engineer' and watch the tone flip.")
