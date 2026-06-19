"""Module 06: grounding. Make the model answer ONLY from given context, and abstain otherwise.

Run from the prompt_engineering folder (Ollama up):
    python module06_grounding/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm

CONTEXT = (
    "Acme Cloud offers three plans. The Starter plan is $10/month and includes 5 projects. "
    "The Pro plan is $30/month and includes unlimited projects and email support. "
    "All plans include a 14-day free trial."
)

ANSWERABLE = "How much is the Pro plan and what does it include?"
UNANSWERABLE = "Does Acme Cloud offer phone support?"   # not in the context


def grounded(question):
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, reply exactly: not in the context.\n\n"
        f"Context:\n{CONTEXT}\n\nQuestion: {question}"
    )


def ungrounded(question):
    return f"{CONTEXT}\n\n{question}"


if __name__ == "__main__":
    llm.preflight()
    print("\n--- ANSWERABLE question (in the context) ---")
    llm.show("ungrounded", llm.ask(ungrounded(ANSWERABLE)))
    llm.show("grounded", llm.ask(grounded(ANSWERABLE)))
    print("\n--- UNANSWERABLE question (NOT in the context) ---")
    llm.show("ungrounded (may hallucinate)", llm.ask(ungrounded(UNANSWERABLE)))
    llm.show("grounded (should abstain)", llm.ask(grounded(UNANSWERABLE)))
    print("\nNOTICE: for the answerable question both do fine. For the UNANSWERABLE one, the\n"
          "ungrounded prompt tends to make something up (phone support that wasn't stated), while\n"
          "the grounded prompt replies 'not in the context'. Telling the model to use only the\n"
          "provided text AND to abstain is the core anti-hallucination move, and it's exactly what\n"
          "pairs with RAG (answer from retrieved passages, say 'I don't know' otherwise).")
