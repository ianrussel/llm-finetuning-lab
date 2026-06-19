"""Module 05: chain-of-thought. Ask the model to reason step by step on a multi-step problem.

Run from the prompt_engineering folder (Ollama up):
    python module05_chain_of_thought/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm

# A small multi-step problem. Correct answer: 3*12=36, -17=19, +2*12=24 -> 43.
PROBLEM = ("A shop starts with 3 boxes of 12 apples each. It sells 17 apples, then receives "
           "2 more full boxes of 12. How many apples does the shop have now?")

# Direct: forbid working. Small models often slip on multi-step arithmetic this way.
WEAK = f"{PROBLEM}\nAnswer with only the final number, nothing else."

# Chain-of-thought: allow step-by-step working, then a clearly marked final answer.
STRONG = (f"{PROBLEM}\n\nThink step by step, showing each calculation. "
          "Then put the final answer on the last line in the form 'Answer: <number>'.")

if __name__ == "__main__":
    llm.compare("solve a multi-step word problem (answer is 43)", WEAK, STRONG, num_predict=400)
    print("\nNOTICE: forcing a one-shot number often produces a wrong answer on multi-step problems;\n"
          "letting the model work step by step (then state the answer) usually gets it right, because\n"
          "it computes intermediate results instead of guessing in one jump. The cost is verbosity\n"
          "and tokens, so reserve chain-of-thought for genuinely multi-step tasks, and parse the\n"
          "final 'Answer:' line when you only need the result.")
