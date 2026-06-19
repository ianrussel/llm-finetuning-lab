# Module 05: chain-of-thought (reasoning)

For multi-step problems (math, logic, planning, anything with intermediate results), asking
the model to **reason step by step before answering** often turns a wrong answer into a right
one. Working through the steps lets it compute intermediate values instead of guessing the
final answer in one jump.

## How to prompt for it
- "Think step by step, then give the final answer."
- Mark the final answer so you can extract it: "put the final answer on the last line as
  'Answer: <number>'."
- For consistency you can show one worked example (few-shot + reasoning).

## When it helps, and when it doesn't
- **Helps:** arithmetic, multi-hop reasoning, constraints to satisfy, decisions with several
  factors.
- **Doesn't help much:** simple lookups, classification, or formatting, where it just adds
  verbosity and tokens.

## The costs (don't over-use it)
- More tokens = slower and more expensive.
- The visible reasoning isn't always the "real" reason, treat it as a tool for better answers,
  not a literal audit trail.
- On reasoning-tuned models you control this differently (thinking mode), and over-long chains
  can hurt; aim for reasoning proportional to difficulty (the length-dynamics point from the
  reasoning-model work).

## Run it
```
python module05_chain_of_thought/demo.py
```
It solves a multi-step word problem (answer: 43) two ways: forced one-number answer vs
step-by-step then a marked final answer.

## What to notice
The direct prompt often gets it wrong on the small local model; the step-by-step prompt gets
it right because it does the arithmetic in stages. When you only need the result, keep the
reasoning but **parse the final `Answer:` line**, you get the accuracy without dumping the
whole chain to the user.
