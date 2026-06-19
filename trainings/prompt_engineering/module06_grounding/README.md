# Module 06: grounding and anti-hallucination

Models will confidently invent things they don't know. The fix in a prompt is **grounding**:
give the model the source material, tell it to answer *only* from that material, and tell it
to **say so when the answer isn't there**.

## The pattern
```
Answer using ONLY the context below. If the answer is not in the context,
reply exactly: not in the context.

Context:
<the passage(s)>

Question: <the question>
```
Two instructions do the work:
- **"use only the context"** stops it from drawing on (possibly wrong) memory,
- **"if not present, say 'not in the context'"** gives it permission to abstain instead of
  guessing. Without the second, models tend to fabricate rather than admit ignorance.

## Why it matters
- It makes answers **checkable** against a source and far less likely to be made up.
- It's the inference-time half of RAG: you retrieve relevant passages, then ground the model
  on them and let it abstain when retrieval missed. (This is the same idea as Track B / RAFT,
  which fine-tunes that behavior in.)

## Run it
```
python module06_grounding/demo.py
```
It asks an **answerable** question and an **unanswerable** one (phone support, which isn't in
the context), each with an ungrounded prompt and a grounded one.

## What to notice
On the answerable question, both are fine. On the unanswerable one, the ungrounded prompt
tends to **hallucinate** (invents a phone-support answer), while the grounded prompt **abstains**
("not in the context"). Abstention is a feature: a system that says "I don't know" is far more
trustworthy than one that confidently makes things up.

Try it: remove the "if not present, say 'not in the context'" line and watch the grounded
prompt start guessing again, the abstention instruction is what earns the trust.
