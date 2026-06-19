# Module 01: anatomy of a prompt + clarity

The biggest, cheapest win in prompt engineering is being specific. The model does what you
*said*, not what you *meant*, so most "bad outputs" are really under-specified prompts.

## The parts of a prompt
A well-formed prompt usually has some of these, in this rough order:
- **Role / system** — who the model is acting as (set in the system message; module 04).
- **Instruction** — the task, stated as a clear imperative.
- **Context** — the material to work on (put it after the instruction, clearly delimited).
- **Examples** — show the desired form when describing it is hard (module 02).
- **Output format** — exactly what shape the answer should take (module 03).
- **Constraints** — length, tone, what to include/exclude (module 04).

You don't need all of them every time, but naming them helps you see what's missing when a
prompt underperforms.

## The lever: clarity and specificity
Replace vague asks with exact ones. For every prompt, pin down:
- **Audience** (for a manager? a five-year-old? an API?)
- **Format** (bullets? one sentence? JSON?)
- **Length** (how many words/points?)
- **Focus** (what matters, what to ignore?)

## Run it
```
python module01_anatomy_and_clarity/demo.py
```
`demo.py` summarizes an incident report two ways: a vague "summarize this" vs a prompt that
fixes audience, format (3 bullets), length (<15 words each), and focus (impact + actions).

## What to notice
The weak output varies in length and structure run-to-run and may include background you
don't care about. The strong output is consistent and to the point. You didn't make the
model smarter, you removed the ambiguity it was guessing through.

Then experiment: change one constraint at a time (length, audience, focus) and watch the
output follow. That tight loop, change the words, see the effect, is the whole skill.
