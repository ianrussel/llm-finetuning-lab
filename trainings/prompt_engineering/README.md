# Prompt engineering (hands-on)

A self-paced course to learn prompt engineering by running it. Each module pairs a short
concept page with a runnable demo that shows the same task with a **weak** prompt and a
**strong** prompt, so you see the technique change the model's output with your own eyes.

The demos call a LOCAL model through Ollama, so everything is free and offline. They use
only the Python standard library (no torch/transformers), so any python works as long as
Ollama is up.

## Setup (once)

```
ollama pull qwen2.5:3b-instruct
ollama serve     # leave running in another terminal
```

## How to run a module

From this folder:

```
python module01_anatomy_and_clarity/demo.py
```

Each demo prints the weak prompt + its response, then the strong prompt + its response, then
a note on what to notice. Read the module's README first, run the demo, then try editing the
prompts yourself, that is where the learning happens.

## The path

| Module | Technique | Why it matters |
|--------|-----------|----------------|
| 01 | Anatomy of a prompt + clarity/specificity | the single biggest lever: say exactly what you want |
| 02 | Few-shot (in-context examples) | show the format/behavior instead of describing it |
| 03 | Structured output (JSON, delimiters) | get machine-parseable, reliable output |
| 04 | Roles and constraints (system prompt) | control persona, tone, length, do/don't |
| 05 | Chain-of-thought (reasoning) | step-by-step for multi-step problems, and its cost |
| 06 | Grounding and anti-hallucination | answer only from context, abstain when unsure |
| 07 | Iterate and evaluate | improve prompts with a tiny A/B test, not by eyeballing |
| 08 | Pitfalls and prompt injection (reading) | common mistakes and untrusted-input safety |

Work them in order. Track progress in PROGRESS.md.

## The one idea to hold

A prompt is an interface, not a wish. The model does what you actually said, not what you
meant. Prompt engineering is making "what you said" precise: clear instructions, the right
context, examples when needed, an explicit output format, and constraints. When a prompt
fails, the fix is almost always to be more specific or to show an example, then test it.

## Where this connects to the rest of the repo

- Good prompts are the cheap first thing to try before fine-tuning (see
  ../FINE_TUNING_PLAYBOOK.md, Step 0 and the few-shot check).
- The grounding module (06) is the same idea as Track B / RAFT: answer only from provided
  context and abstain otherwise.
- The evaluate module (07) is the same discipline as the two-axis eval: measure on a small
  fixed set, do not trust a single example.
