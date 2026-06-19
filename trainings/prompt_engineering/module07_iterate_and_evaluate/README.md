# Module 07: iterate and evaluate

Prompt engineering is not a one-shot art; it's a loop. The real skill is **measuring** prompt
variants on real cases and keeping the better one, instead of eyeballing a single output and
declaring victory.

## Why eyeballing fails
- Models are non-deterministic; one good output can be luck.
- A prompt that nails one example may fail on the next.
- "It looks better" is not comparable across changes.

## The method
1. Build a small **fixed test set** of inputs with known-correct answers (even ~6 to 20 helps).
2. Write 2+ prompt **variants**.
3. Run each variant over the whole set and **score** it (exact match, contains the right label,
   a checker function).
4. Keep the variant that scores higher. Change one thing at a time so you know what helped.

This is the exact discipline the fine-tuning tracks use for models (gold set + metric); here
you apply it to prompts.

## Run it
```
python module07_iterate_and_evaluate/demo.py
```
It scores two classifier prompts (loose zero-shot vs constrained few-shot) over 6 labelled
messages and prints per-case results plus each variant's accuracy and the winner.

## What to notice
You choose the prompt by a **number over several cases**, not by a vibe on one. Usually the
constrained few-shot variant wins, but the point is that you can now *prove* it. Then extend
the test set, add a third variant, and keep iterating, the loop (write, measure, keep the best)
is the job.

## Tips
- Use `temperature=0` for evaluation so scores reflect the prompt, not sampling luck.
- Include hard/edge cases in the test set, those are where prompts differ.
- Track your test set in a file as it grows; it becomes your prompt regression suite.
