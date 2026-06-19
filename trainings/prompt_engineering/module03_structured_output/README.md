# Module 03: structured output

When a program (not a human) consumes the answer, you need a reliable, parseable shape, most
often JSON. The model will happily produce JSON, but only if you ask precisely; otherwise it
wraps it in prose or code fences and your parser breaks.

## How to get clean structured output
- Name the **exact keys** and their **types** (`"amount_usd": number`).
- Say **"output ONLY the JSON"** and **"no prose, no code fences."**
- Optionally give one example of the exact object.
- Use clear **delimiters** around the input so the model doesn't confuse data with instructions.

## Delimiters
Wrap the material you're operating on so it's unambiguous, e.g. label it `Sentence:` / `TEXT:`
or fence it. This separates "the thing to process" from "the instructions," which also helps
against prompt injection (module 08).

## Run it
```
python module03_structured_output/demo.py
```
It extracts name/date/amount from a sentence two ways, then actually runs `json.loads` on each
output so you can see which one a program could use.

## What to notice
The loose prompt's output often fails to parse (extra prose, or ```json fences); the strict
prompt's output parses directly into a dict with the keys you asked for. The lesson: for
machine-consumed output, the format spec is not optional, and you should validate the result
(parse it, check the keys) rather than trust it, exactly what the synthetic-data pipelines do
with their lenient JSON parsing + checks.

Try it: drop the "no code fences" line and see if the model starts wrapping the JSON in
```json fences again.
