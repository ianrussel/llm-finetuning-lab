# Prompt engineering: progress

My hands-on log for learning prompt engineering. Goal: be able to take a vague request and
turn it into a precise, tested prompt, and know which lever (clarity, examples, format,
constraints, reasoning, grounding) to reach for.

Status: [ ] not started, [~] in progress, [x] done

| Module | Technique | Status |
|--------|-----------|--------|
| 01 | Anatomy + clarity/specificity | [ ] |
| 02 | Few-shot (in-context examples) | [ ] |
| 03 | Structured output (JSON) | [ ] |
| 04 | Roles and constraints | [ ] |
| 05 | Chain-of-thought | [ ] |
| 06 | Grounding / anti-hallucination | [ ] |
| 07 | Iterate and evaluate | [ ] |
| 08 | Pitfalls and injection (reading) | [ ] |

## Setup
- [ ] Ollama running with qwen2.5:3b-instruct (ollama serve)

## Per module
- [ ] 01: ran the demo, saw weak vs strong, then wrote my own specific prompt
- [ ] 02: saw few-shot make the output consistent; tried changing the examples
- [ ] 03: got valid JSON out and parsed it; broke it on purpose to see why format rules matter
- [ ] 04: controlled tone/length with a system prompt + constraints
- [ ] 05: saw step-by-step fix a multi-step problem; noticed the verbosity cost
- [ ] 06: made the model abstain on an unanswerable question (grounding)
- [ ] 07: ran the A/B harness and picked the better prompt by score, not by feel
- [ ] 08: read the pitfalls + injection notes

## What I learned / surprised me
-
