# Module 04: roles and constraints

The system prompt is where you set *durable* behavior, who the model is and the rules it
always follows, while the user message carries the specific request. Used well, it controls
persona, tone, length, and what to do or avoid.

## Role / persona
Tell the model who it is acting as: "a patient tutor for beginners," "a terse senior
engineer," "a careful medical-information assistant that defers to professionals." The role
shifts vocabulary, depth, and tone at once.

## Constraints (be explicit)
State the rules plainly, ideally as a short list:
- **length:** "under 80 words," "exactly 3 bullets,"
- **audience/level:** "for someone with no coding background,"
- **tone:** "friendly," "formal," "no marketing language,"
- **do/don't:** "use one analogy," "no code," "do not invent facts."

Positive instructions ("use one analogy") tend to work better than vague negatives, but clear
"don't" rules ("no jargon") help too.

## System vs user
Put the stable stuff (role, format, rules) in the **system** prompt and the changing request
in the **user** message. Then one system prompt governs many different user questions
consistently, which is exactly how the training tracks set their task contract.

## Run it
```
python module04_roles_and_constraints/demo.py
```
It asks "explain recursion" with no role vs with a beginner-tutor system prompt plus
constraints (one analogy, no code, under 80 words).

## What to notice
Same question, very different answers: generic and possibly long without a role; targeted,
constrained, and on-tone with one. Then change the role ("a terse senior engineer," "a pirate")
and watch tone and length follow the system prompt while the question stays the same.
