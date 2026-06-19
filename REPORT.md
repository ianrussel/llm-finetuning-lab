PART A: Automated dataset construction from a knowledge source

What I did. I built the corpus-to-dataset pipeline end to end on two tracks and used them to
learn where the automation holds and where it breaks. Track A turns the banking77 corpus into a
77-way intent classification set. Track B turns SQuAD v2 Wikipedia passages into a grounded
question-answering set that answers only from the supplied context and abstains when the answer
is not present. Both run the same spine, so the spine is the real deliverable. I rolled the
pipeline by hand instead of using Distilabel, Augmentoolkit or Bonito, so I would understand
each step; those tools automate the same stages.

The pipeline. Seven stages, each shaped by the task contract written first:
1. Define the contract and metric before touching data: exact input, exact output, exact
   scoring. Without this, "balanced" has no axes to balance on.
2. Reserve a sacred evaluation set up front, stratified and decontaminated, and never let
   anything derived from it enter training. In Track A this is 1540 gold rows, 20 per intent.
   In Track B it is a held-out set of passages split by document, so a generated question can
   never be grounded in a gold passage.
3. Extract a small verified seed in the exact task shape, stratified across the balance axes.
   Track A used 385 rows, 5 per intent. Small is enough once the task is well defined.
4. Expand with grounded synthetic generation. Generate from the real source, varying phrasing,
   structure and difficulty, but never the verified fact or label. Track A paraphrased and
   evolved each seed; Track B generated answerable and unanswerable questions from each passage.
5. Quality gate, all as pipeline steps: an LLM-as-judge faithfulness pass, dedup and near-dedup,
   decontamination against gold, and a per-slice balance report. Track A judged 3010 candidates,
   kept 2179, and assembled 2562 training rows after dedup and decontamination.
6. Train a LoRA/QLoRA adapter on the same base the evaluator scores.
7. Evaluate on two axes (task gain and regression) and iterate on the weak slices.

What it produced. Track A went from a base macro-F1 of 0.201 to 0.534 with the seed alone and
0.769 with seed plus synthetic, while the general-capability sentinel held at 11/12 the whole
way and the valid-label rate climbed from 0.729 to 0.990. The synthetic data did most of its
work on the confusable card intents, with gains of 0.57 to 0.82 F1 on the pairs the base model
had been collapsing together. So the automated synthetic step more than doubled the metric over
the base and added another large jump over the seed-only control, with no measurable forgetting.

Q1. Given a large knowledge source and one target task, how would you structure an automated
pipeline that produces a balanced training set? Treat balance as a measured target, not an
afterthought. Name the balance axes from the contract (class, mode, difficulty, length, source
coverage), reserve a balanced gold set first, seed small and stratified, then generate to
per-slice quotas rather than in bulk. After the judge and dedup gates, measure a per-slice
histogram, top up the thin slices with targeted generation, and cap or downsample the heavy
ones. Keep the real seeds in the final mix as the guard against mode collapse. In Track A this
was per-intent counts across 77 intents; in Track B it was the answerable-versus-unanswerable
ratio that teaches the model when to abstain.

Q2. Where do you put the human in the loop so quality stays high without hand-labeling
everything? At the leverage points, not on every row. The human defines the contract and the
acceptance criteria, verifies the small sacred gold and seed sets, calibrates and audits the
LLM-judge against a small hand-labeled sample, reviews the error analysis to name the next
weakness, and makes the stop-or-continue call. Automation does the bulk generation,
per-candidate judging, dedup, decontamination and scoring. The mechanism that keeps it honest is
sample-based gates: review a random sample at each gate, measure that gate's error rate, and only
intervene when it exceeds tolerance. That turns "review everything" into "measure the reviewer,"
a fixed small cost no matter how large the set grows. In my own-corpus path for Track B the
drafted gold is written with verified=false on purpose, so a human has to check each answer
against its passage before any eval number counts.

Q3. How do you stop an automated generator from drifting off-source or producing a skewed,
repetitive set? Five controls, all in the pipeline. First, ground every generation in a real
passage or verified label and never let the model invent the fact or the label; vary only the
phrasing. Second, judge faithfulness per candidate and drop anything the judge cannot tie back to
the source (in Track B the judge also confirms an "unanswerable" question really is unanswerable,
so the model is not taught to abstain wrongly). Third, dedup and near-dedup with character-shingle
Jaccard, so near-identical rows cannot pile up and fake coverage. Fourth, decontaminate against
the gold set, exact and near-duplicate, before training. Fifth, enforce per-slice quotas and a
balance report, and keep the real seeds in the mix, then let the iterate loop's error analysis
catch any drift that slips through by showing which slice regressed. I saw this failure mode
directly: in Track A's iteration round, pushing more targeted data nudged the model into
over-specialization and the sentinel dropped from 11/12 to 10/12, so I stopped. That is the
skew-and-drift risk made concrete, and the two-axis eval is what caught it.


PART B: Reasoning models and reasoning-chain data

What I did. I looked at reasoning training from both sides. First I built a minimal reasoning
distillation SFT from scratch (Qwen2.5-0.5B-Instruct on a slice of GSM8K, mixed 75/25 with
Alpaca) so I could see the trace format and the data mix in plain code on a free T4. Then I ran
an Unsloth GRPO notebook to watch the reinforcement-learning approach, where the model learns to
reason from a reward instead of from copied traces. The GRPO notebook I ran happened to be a
vision one, so the visual task is beside the point, but the GRPO mechanics it showed are the same
as for text.

How a reasoning model's output is structured. The target is not question to answer, it is
question to an explicit reasoning trace to the answer, with the trace wrapped in a marker such as
<think> ... </think> and the final answer after it. The model is trained to show its work and
then commit to an answer, so the trace is part of the label, not just the result.

The two ways to get reasoning. There are two distinct paradigms, and I touched both.
- Distillation SFT. A strong teacher produces a (reasoning, answer) pair for each question, you
  keep only the items whose final answer is correct, and you do plain supervised fine-tuning on
  those triples. Small high-quality sets go a long way (s1 used about 1000 examples, LIMO about
  800). In my run the traces came ready-made from GSM8K, which already ships step-by-step
  solutions; in a real pipeline the teacher model would generate them and a correctness check
  would gate them. This is the cheaper, more controllable path, and it slots straight into the
  Part A pipeline: the teacher generates, the judge or a correctness check filters.
- Reinforcement learning (GRPO). Instead of copying traces, the model generates a group of its
  own completions per prompt, each is scored by a reward (did it reach the right answer, did it
  follow the format), and the policy is pushed toward the above-average completions in each
  group. This is the DeepSeek-R1 paradigm: it learns to reason with no distilled traces at all.
  It is heavier, because it generates several completions per step to compute the reward.

The 75/25 mix. I trained on about 75 percent reasoning data and 25 percent ordinary instruction
data on purpose, so the model keeps its general chat ability and does not learn to think on every
trivial prompt. This is the same idea as the rehearsal mix in Part C: the reasoning data is the
task, the plain instructions are the rehearsal that protects general behaviour.

Reasoning length dynamics. Teacher traces are often far longer than necessary, and training
naively on them produces a model that over-thinks or always emits a bloated reasoning block. The
aim is variety and appropriate length, reasoning in proportion to difficulty rather than maximum
length, so in a real build I would filter or trim the longest traces and keep a spread of lengths.

What I observed. In the SFT run the model started producing a <think> trace before its answer on
held-out GSM8K questions where the base model had not. [insert one base-vs-tuned example, and
whether the arithmetic got more reliable]. In the GRPO run the reward started near zero and was
noisy for the first few steps, exactly as the notebook warns (it takes roughly 150 to 200 steps
before the reward moves); [insert the reward at the start versus after about 200 steps, plus what
the completion length and KL did]. The GRPO training loss reads 0 throughout because the objective
is reward-based, not cross-entropy, so the reward column is the real signal, not the loss.

Takeaway. Distillation SFT is the practical default for building a reasoning specialist: cheap,
controllable, and it drops straight into the automated dataset pipeline from Part A. GRPO is the
stronger, heavier path for when you have a reliable reward and want the model to discover
reasoning rather than copy it. For the task-specific work here I would reach for distillation
first, and treat the 75/25 mix and trace-length control as the two knobs that most affect whether
the model reasons well without losing general ability.


Part C


Part D