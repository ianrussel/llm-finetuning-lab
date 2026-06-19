# The automated training pipeline (Part 2)

Part 1 turned a linked knowledge base + a target task into a training dataset. Part 2 is the
other half: a predefined, **config-driven** pipeline that takes that dataset, trains an
adapter, and accepts it only if a **two-axis gate** says it improved the task without losing
general quality. Config in, trained-and-gated adapter out.

This is research plus experiment, not a production build. The point is to understand how
config-driven, reproducible training pipelines are built and how they keep quality from
slipping, and to wire a small working version on top of your Part 1 output.

## The idea

The **config is the single source of truth**. Everything about a run lives in `config.yaml`:
base model, LoRA settings, the data mix (replay or not), epochs, learning rate, sequence
length, the quality guardrails, and the acceptance thresholds. Rules propose the config, a
human confirms it, and then the pipeline runs hands-off and either accepts or rejects the
result. The same pipeline runs for any task whose dataset Part 1 produced.

## Pipeline stages (map to the brief)

1. **Configure** (`propose_config.py` -> `config.yaml`): rules propose hyperparameters from the
   dataset; you review and set `confirmed: true`. Config is a defined, reviewable step.
2. **Train** (`train_from_config.py`): a LoRA/QLoRA run driven entirely by the config. `epochs`
   is an upper bound, not a target: when `train.early_stopping.enabled` is set, the run adapts its
   length per task (see stage 3).
3. **Quality guardrails** (in the config): the replay mix is the first-class lever, and adaptive
   training length is wired here too. After every epoch the run scores both axes (a clean bf16
   re-load of the saved adapter, the same measurement the gate uses, not the live 4-bit model) and
   stops as soon as a regression axis slips below its best by more than `regression_tolerance`
   (forgetting has begun) or task macro-F1 stops improving for `patience` evals, keeping the best
   adapter (`early_stopping.py`). So the run trains as long as the task needs and no further, rather
   than a hand-tuned epoch count. Smaller LR, regularization toward base, and merging remain
   documented knobs you can add as experiments.
4. **Evaluate** (`evaluate_from_config.py`): the two axes, task macro-F1/accuracy on the gold
   set, and three regression probes (sentinel, reasoning, tools).
5. **Gate** (`gate.py`): accept the adapter only if task gain >= `min_task_gain_macro_f1` AND no
   probe drops by more than `max_regression_drop`. On reject, the pipeline applies one
   configured adjustment and re-runs once.
6. **Output**: the accepted adapter plus its config and eval numbers in `runs/`, reproducible.

`pipeline.py` orchestrates 2 to 6.

## Files

```
config.yaml              the single source of truth (review + confirm before running)
propose_config.py        rules propose a config draft from the dataset (-> config.proposed.yaml)
common_p2.py             config load + path resolve + scoring helpers (no torch)
train_from_config.py     config-driven LoRA training (replay vs no-replay)
early_stopping.py        adaptive training length: two-axis early stopping callback
evaluate_from_config.py  two-axis evaluation (task gold + regression probes)
gate.py                  acceptance gate logic (pure, importable)
pipeline.py              orchestrator: base -> train -> eval -> gate -> one adjusted re-run
runs/                    adapters, result_*.json, report.json (created on first run)
kaggle_notebook/         pipeline2_kaggle.md: paste-ready cells to run the loop on a Kaggle T4
WRITEUP.md               the short write-up deliverable (fill in with your numbers)
```

It reads the Part 1 dataset from `../task-dataset-generation-pipeline1/` (set in `config.yaml`
under `data:`). Generation already happened in Part 1; Part 2 only trains and evaluates.

## Run order

```
# 0. setup (same venv as the rest of the repo) + pyyaml is required
pip install pyyaml   # if not already present

# 1. (optional) let the rules propose a config, then fold it into config.yaml
../../.venv/bin/python propose_config.py
#    review config.proposed.yaml, edit config.yaml, set confirmed: true

# 2. run the full gate loop (base -> train -> eval -> gate -> one adjusted re-run)
../../.venv/bin/python pipeline.py

# 3. the guardrail experiment: no-replay vs replay, regression axis compared
../../.venv/bin/python pipeline.py --experiment guardrails
```

You can also run any stage on its own (each is a normal CLI), which is the best way to learn
the flow: `train_from_config.py --variant replay`, `evaluate_from_config.py --name base`,
`gate.py --candidate helpdesk-resolution-replay`.

Compute: training is heavy for a 4 GB local GPU (long context + trace). Run it on a free Kaggle
T4 instead, `kaggle_notebook/pipeline2_kaggle.md` has paste-ready cells that reproduce the whole
loop (config -> train -> eval -> gate -> adjusted re-run, plus the guardrail experiment) with the
config inlined. Keep everything small, this is about getting the loop right, not scale.

## The four experiments (from the brief)

1. **Map the landscape** (research): how do Axolotl, LLaMA-Factory, torchtune, TRL, and Unsloth
   build config-driven runs, and how do they preserve quality (replay, regularization, merging,
   early stopping on a general metric)? Notes go in `WRITEUP.md`.
2. **Wire a small config-driven run**: `pipeline.py` on the Part 1 dataset.
3. **Exercise the guardrails**: `pipeline.py --experiment guardrails` trains without and with the
   replay mix and compares the regression axis, so you see the effect in your own numbers.
4. **Run the gate**: the default `pipeline.py` measures both axes against the acceptance criteria
   and lets a failed gate trigger one adjusted re-run. That loop is the heart of the pipeline.

## Reference implementations to study

- **Axolotl** (primary): one YAML drives preprocessing, training, eval, quantization, inference;
  reproducible by sharing the config. github.com/axolotl-ai-cloud/axolotl
- **LLaMA-Factory**: 100+ models, zero-code CLI + web UI, config-driven. github.com/hiyouga/LLaMA-Factory
- **torchtune**: PyTorch-native YAML recipes, full control. github.com/pytorch/torchtune
- **TRL**: the trainer library (SFT/DPO/GRPO) the others build on. huggingface.co/docs/trl
- **Unsloth**: fast single-GPU runs on Kaggle/Colab (what you already used). github.com/unslothai/unsloth

This pipeline borrows their core idea (config as the single source of truth) in a small,
readable form, and adds the two-axis gate as the accept/reject decision.

## Test the accepted adapter

After `pipeline.py` accepts an adapter, run it on gold tickets interactively:

```
../../.venv/bin/python run_trained.py
# gold index (blank = next)> 0
```

It loads the gate-accepted adapter from `runs/report.json` (or pass `--adapter <path>`), and
prints the reasoning trace + predicted vs true resolution. `--index N` for a one-shot run.
