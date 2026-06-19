"""Adaptive training length (the self-adjusting part of the guardrails).

The aim is not a hand-tuned epoch count. Each task should train as long as it needs and no
further, so this callback evaluates BOTH axes after every epoch and:

  - keeps the best checkpoint (highest task macro-F1 among epochs that did not regress), and
  - stops the run when a regression axis falls more than `regression_tolerance` below its best
    seen so far (forgetting has begun), or task macro-F1 has not improved by `min_task_delta`
    for `patience` consecutive evals.

So `train.epochs` in the config is only an upper bound on the budget; the actual length adapts
per task. The final gate (candidate vs the base model, in gate.py) is unchanged. This just hands
it the adapter from the epoch where the run was at its best, instead of whatever the last epoch
happened to be.

IMPORTANT: the per-epoch signal is measured the SAME way the gate measures it. Each epoch the
current adapter is saved and re-loaded on the clean bf16 base (via evaluate_from_config._load),
then scored on the full gold set + probes. An earlier version generated from the live 4-bit
QLoRA model over a gold subset; that signal was noisy and biased (strict tool-call scores and the
binary-task macro-F1 swung wildly), so it stopped good runs at epoch 1. Evaluating the saved
adapter cleanly is slower per epoch but makes the stop/selection decisions trustworthy.

Defaults live in the config under `train.early_stopping` and the user can override them or set
`enabled: false` to train a fixed `epochs` the old way.
"""

import os

import torch
from transformers import TrainerCallback

import common_p2 as c
import evaluate_from_config as ef


# probe axis -> (config filename key, scoring mode, max_new_tokens); mirrors evaluate_from_config.
_PROBE_AXES = [
    ("sentinel", "sentinel", "any", 32),
    ("reasoning", "reasoning_probes", "any", 256),
    ("tools", "tool_probes", "all", 128),
]


class TwoAxisEarlyStopping(TrainerCallback):
    """Evaluate task + regression axes each epoch (clean bf16 re-load, like the gate); keep the
    best adapter and stop when it is time. Writes the best adapter into `out_dir`."""

    def __init__(self, cfg, tok, out_dir, es):
        self.tok = tok
        self.out = out_dir
        self.base_model = cfg["base_model"]
        self.patience = es.get("patience", 2)
        self.min_delta = es.get("min_task_delta", 0.005)
        self.tol = es.get("regression_tolerance", cfg["acceptance"]["max_regression_drop"])
        self.batch = es.get("eval_batch", 8)
        limit = es.get("eval_task_limit", 0)  # 0 = full gold, to match the gate exactly

        self.labels = c.load_labels(c.data_path(cfg, "labels"))
        gold = c.read_jsonl(c.data_path(cfg, "gold"))
        self.gold = gold[:limit] if limit else gold

        # Pre-build probe prompts once so each epoch's eval is just generation + scoring.
        self.probes = []
        for axis, fname_key, mode, max_new in _PROBE_AXES:
            path = c.probe_path(cfg, fname_key)
            if os.path.exists(path):
                rows = c.read_jsonl(path)
                prompts = [[{"role": "user", "content": p["question"]}] for p in rows]
                self.probes.append((axis, prompts, rows, mode, max_new))

        self.history = []
        self.peaks = {}
        self.best_task = float("-inf")
        self.best_epoch = None
        self.bad = 0
        self.stop_reason = None

    def _score(self, adapter_dir):
        """Load the saved adapter on a clean bf16 base and score both axes, identical to the gate's
        evaluate(). A fresh eval model is loaded and freed each epoch so it never sees the noisy
        4-bit training state."""
        model, tok = ef._load(self.base_model, adapter_dir)
        try:
            raw = ef._gen(model, tok, [r["messages"] for r in self.gold], 384, self.batch)
            preds = [c.c_predict_label(o, self.labels) for o in raw]
            task = c.macro_f1([r["label"] for r in self.gold], preds, self.labels)
            scores = {}
            for axis, prompts, rows, mode, max_new in self.probes:
                out = ef._gen(model, tok, prompts, max_new, self.batch)
                scores[axis] = c.score_probes(out, rows, mode)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return task, scores

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        epoch = round(state.epoch or 0)

        tmp = os.path.join(self.out, "_epoch_eval")
        try:
            model.save_pretrained(tmp)          # snapshot this epoch's adapter, then score it clean
            task, scores = self._score(tmp)
        except Exception as e:  # eval should not crash the run; degrade and keep training
            print(f"[early-stop] epoch {epoch}: eval skipped ({type(e).__name__}: {e})")
            return control

        # Worst drop of any regression axis from its best value seen so far this run.
        reg_drop = 0.0
        for axis, value in scores.items():
            reg_drop = max(reg_drop, self.peaks.get(axis, value) - value)
        reg_ok = reg_drop <= self.tol

        self.history.append({"epoch": epoch, "task": round(task, 4),
                             **{k: round(v, 4) for k, v in scores.items()}})
        print(f"[early-stop] epoch {epoch}: task={task:.3f} "
              + " ".join(f"{k}={v:.3f}" for k, v in scores.items())
              + f" | reg_drop_from_peak={reg_drop:.3f} (tol {self.tol:.3f})")

        improved = task > self.best_task + self.min_delta
        if reg_ok and improved:
            self.best_task = task
            self.best_epoch = epoch
            self.bad = 0
            model.save_pretrained(self.out)
            self.tok.save_pretrained(self.out)
            print(f"[early-stop] new best task={task:.3f}; kept adapter from epoch {epoch}")
        else:
            self.bad += 1

        # Update peaks after measuring the drop, so a one-epoch dip is judged against the prior best.
        for axis, value in scores.items():
            self.peaks[axis] = max(self.peaks.get(axis, value), value)

        if not reg_ok:
            self.stop_reason = (f"regression axis fell {reg_drop:.3f} below its best "
                                f"(> tol {self.tol:.3f}) at epoch {epoch}")
        elif self.bad >= self.patience:
            self.stop_reason = (f"task macro-F1 did not improve by {self.min_delta} for "
                                f"{self.bad} eval(s) by epoch {epoch}")

        if self.stop_reason:
            print(f"[early-stop] stopping: {self.stop_reason}; best epoch = {self.best_epoch}")
            control.should_training_stop = True
        return control
