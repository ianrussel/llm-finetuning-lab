"""Adaptive training length (the self-adjusting part of the guardrails).

The aim is not a hand-tuned epoch count. Each task should train as long as it needs and no further,
so this callback evaluates BOTH axes after every epoch, keeps the best checkpoint, and stops when a
regression axis slips below its best or task macro-F1 plateaus for `patience` evals. `train.epochs`
is only an upper bound; the actual length adapts per task. The final gate (gate.py, vs the base
model) is unchanged; this just hands it the adapter from the run's best epoch.

Two eval modes (config: `train.early_stopping.eval_mode`), because the in-loop measurement competes
with the trainer for VRAM:

  - "clean" (default): save the adapter and re-load it on a fresh bf16 base (via
    evaluate_from_config._load), scored on the full gold set + probes. This matches the gate exactly,
    but loads a SECOND model copy, which does not fit beside a 3.8B/4B trainer on a free T4.
  - "resident": score the model that is ALREADY in memory (the 4-bit QLoRA trainer + its adapter),
    no second copy. Lighter, so adaptive length works on 4B+. It is the as-trained (4-bit) model, so
    its absolute numbers can differ a little from the bf16 gate; treat it as a best-epoch / plateau /
    regression proxy, with the bf16 gate as the final arbiter.

Defaults live in the config under `train.early_stopping`; set `enabled: false` to train fixed epochs.
"""

import os

import torch
from transformers import TrainerCallback

import common_p2 as c
import evaluate_from_config as ef

# probe axis -> (config filename key, scoring mode, max_new_tokens); mirrors evaluate_from_config.
_PROBE_AXES = [
    ("sentinel", "sentinel", "any", 128),    # 128 so a reasoning model's answer survives the think block
    ("reasoning", "reasoning_probes", "any", 256),
    ("tools", "tool_probes", "all", 128),
]


class TwoAxisEarlyStopping(TrainerCallback):
    """Evaluate task + regression axes each epoch; keep the best adapter and stop when it is time.
    Writes the best adapter into `out_dir`."""

    def __init__(self, cfg, tok, out_dir, es):
        self.tok = tok
        self.out = out_dir
        self.base_model = cfg["base_model"]
        self.mode = es.get("eval_mode", "clean")     # "clean" (bf16 reload) or "resident" (no 2nd copy)
        self.patience = es.get("patience", 2)
        self.min_delta = es.get("min_task_delta", 0.005)
        self.tol = es.get("regression_tolerance", cfg["acceptance"]["max_regression_drop"])
        self.batch = es.get("eval_batch", 8)
        limit = es.get("eval_task_limit", 0)          # 0 = full gold

        self.labels = c.load_labels(c.data_path(cfg, "labels"))
        gold = c.read_jsonl(c.data_path(cfg, "gold"))
        self.gold = gold[:limit] if limit else gold

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

    def _run(self, model, tok):
        raw = ef._gen(model, tok, [r["messages"] for r in self.gold], 384, self.batch)
        task = c.macro_f1([r["label"] for r in self.gold],
                          [c.c_predict_label(o, self.labels) for o in raw], self.labels)
        scores = {}
        for axis, prompts, rows, mode, max_new in self.probes:
            scores[axis] = c.score_probes(ef._gen(model, tok, prompts, max_new, self.batch), rows, mode)
        return task, scores

    def _score_clean(self, adapter_dir):
        """Re-load the saved adapter on a fresh bf16 base, exactly like the gate. Second copy."""
        model, tok = ef._load(self.base_model, adapter_dir)
        try:
            return self._run(model, tok)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _score_resident(self, model):
        """Score the model already in memory (no second copy). Toggle eval/cache/checkpointing for
        generation, then restore the training state so training continues unchanged."""
        prev_pad = self.tok.padding_side
        prev_cache = getattr(model.config, "use_cache", True)
        was_training = model.training
        was_gc = bool(getattr(model, "is_gradient_checkpointing", False))
        self.tok.padding_side = "left"
        model.config.use_cache = True
        model.eval()
        if was_gc:
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
        try:
            return self._run(model, self.tok)
        finally:
            self.tok.padding_side = prev_pad
            model.config.use_cache = prev_cache
            if was_gc:
                try:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except Exception:
                    pass
            if was_training:
                model.train()

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        epoch = round(state.epoch or 0)

        try:
            if self.mode == "resident":
                task, scores = self._score_resident(model)
            else:
                tmp = os.path.join(self.out, "_epoch_eval")
                model.save_pretrained(tmp)
                task, scores = self._score_clean(tmp)
        except Exception as e:  # eval should not crash the run; degrade and keep training
            print(f"[early-stop] epoch {epoch}: eval skipped ({type(e).__name__}: {e})")
            return control

        reg_drop = 0.0
        for axis, value in scores.items():
            reg_drop = max(reg_drop, self.peaks.get(axis, value) - value)
        reg_ok = reg_drop <= self.tol + 1e-9   # epsilon: a drop of exactly the tolerance is tolerated

        self.history.append({"epoch": epoch, "task": round(task, 4),
                             **{k: round(v, 4) for k, v in scores.items()}})
        print(f"[early-stop] epoch {epoch} ({self.mode}): task={task:.3f} "
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
