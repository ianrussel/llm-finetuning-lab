"""Stage 3.1 (Configure): rules propose a training config from the dataset; human confirms.

This is the "rules now, agent later" layer from the brief: simple, transparent rules
look at the generated dataset and propose hyperparameters, writing a draft config the
human reviews. It writes config.proposed.yaml (it never clobbers your config.yaml), so
you diff it, copy what you want into config.yaml, and set confirmed: true.

Run from this folder:
    ../../.venv/bin/python propose_config.py
"""

import os

import yaml

import common_p2 as c


def propose(cfg):
    """Apply simple, explained rules over the dataset to fill in hyperparameters."""
    notes = []
    train_file = c.data_path(cfg, "task_train")
    n = len(c.read_jsonl(train_file)) if os.path.exists(train_file) else 0

    # epochs is now the UPPER BOUND on the budget; early stopping picks the actual length per
    # task. Rules set a generous starting cap (smaller sets get more passes, LIMA-style); the
    # adaptive stop keeps a run from overshooting it.
    epochs = 6 if n < 500 else (4 if n < 2000 else 2)
    notes.append(f"{n} training rows -> max epochs={epochs} (upper bound; early stopping adapts the real length)")

    # class balance in gold drives the metric choice (macro-F1) and a reminder.
    gold_file = c.data_path(cfg, "gold")
    if os.path.exists(gold_file):
        gold = c.read_jsonl(gold_file)
        from collections import Counter
        bal = Counter(r.get("label") for r in gold)
        notes.append(f"gold balance {dict(bal)} -> macro-F1 is the headline metric")

    # Adaptive training length: propose the starting budget plus sensible early-stopping
    # defaults. regression_tolerance mirrors the acceptance gate so the two agree by default.
    early_stopping = {
        "enabled": True,
        "patience": 2,
        "min_task_delta": 0.005,
        "regression_tolerance": cfg.get("acceptance", {}).get("max_regression_drop", 0.05),
        "eval_task_limit": 0,  # 0 = full gold per epoch, so the stop signal matches the gate
        "eval_batch": 8,
    }
    notes.append("early_stopping.enabled=true (train as long as the task needs, then stop)")

    cfg = dict(cfg)
    cfg["train"] = {**cfg["train"], "epochs": epochs,
                    "early_stopping": {**cfg["train"].get("early_stopping", {}), **early_stopping}}
    cfg["guardrails"] = {**cfg["guardrails"], "replay_mix": True}
    notes.append("replay_mix=true proposed (preserve general + tool ability)")
    cfg["confirmed"] = False
    return cfg, notes


def main():
    cfg = c.load_config()
    proposed, notes = propose(cfg)
    out = os.path.join(c.HERE, "config.proposed.yaml")
    with open(out, "w") as f:
        f.write("# Proposed by propose_config.py (rules over the dataset). Review, then\n")
        f.write("# copy the values you accept into config.yaml and set confirmed: true.\n")
        yaml.safe_dump(proposed, f, sort_keys=False)

    print("proposal rationale:")
    for nt in notes:
        print("  -", nt)
    print(f"\nwrote {out} (confirmed=false).")
    print("ACTION: review it, fold accepted values into config.yaml, set confirmed: true, "
          "then run pipeline.py.")


if __name__ == "__main__":
    main()
