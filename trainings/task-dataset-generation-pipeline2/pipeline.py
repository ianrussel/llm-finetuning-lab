"""The automated training pipeline orchestrator (Part 2, the heart of the module).

Config in -> trained, gated adapter out. It runs the loop the brief describes:

  1. evaluate the BASE model on both axes (cached in runs/result_base.json).
  2. train a candidate adapter, driven entirely by config.yaml.
  3. evaluate the candidate on both axes.
  4. GATE: accept only if task improves enough AND nothing regresses too much.
  5. if the gate REJECTS, apply ONE configured adjustment (adjust_on_fail) and re-run
     once. That adjusted re-run is what makes the automation trustworthy.

It writes runs/report.json with the config snapshot, every attempt, and the verdict,
so the accepted adapter is reproducible.

Modes:
  (default)              the gate loop above.
  --experiment guardrails   train once WITHOUT replay and once WITH replay, evaluate
                            both, and print the regression-axis comparison so you see
                            the quality-preservation effect with your own numbers.

Run from this folder (GPU; this is heavy, run it yourself):
    ../../.venv/bin/python pipeline.py
    ../../.venv/bin/python pipeline.py --experiment guardrails
"""

import argparse
import json
import os

import common_p2 as c
from train_from_config import run_training
from evaluate_from_config import evaluate
import gate


def _base_results(cfg, refresh=False):
    """Evaluate the base model once, then reuse the cached result."""
    path = os.path.join(c.out_dir(cfg), "result_base.json")
    if os.path.exists(path) and not refresh:
        print("[pipeline] using cached base result:", path)
        with open(path) as f:
            return json.load(f)
    return evaluate(cfg, "base", adapter=None)


def _train_eval(cfg, run_name, data_key, lr=None, epochs=None):
    data = c.data_path(cfg, data_key)
    adapter = run_training(cfg, run_name, data, lr=lr, epochs=epochs)
    res = evaluate(cfg, run_name, adapter=adapter)
    return adapter, res


def gate_loop(cfg, refresh_base=False):
    base = _base_results(cfg, refresh=refresh_base)
    acceptance = cfg["acceptance"]
    attempts = []

    # Attempt 1: the configured run (replay or not, per guardrails).
    replay = cfg["guardrails"]["replay_mix"]
    data_key = "task_train" if replay else "task_train_noreplay"
    name1 = f"{cfg['name']}-{'replay' if replay else 'noreplay'}"
    adapter1, res1 = _train_eval(cfg, name1, data_key)
    v1 = gate.decide(base, res1, acceptance)
    attempts.append({"run": name1, "adapter": adapter1, "verdict": v1})
    _print_verdict(name1, v1)

    accepted = adapter1 if v1["accept"] else None

    # Attempt 2: one adjusted re-run if the gate rejected.
    if not v1["accept"]:
        adj = cfg.get("adjust_on_fail", {})
        lr = cfg["train"]["learning_rate"] * adj.get("lower_lr_factor", 1.0)
        epochs = adj.get("reduce_epochs_to", cfg["train"]["epochs"])
        data_key2 = "task_train" if adj.get("force_replay", True) else data_key
        name2 = f"{cfg['name']}-adj"
        print(f"\n[pipeline] gate rejected; one adjusted re-run "
              f"(force_replay={adj.get('force_replay', True)}, lr={lr}, epochs={epochs})")
        adapter2, res2 = _train_eval(cfg, name2, data_key2, lr=lr, epochs=epochs)
        v2 = gate.decide(base, res2, acceptance)
        attempts.append({"run": name2, "adapter": adapter2, "verdict": v2})
        _print_verdict(name2, v2)
        if v2["accept"]:
            accepted = adapter2

    report = {"config": cfg, "base": _slim(base), "attempts": attempts,
              "accepted_adapter": accepted}
    path = os.path.join(c.out_dir(cfg), "report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[pipeline] {'ACCEPTED ' + accepted if accepted else 'NO ADAPTER ACCEPTED'}")
    print(f"[pipeline] wrote {path}")


def guardrails_experiment(cfg, refresh_base=False):
    """Experiment #3: train without and with the replay mix, compare the regression axis."""
    base = _base_results(cfg, refresh=refresh_base)
    _, res_no = _train_eval(cfg, f"{cfg['name']}-noreplay", "task_train_noreplay")
    _, res_yes = _train_eval(cfg, f"{cfg['name']}-replay", "task_train")

    def line(tag, r):
        t = r["task"]
        probes = {k: (r[k]["score"] if r.get(k) else None) for k in ("sentinel", "reasoning", "tools")}
        return (f"  {tag:<9} macroF1={t['macro_f1']:.3f}  "
                + "  ".join(f"{k}={v:.3f}" if v is not None else f"{k}=n/a" for k, v in probes.items()))

    print("\n=== guardrail experiment: replay mix effect on the regression axis ===")
    print(line("base", base))
    print(line("no-replay", res_no))
    print(line("replay", res_yes))
    print("Expectation: replay holds the sentinel/reasoning/tool scores up vs no-replay, "
          "at a small or no cost to task macro-F1. Read your own numbers above.")
    path = os.path.join(c.out_dir(cfg), "guardrail_experiment.json")
    with open(path, "w") as f:
        json.dump({"base": _slim(base), "noreplay": _slim(res_no), "replay": _slim(res_yes)},
                  f, indent=2)
    print(f"wrote {path}")


def _slim(r):
    return {"task": r["task"], "sentinel": r.get("sentinel"),
            "reasoning": r.get("reasoning"), "tools": r.get("tools")}


def _print_verdict(name, v):
    print(f"\n[gate] {name}: {'ACCEPT' if v['accept'] else 'REJECT'}")
    for r in v["reasons"]:
        print("   -", r)
    print("   per-probe deltas:", v["regressions"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["guardrails"], default=None)
    ap.add_argument("--refresh-base", action="store_true", help="re-evaluate the base model")
    args = ap.parse_args()

    cfg = c.load_config()
    if not cfg.get("confirmed"):
        raise SystemExit("config not confirmed. Review config.yaml, set confirmed: true, re-run.")

    if args.experiment == "guardrails":
        guardrails_experiment(cfg, refresh_base=args.refresh_base)
    else:
        gate_loop(cfg, refresh_base=args.refresh_base)


if __name__ == "__main__":
    main()
