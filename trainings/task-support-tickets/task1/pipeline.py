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
import statistics

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


def _train_eval(cfg, run_name, data_key, lr=None, epochs=None, seed=0):
    data = c.data_path(cfg, data_key)
    adapter = run_training(cfg, run_name, data, lr=lr, epochs=epochs, seed=seed)
    res = evaluate(cfg, run_name, adapter=adapter)
    return adapter, res


def _seeds(cfg):
    return cfg["train"].get("seeds", [cfg["train"].get("seed", 0)])


def _run_stage(cfg, base, base_name, data_key, acceptance, lr=None, epochs=None):
    """Train + evaluate + gate once per seed. Returns the list of per-seed run dicts."""
    seeds = _seeds(cfg)
    runs = []
    for s in seeds:
        name_s = f"{base_name}-s{s}" if len(seeds) > 1 else base_name
        adapter, res = _train_eval(cfg, name_s, data_key, lr=lr, epochs=epochs, seed=s)
        v = gate.decide(base, res, acceptance)
        _print_verdict(name_s, v)
        runs.append({"seed": s, "run": name_s, "adapter": adapter, "verdict": v})
    return runs


def _aggregate(runs, acceptance):
    """Decide on the MEDIAN across seeds, so one noisy run cannot flip accept/reject. Ship the
    best adapter among the seeds that individually pass (else the best overall)."""
    gains = [r["verdict"]["task_gain"] for r in runs]
    drops = [r["verdict"]["worst_regression_drop"] for r in runs]
    med_gain = statistics.median(gains)
    med_drop = statistics.median(drops)
    accept = (med_gain >= acceptance["min_task_gain_macro_f1"]
              and med_drop <= acceptance["max_regression_drop"])
    passing = [r for r in runs if r["verdict"]["accept"]]
    chosen = max(passing or runs, key=lambda r: r["verdict"]["task_gain"])
    return {
        "accept": accept,
        "median_task_gain": round(med_gain, 4),
        "median_worst_drop": round(med_drop, 4),
        "task_gain_spread": [round(min(gains), 4), round(max(gains), 4)],
        "seeds_passed": f"{len(passing)}/{len(runs)}",
        "chosen_run": chosen["run"],
        "chosen_adapter": chosen["adapter"],
    }


def _print_aggregate(name, agg):
    print(f"\n[gate] {name} across seeds: {'ACCEPT' if agg['accept'] else 'REJECT'} "
          f"(median task_gain={agg['median_task_gain']:+.3f}, median worst_drop={agg['median_worst_drop']:.3f}, "
          f"seeds_passed={agg['seeds_passed']}, task_gain range={agg['task_gain_spread']})")
    print(f"   chosen adapter: {agg['chosen_run']}")


def gate_loop(cfg, refresh_base=False):
    base = _base_results(cfg, refresh=refresh_base)
    acceptance = cfg["acceptance"]
    seeds = _seeds(cfg)
    print(f"[pipeline] seeds = {seeds} (gate decides on the median across them)")
    attempts = []

    # Attempt 1: the configured run (replay or not, per guardrails), once per seed.
    replay = cfg["guardrails"]["replay_mix"]
    data_key = "task_train" if replay else "task_train_noreplay"
    name1 = f"{cfg['name']}-{'replay' if replay else 'noreplay'}"
    runs1 = _run_stage(cfg, base, name1, data_key, acceptance)
    agg1 = _aggregate(runs1, acceptance)
    _print_aggregate(name1, agg1)
    attempts.append({"stage": name1, "seeds": [_slim_run(r) for r in runs1], "aggregate": agg1})

    accepted = agg1["chosen_adapter"] if agg1["accept"] else None

    # Attempt 2: one adjusted re-run if the gate rejected, also across seeds.
    if not agg1["accept"]:
        adj = cfg.get("adjust_on_fail", {})
        lr = cfg["train"]["learning_rate"] * adj.get("lower_lr_factor", 1.0)
        epochs = adj.get("reduce_epochs_to", cfg["train"]["epochs"])
        data_key2 = "task_train" if adj.get("force_replay", True) else data_key
        name2 = f"{cfg['name']}-adj"
        print(f"\n[pipeline] gate rejected; one adjusted re-run "
              f"(force_replay={adj.get('force_replay', True)}, lr={lr}, epochs={epochs})")
        runs2 = _run_stage(cfg, base, name2, data_key2, acceptance, lr=lr, epochs=epochs)
        agg2 = _aggregate(runs2, acceptance)
        _print_aggregate(name2, agg2)
        attempts.append({"stage": name2, "seeds": [_slim_run(r) for r in runs2], "aggregate": agg2})
        if agg2["accept"]:
            accepted = agg2["chosen_adapter"]

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


def _slim_run(r):
    return {"seed": r["seed"], "run": r["run"], "adapter": r["adapter"], "verdict": r["verdict"]}


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
