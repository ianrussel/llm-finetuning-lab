"""Orchestrator for the open-ended (generative) task: config in, trained-and-gated adapter out.

Same loop as the classification pipeline (base -> train -> evaluate -> gate -> one adjusted
re-run), and the same multi-seed median, but the task axis is the LLM-judge win-rate of the
candidate over the base instead of a macro-F1 gain:

  1. evaluate the BASE model: generate its answers on the held-out tickets + score regression probes.
  2. per seed: train a candidate, generate its answers + probes.
  3. judge: a stronger model compares base vs candidate answers per ticket -> win-rate.
  4. gate: accept if median win-rate >= min_win_rate AND no probe regressed too much.
  5. if the median rejects, apply one configured adjustment and re-run once.

Run from this folder (GPU; heavy, the judge is a 7B):
    ../../.venv/bin/python pipeline_gen.py
"""

import argparse
import json
import os
import statistics

import torch

import common_p2 as c
from train_from_config import run_training
from evaluate_gen import evaluate_gen
import judge as J
import gate_gen


def _seeds(cfg):
    return cfg["train"].get("seeds", [cfg["train"].get("seed", 0)])


def _base_results(cfg, refresh=False):
    path = os.path.join(c.out_dir(cfg), "result_base.json")
    if os.path.exists(path) and not refresh:
        print("[pipeline-gen] using cached base result:", path)
        with open(path) as f:
            return json.load(f)
    return evaluate_gen(cfg, "base", adapter=None)


def _judge_winrates(cfg, base, cand_results):
    """Load the judge once and score every candidate's answers vs base (win-rate per candidate)."""
    jm = cfg["judge"]["model"]
    jb = cfg["judge"].get("batch", 4)
    print(f"[pipeline-gen] judging {len(cand_results)} candidate(s) with {jm}")
    mt = J._load_judge(jm)
    try:
        wrs = []
        for res in cand_results:
            r = J.judge_pairwise(jm, base["tickets"], base["answers"], res["answers"], batch=jb, model_tok=mt)
            wrs.append(r["win_rate"])
    finally:
        del mt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return wrs


def _run_stage(cfg, base, base_name, data_key, acceptance, lr=None, epochs=None):
    seeds = _seeds(cfg)
    runs = []
    for s in seeds:
        name_s = f"{base_name}-s{s}" if len(seeds) > 1 else base_name
        adapter = run_training(cfg, name_s, c.data_path(cfg, data_key), lr=lr, epochs=epochs, seed=s)
        res = evaluate_gen(cfg, name_s, adapter=adapter)
        runs.append({"seed": s, "run": name_s, "adapter": adapter, "res": res})
    win_rates = _judge_winrates(cfg, base, [r["res"] for r in runs])
    for r, wr in zip(runs, win_rates):
        r["verdict"] = gate_gen.decide(base, r["res"], wr, acceptance)
        _print_verdict(r["run"], r["verdict"])
    return runs


def _aggregate(runs, acceptance):
    wrs = [r["verdict"]["win_rate"] for r in runs]
    drops = [r["verdict"]["worst_regression_drop"] for r in runs]
    med_wr = statistics.median(wrs)
    med_drop = statistics.median(drops)
    accept = med_wr >= acceptance["min_win_rate"] and med_drop <= acceptance["max_regression_drop"]
    passing = [r for r in runs if r["verdict"]["accept"]]
    chosen = max(passing or runs, key=lambda r: r["verdict"]["win_rate"])
    return {"accept": accept, "median_win_rate": round(med_wr, 4),
            "median_worst_drop": round(med_drop, 4),
            "win_rate_spread": [round(min(wrs), 4), round(max(wrs), 4)],
            "seeds_passed": f"{len(passing)}/{len(runs)}",
            "chosen_run": chosen["run"], "chosen_adapter": chosen["adapter"]}


def _print_verdict(name, v):
    print(f"\n[gate] {name}: {'ACCEPT' if v['accept'] else 'REJECT'}")
    for r in v["reasons"]:
        print("   -", r)
    print("   per-probe deltas:", v["regressions"])


def _print_aggregate(name, agg):
    print(f"\n[gate] {name} across seeds: {'ACCEPT' if agg['accept'] else 'REJECT'} "
          f"(median win-rate={agg['median_win_rate']:.3f}, median worst_drop={agg['median_worst_drop']:.3f}, "
          f"seeds_passed={agg['seeds_passed']}, win-rate range={agg['win_rate_spread']}) "
          f"-> chosen {agg['chosen_run']}")


def _slim_run(r):
    return {"seed": r["seed"], "run": r["run"], "adapter": r["adapter"], "verdict": r["verdict"]}


def gate_loop(cfg, refresh_base=False):
    base = _base_results(cfg, refresh=refresh_base)
    acceptance = cfg["acceptance"]
    print(f"[pipeline-gen] seeds = {_seeds(cfg)} (gate decides on the median win-rate)")
    attempts = []

    replay = cfg["guardrails"]["replay_mix"]
    data_key = "task_train" if replay else "task_train_noreplay"
    name1 = f"{cfg['name']}-{'replay' if replay else 'noreplay'}"
    runs1 = _run_stage(cfg, base, name1, data_key, acceptance)
    agg1 = _aggregate(runs1, acceptance)
    _print_aggregate(name1, agg1)
    attempts.append({"stage": name1, "seeds": [_slim_run(r) for r in runs1], "aggregate": agg1})

    accepted = agg1["chosen_adapter"] if agg1["accept"] else None

    if not agg1["accept"]:
        adj = cfg.get("adjust_on_fail", {})
        lr = cfg["train"]["learning_rate"] * adj.get("lower_lr_factor", 1.0)
        epochs = adj.get("reduce_epochs_to", cfg["train"]["epochs"])
        data_key2 = "task_train" if adj.get("force_replay", True) else data_key
        name2 = f"{cfg['name']}-adj"
        print(f"\n[pipeline-gen] gate rejected; one adjusted re-run (lr={lr}, epochs={epochs})")
        runs2 = _run_stage(cfg, base, name2, data_key2, acceptance, lr=lr, epochs=epochs)
        agg2 = _aggregate(runs2, acceptance)
        _print_aggregate(name2, agg2)
        attempts.append({"stage": name2, "seeds": [_slim_run(r) for r in runs2], "aggregate": agg2})
        if agg2["accept"]:
            accepted = agg2["chosen_adapter"]

    report = {"config": cfg, "attempts": attempts, "accepted_adapter": accepted}
    path = os.path.join(c.out_dir(cfg), "report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[pipeline-gen] {'ACCEPTED ' + accepted if accepted else 'NO ADAPTER ACCEPTED'}")
    print(f"[pipeline-gen] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-base", action="store_true")
    args = ap.parse_args()
    cfg = c.load_config()
    if not cfg.get("confirmed"):
        raise SystemExit("config not confirmed. Review config.yaml, set confirmed: true, re-run.")
    gate_loop(cfg, refresh_base=args.refresh_base)


if __name__ == "__main__":
    main()
