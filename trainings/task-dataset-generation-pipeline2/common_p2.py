"""Shared helpers for the Part 2 automated training pipeline.

Config loading + path resolution (the config is the single source of truth), plus
the small scoring helpers the evaluator and gate need. Kept dependency-light so it
imports without torch.
"""

import json
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config.yaml")
LABELS_FALLBACK = ["Done", "Won't Do"]


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p):
    """Resolve a config path relative to this pipeline2 folder (absolute paths pass
    through). This is why the run works regardless of the current directory."""
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def data_path(cfg, key):
    return os.path.join(resolve(cfg["data"]["dir"]), cfg["data"][key])


def probe_path(cfg, key):
    return os.path.join(resolve(cfg["data"]["probes_dir"]), cfg["data"][key])


def out_dir(cfg):
    d = resolve(cfg.get("output_dir", "runs"))
    os.makedirs(d, exist_ok=True)
    return d


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    if path and os.path.exists(path):
        with open(path) as f:
            labs = [l.strip() for l in f if l.strip()]
            if labs:
                return labs
    return list(LABELS_FALLBACK)


def normalize(s):
    return " ".join(s.lower().split())


def c_predict_label(output, labels):
    """Read the label from a reasoning-then-answer output (tail after </think>),
    lenient but unable to inflate the score."""
    tail = output.rsplit("</think>", 1)[-1] if "</think>" in output else output
    by = {normalize(l): l for l in labels}
    lines = [x for x in tail.splitlines() if x.strip()]
    last = normalize(lines[-1]) if lines else ""
    if last in by:
        return by[last]
    low = normalize(tail)
    hits = [l for l in labels if normalize(l) in low]
    return max(hits, key=len) if hits else None


def macro_f1(gold, pred, labels):
    total = 0.0
    for l in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == l and p == l)
        fp = sum(1 for g, p in zip(gold, pred) if g != l and p == l)
        fn = sum(1 for g, p in zip(gold, pred) if g == l and p != l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        total += 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return total / len(labels)


def score_probes(raw_outputs, probes, mode="any"):
    """Fraction of probes hit. mode='any': output contains any expected answer;
    mode='all': output contains every expected substring (used for tool calls)."""
    ok = 0
    for o, p in zip(raw_outputs, probes):
        low = o.lower()
        want = [a.lower() for a in p["answers"]]
        ok += (all(a in low for a in want) if mode == "all" else any(a in low for a in want))
    return ok / len(probes) if probes else 0.0
