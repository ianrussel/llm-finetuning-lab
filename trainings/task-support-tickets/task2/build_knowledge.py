"""Build a small held-out knowledge-absorption Q&A set from the Jira corpus (the third axis).

Goal (a touch, not the main gate): does fine-tuning absorb DAOS *domain* knowledge? We ask "which
component does this issue affect?" - a domain fact (Control Plane, Erasure Code, Rebuild, ...) that
is distinct from the task label (issue type), so it does not just leak the task. Built from issues
that are NOT in the training or gold sets (decontaminated by the exact issue text the trainer saw),
and scored closed-book with no retrieval (knowledge.py). If the fine-tuned model names the right
component more often than base, some domain knowledge stuck.

Run from this folder (CPU):
    ../../../.venv/bin/python build_knowledge.py
Writes data/knowledge_probes.jsonl. Keep it small; this is a supplementary signal.
"""

import json
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SRC = os.path.join(DATA, "jira_issues.jsonl")
TRAIN_FILES = ["train_synth.jsonl", "train_mix.jsonl", "gold.jsonl"]  # decontaminate against all of these

SEED = 0
N_PROBES = 40
PER_COMPONENT_CAP = 6           # spread across components, do not let one dominate
DESC_MAX_CHARS = 1500           # must match prepare_data.py so decontamination texts line up
Q_DESC_CHARS = 300             # shorter snippet in the actual question (closed-book, keep compact)

INSTRUCTION = ("You are a DAOS engineer. Read the issue and name the DAOS subsystem/component it "
               "affects. Answer with just the component name.")


def norm(s):
    return " ".join(str(s).lower().split())


def issue_text(summary, desc, cap):
    return "\n".join(x for x in [str(summary).strip(), str(desc or "")[:cap].strip()] if x)


def main():
    random.seed(SEED)
    if not os.path.exists(SRC):
        raise SystemExit(f"no {SRC}; run extract_jira.sh + prepare_data.py first")

    # texts the trainer/gold actually saw (the user message == the issue text from prepare_data)
    seen = set()
    for fn in TRAIN_FILES:
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
                seen.add(norm(user))
    print(f"[know] decontamination set: {len(seen)} training/gold issue texts")

    # candidate knowledge issues: have component(s), not in training/gold
    by_comp = defaultdict(list)
    n_total = n_with_comp = 0
    for line in open(SRC):
        if not line.strip():
            continue
        f = (json.loads(line).get("fields") or {})
        n_total += 1
        comps = [c.get("name") for c in (f.get("components") or []) if c.get("name")]
        summary = f.get("summary")
        if not summary or not comps:
            continue
        n_with_comp += 1
        if norm(issue_text(summary, f.get("description"), DESC_MAX_CHARS)) in seen:
            continue  # held-out only
        q_text = issue_text(summary, f.get("description"), Q_DESC_CHARS)
        # index by the first (primary) component for balanced sampling
        by_comp[comps[0]].append((q_text, comps))

    # sample up to PER_COMPONENT_CAP per component until we hit N_PROBES, favouring variety
    picked, seen_q = [], set()
    comps_sorted = sorted(by_comp, key=lambda c: -len(by_comp[c]))
    for c in comps_sorted:
        random.shuffle(by_comp[c])
    round_i = 0
    while len(picked) < N_PROBES and any(by_comp.values()):
        for c in comps_sorted:
            if round_i < len(by_comp[c]) and round_i < PER_COMPONENT_CAP:
                q_text, comps = by_comp[c][round_i]
                k = norm(q_text)
                if k not in seen_q:
                    seen_q.add(k)
                    picked.append((q_text, comps))
                    if len(picked) >= N_PROBES:
                        break
        round_i += 1
        if round_i > PER_COMPONENT_CAP:
            break

    if not picked:
        raise SystemExit("no held-out issues with components found; check the corpus / decontamination")

    with open(os.path.join(DATA, "knowledge_probes.jsonl"), "w") as fout:
        for q_text, comps in picked:
            q = f"{INSTRUCTION}\n\nIssue:\n{q_text}"
            answers = sorted({norm(c) for c in comps})   # any correct component counts (score 'any')
            fout.write(json.dumps({"question": q, "answers": answers}) + "\n")

    dist = Counter(comps[0] for _, comps in picked)
    print(f"[know] scanned {n_total} issues, {n_with_comp} with components")
    print(f"[know] wrote {len(picked)} held-out knowledge probes -> data/knowledge_probes.jsonl")
    print(f"[know] primary-component spread: {dict(dist.most_common())}")
    print("[know] next: score with knowledge.py --adapter <accepted-adapter-dir>")


if __name__ == "__main__":
    main()
