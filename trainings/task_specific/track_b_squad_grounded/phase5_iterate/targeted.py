"""Phase 5, step 2: generate TARGETED grounded QA for the named failure mode.

error_analysis.py says which behaviour is weak (hallucinating on unanswerables,
or missing answerable questions). This generates more of exactly that kind from
fresh passages, instead of another undirected pass. It reuses the Phase 2
document-grounded generators, just weighted toward the targeted mode(s).

Reads  : data/error_analysis.json (targets), data/passages.jsonl
Writes : data/gen_targeted.jsonl  (same shape as gen_qa.jsonl -> build_v2.py)

Run from the track_b_squad_grounded folder (Ollama up):
    ../../../.venv/bin/python phase5_iterate/targeted.py
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2_synthetic"))
import common
import qgen   # reuse gen_answerable / gen_unanswerable

OUT_FILE = f"{common.DATA}/gen_targeted.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", type=int, default=200,
                    help="how many fresh passages to generate from")
    ap.add_argument("--per-passage", type=int, default=3,
                    help="targeted items to ask for, per passage, for each weak mode")
    args = ap.parse_args()

    qgen.sdg.preflight()
    ea_path = f"{common.DATA}/error_analysis.json"
    if not os.path.exists(ea_path):
        raise SystemExit(f"no {ea_path}; run phase5_iterate/error_analysis.py first.")
    targets = json.load(open(ea_path))["targets"]
    if not targets:
        raise SystemExit("error_analysis found no weak mode over its bar. Stop iterating.")
    modes = {t["mode"] for t in targets}

    pool = common.read_jsonl(f"{common.DATA}/passages.jsonl")[:args.passages]
    written = 0
    with open(OUT_FILE, "w") as f:
        for i, p in enumerate(pool, 1):
            ctx, title = p["context"], p["title"]
            if "answerable" in modes:
                for q, a in qgen.gen_answerable(ctx, args.per_passage):
                    f.write(json.dumps({"passage": ctx, "title": title, "question": q,
                                        "answer": a, "answerable": True,
                                        "method": "targeted:answerable"}, ensure_ascii=False) + "\n")
                    written += 1
            if "unanswerable" in modes:
                for q in qgen.gen_unanswerable(ctx, args.per_passage):
                    f.write(json.dumps({"passage": ctx, "title": title, "question": q,
                                        "answer": None, "answerable": False,
                                        "method": "targeted:unanswerable"}, ensure_ascii=False) + "\n")
                    written += 1
            print(f"[{i}/{len(pool)}] targeted={written}".ljust(50), end="\r")
    print(f"\nWrote {written} targeted candidates to {OUT_FILE} (modes: {sorted(modes)}). "
          f"Next: build_v2.py")


if __name__ == "__main__":
    main()
