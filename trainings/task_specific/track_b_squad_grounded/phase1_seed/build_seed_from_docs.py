"""Phase 1 (alternative): build the corpus from YOUR OWN docs instead of SQuAD.

Drop-in replacement for build_seed.py when you want Track B to run on a real
documentation / FAQ corpus. It ingests a folder of documents, chunks them into
passages, reserves a held-out set for gold (split BY DOCUMENT so generation can
never leak into eval), and writes the same files the rest of the pipeline reads:

  data/passages.jsonl   the train passage pool (Phase 2 generates QA from these)
  data/gold.jsonl       held-out eval set (machine-DRAFTED, must be hand-verified)
  data/seed.jsonl       a small training seed (machine-drafted, verify too)

Supported inputs: .md, .txt, .html (tags stripped), .rst. Convert PDFs to text
first (e.g. `pdftotext`).

IMPORTANT, the gold set is sacred. Unlike SQuAD your docs have no verified QA, so
this drafts gold/seed with the local model and marks every row "verified": false.
You MUST open data/gold.jsonl, check each answer against its oracle passage, fix
or drop the bad rows, and set "verified": true before trusting any eval number.
A model graded on machine-drafted gold measures nothing. Use --no-generate to get
just the passage pool and write the gold QA by hand.

Run from the track_b_squad_grounded folder (Ollama up unless --no-generate):
    ../../../.venv/bin/python phase1_seed/build_seed_from_docs.py --docs-dir /path/to/docs
"""

import argparse
import html
import os
import random
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import common

EXTS = (".md", ".txt", ".html", ".htm", ".rst")
SEED = 0


def read_doc(path):
    """Load a document to plain-ish text. Light cleanup per format; we are not
    trying to be perfect, just to get readable prose for chunking."""
    with open(path, errors="ignore") as f:
        text = f.read()
    if path.endswith((".html", ".htm")):
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = html.unescape(text)
    else:  # md / rst / txt: drop code fences and heading/markup punctuation
        text = re.sub(r"(?s)```.*?```", " ", text)
        text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)   # ATX headings
        text = re.sub(r"[`*_>|]", " ", text)
    return text


def chunk(text, min_words, max_words):
    """Split into passages on blank lines, then greedily merge short paragraphs
    up to max_words and drop anything below min_words (boilerplate, nav, etc.)."""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    passages, buf = [], []
    for p in paras:
        buf.append(p)
        if len(" ".join(buf).split()) >= max_words:
            passages.append(" ".join(buf))
            buf = []
    if buf:
        passages.append(" ".join(buf))
    # clip overly long ones and drop the too-short
    out = []
    for p in passages:
        words = p.split()
        if len(words) < min_words:
            continue
        out.append(" ".join(words[:max_words]))
    return out


def load_corpus(docs_dir, min_words, max_words):
    """Return [{title, context}] across all docs. Title is the file stem so
    distractors can be drawn from OTHER documents."""
    files = []
    for root, _, names in os.walk(docs_dir):
        for n in names:
            if n.lower().endswith(EXTS):
                files.append(os.path.join(root, n))
    if not files:
        raise SystemExit(f"no {EXTS} files under {docs_dir}")
    corpus = []
    for path in sorted(files):
        title = os.path.splitext(os.path.basename(path))[0]
        for ctx in chunk(read_doc(path), min_words, max_words):
            corpus.append({"title": title, "context": ctx, "_doc": path})
    return corpus, len(files)


def raft(rng, oracle, title, pool, n_distractors):
    """oracle passage + n distractors from other documents, shuffled."""
    cand = [p["context"] for p in pool if p["title"] != title and p["context"] != oracle]
    distractors = rng.sample(cand, min(n_distractors, len(cand)))
    passages = [oracle] + distractors
    rng.shuffle(passages)
    return passages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", required=True, help="folder of .md/.txt/.html/.rst docs")
    ap.add_argument("--gold-doc-frac", type=float, default=0.2,
                    help="fraction of DOCUMENTS held out for gold (disjoint from train)")
    ap.add_argument("--min-words", type=int, default=30)
    ap.add_argument("--max-words", type=int, default=180)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--gold-passages", type=int, default=40, help="held-out passages to draft gold from")
    ap.add_argument("--seed-passages", type=int, default=60, help="train passages to draft seed from")
    ap.add_argument("--answerable-per-passage", type=int, default=2)
    ap.add_argument("--unanswerable-per-passage", type=int, default=1)
    ap.add_argument("--no-generate", action="store_true",
                    help="only build passages.jsonl + gold_passages.jsonl; write gold/seed QA by hand")
    args = ap.parse_args()

    rng = random.Random(SEED)
    corpus, n_files = load_corpus(args.docs_dir, args.min_words, args.max_words)
    print(f"loaded {n_files} files -> {len(corpus)} passages")

    # Split BY DOCUMENT so a gold passage's article never appears in training.
    docs = sorted({p["_doc"] for p in corpus})
    rng.shuffle(docs)
    n_gold_docs = max(1, int(len(docs) * args.gold_doc_frac))
    gold_docs = set(docs[:n_gold_docs])
    train_pool = [p for p in corpus if p["_doc"] not in gold_docs]
    gold_pool = [p for p in corpus if p["_doc"] in gold_docs]
    print(f"split: {len(train_pool)} train passages, {len(gold_pool)} gold passages "
          f"({n_gold_docs}/{len(docs)} docs held out)")

    os.makedirs(common.DATA, exist_ok=True)
    common.write_jsonl(f"{common.DATA}/passages.jsonl",
                       [{"title": p["title"], "context": p["context"]} for p in train_pool])
    common.write_jsonl(f"{common.DATA}/gold_passages.jsonl",
                       [{"title": p["title"], "context": p["context"]} for p in gold_pool])
    print("wrote data/passages.jsonl (Phase 2 generates from these) + data/gold_passages.jsonl")

    if args.no_generate:
        print("\n--no-generate: skipped QA drafting. Hand-write data/gold.jsonl and "
              "data/seed.jsonl\nusing the rows in gold_passages.jsonl / passages.jsonl as "
              "context (see build_row in common.py for the format).")
        return

    # Draft gold + seed QA with the local model. Reuse the Phase 2 generators.
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "phase2_synthetic"))
    import qgen
    qgen.sdg.preflight()

    def draft(passages_subset, distractor_pool, as_gold):
        rows = []
        for p in passages_subset:
            ctx, title = p["context"], p["title"]
            for q, a in qgen.gen_answerable(ctx, args.answerable_per_passage):
                rp = raft(rng, ctx, title, distractor_pool, args.distractors)
                if as_gold:
                    row = common.build_gold_row(rp, q, [a], True); row["oracle"] = ctx
                else:
                    row = common.build_row(rp, q, a)
                row["verified"] = False
                rows.append(row)
            for q in qgen.gen_unanswerable(ctx, args.unanswerable_per_passage):
                rp = raft(rng, ctx, title, distractor_pool, args.distractors)
                if as_gold:
                    row = common.build_gold_row(rp, q, [], False); row["oracle"] = ctx
                else:
                    row = common.build_row(rp, q, common.ABSTAIN)
                row["verified"] = False
                rows.append(row)
        return rows

    rng.shuffle(gold_pool)
    rng.shuffle(train_pool)
    gold = draft(gold_pool[:args.gold_passages], train_pool, as_gold=True)
    seed = draft(train_pool[:args.seed_passages], train_pool, as_gold=False)
    common.write_jsonl(f"{common.DATA}/gold.jsonl", gold)
    common.write_jsonl(f"{common.DATA}/seed.jsonl", seed)

    print(f"\nDRAFTED (verified=false): gold {len(gold)} rows, seed {len(seed)} rows")
    print("=" * 70)
    print("ACTION REQUIRED: data/gold.jsonl is machine-drafted and NOT yet trustworthy.")
    print("Open it, check each answer against its oracle passage, fix/drop bad rows,")
    print("set \"verified\": true, then re-run with the verified file. Do the same for")
    print("seed.jsonl. Only then are the eval numbers meaningful.")
    print("=" * 70)
    print("Next (after verifying): phase2_synthetic/qgen.py -> judge.py -> filter.py")


if __name__ == "__main__":
    main()
