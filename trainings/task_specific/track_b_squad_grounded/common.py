"""Shared bits for Track B: the corpus-grounded QA specialist.

Track B mirrors the real scenario: turn a document corpus into a grounded task
and train a specialist that answers ONLY from the supplied context and abstains
when the answer is not there. The corpus is SQuAD v2 (Wikipedia passages +
verified QA, including deliberately unanswerable questions). The task contract
lives here so the seed builder, the document-grounded generator, the trainer and
the evaluator all speak the exact same format.

Contract
  input  : a context block (one or more passages) and a question
  output : the answer, copied as a short span from the context, OR the exact
           abstention string when the context does not contain the answer
  metric : SQuAD-style exact-match / token-F1 on answerable gold, abstention
           accuracy + hallucination rate on unanswerable gold (deterministic),
           plus an optional LLM-as-judge faithfulness pass against the source

RAFT
  Training rows carry the oracle passage (the one holding the answer) mixed with
  distractor passages from other articles, shuffled, so the model learns to find
  the answer among irrelevant context and to abstain when none of the passages
  contain it. Unanswerable rows are all-distractor by construction.
"""

import json
import os
import re
import string

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# The exact abstention string. It is the whole learning signal for the "I don't
# know" behaviour, so it must be fixed and matched verbatim everywhere.
ABSTAIN = "not in the context"

SYSTEM_PROMPT = (
    "You are a question-answering assistant that uses ONLY the provided context. "
    "Read the documents and answer the question with the shortest exact phrase "
    "from the context that answers it. Do not use any outside knowledge. If the "
    f"context does not contain the answer, reply with exactly: {ABSTAIN}"
)


def build_context(passages):
    """Join one or more passages into a single labelled context block. The labels
    give the model a stable structure to attend over when there are distractors."""
    return "\n\n".join(f"[Document {i + 1}]\n{p.strip()}"
                       for i, p in enumerate(passages))


def user_content(passages, question):
    return f"Context:\n{build_context(passages)}\n\nQuestion: {question}"


def build_row(passages, question, answer):
    """One training example in the conversational messages format. `answer` is a
    gold span for an answerable question, or ABSTAIN for an unanswerable one."""
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content(passages, question)},
        {"role": "assistant", "content": answer},
    ]}


def build_gold_row(passages, question, answers, answerable):
    """A gold/eval row: the prompt plus the metadata the scorer needs. `answers`
    is the list of acceptable gold strings (empty for unanswerable)."""
    answer = answers[0] if (answerable and answers) else ABSTAIN
    row = build_row(passages, question, answer)
    row["answers"] = list(answers)
    row["answerable"] = bool(answerable)
    return row


def user_of(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def assistant_of(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "assistant")


def question_of(row):
    """Pull the question back out of a user turn (after the last 'Question: ')."""
    u = user_of(row)
    return u.split("Question:", 1)[1].strip() if "Question:" in u else u


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize(s):
    """Lowercase + collapse whitespace, for dedup and decontamination."""
    return " ".join(s.lower().split())


# --- SQuAD-style answer scoring (the official normalization) -----------------

def _norm_answer(s):
    """Lowercase, strip punctuation, drop articles, collapse whitespace, the
    standard SQuAD answer normalization so 'The Rhine.' == 'rhine'."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_abstention(text):
    """True if the model output means the fixed abstention, robust to casing,
    punctuation and a trailing explanation."""
    n = _norm_answer(text)
    a = _norm_answer(ABSTAIN)
    return n == a or n.startswith(a)


def exact_match(pred, golds):
    """1 if the normalized prediction equals any normalized gold answer."""
    p = _norm_answer(pred)
    return float(any(p == _norm_answer(g) for g in golds))


def token_f1(pred, golds):
    """Best token-overlap F1 of the prediction against any gold answer (SQuAD)."""
    def f1(pred, gold):
        pt, gt = _norm_answer(pred).split(), _norm_answer(gold).split()
        if not pt or not gt:
            return float(pt == gt)
        common = 0
        gpool = list(gt)
        for t in pt:
            if t in gpool:
                common += 1
                gpool.remove(t)
        if common == 0:
            return 0.0
        prec, rec = common / len(pt), common / len(gt)
        return 2 * prec * rec / (prec + rec)
    return max((f1(pred, g) for g in golds), default=0.0)


# --- near-duplicate / contamination checks (same approach as Track A) --------

def char_shingles(s, n=13):
    """Set of overlapping n-character slices of the normalized string, for
    near-duplicate and contamination checks."""
    t = normalize(s)
    if len(t) <= n:
        return {t}
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
