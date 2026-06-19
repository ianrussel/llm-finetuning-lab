"""Measure a grounded-QA model on the two axes, on the SAME sets every time.

  axis 1 (task)       : on data/gold.jsonl, split by question type
                          answerable   -> SQuAD exact-match + token-F1
                          unanswerable -> abstention accuracy + hallucination rate
                        combined into one grounded_score, plus an OPTIONAL
                        LLM-as-judge faithfulness pass (--judge) against the oracle
                        passage, the doc's faithfulness axis.
  axis 2 (regression) : score on sentinel.jsonl, the same fixed general probes as
                        Track A, to catch forgetting after training.

Used by both the Phase 1 baseline and the Phase 4 re-measure. Run from the
track_b_squad_grounded folder.

    ../../../.venv/bin/python eval/evaluate.py --name base
    ../../../.venv/bin/python eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth

Results go to data/result_<name>.json and per-row preds to data/preds_<name>.jsonl.
Use --limit to smoke-test on a slice. --judge adds the faithfulness pass (Ollama up).
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import common

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SENTINEL_PATH = os.path.join(_HERE, "sentinel.jsonl")
F1_CORRECT = 0.5   # an answerable row counts as correct at or above this token-F1


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load(adapter):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=pick_dtype())
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tok


def generate(model, tok, prompts, max_new_tokens, batch):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                 for m in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
        print(f"  {min(i + batch, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return outs


def eval_task(model, tok, rows, batch):
    prompts = [[m for m in r["messages"] if m["role"] != "assistant"] for r in rows]
    raw = generate(model, tok, prompts, max_new_tokens=48, batch=batch)

    per_row, n_ans, n_unans = [], 0, 0
    em_sum = f1_sum = ans_correct = 0.0
    abstain_correct = hallucinated = 0
    for r, out in zip(rows, raw):
        answerable = r["answerable"]
        abstained = common.is_abstention(out)
        rec = {"question": common.question_of(r), "answerable": answerable,
               "gold": r["answers"], "pred": out, "abstained": abstained}
        if answerable:
            n_ans += 1
            em = common.exact_match(out, r["answers"])
            f1 = common.token_f1(out, r["answers"])
            correct = (not abstained) and f1 >= F1_CORRECT
            em_sum += em
            f1_sum += f1
            ans_correct += correct
            rec.update({"em": em, "f1": round(f1, 3), "correct": bool(correct)})
        else:
            n_unans += 1
            correct = abstained
            abstain_correct += abstained
            hallucinated += (not abstained)
            rec.update({"em": None, "f1": None, "correct": bool(correct)})
        per_row.append(rec)

    n = len(rows)
    grounded = (ans_correct + abstain_correct) / n if n else 0.0
    return {
        "n": n, "n_answerable": n_ans, "n_unanswerable": n_unans,
        "em": em_sum / n_ans if n_ans else 0.0,
        "f1": f1_sum / n_ans if n_ans else 0.0,
        "answerable_accuracy": ans_correct / n_ans if n_ans else 0.0,
        "abstention_accuracy": abstain_correct / n_unans if n_unans else 0.0,
        "hallucination_rate": hallucinated / n_unans if n_unans else 0.0,
        "grounded_score": grounded,
        "rows": per_row,
    }


def judge_faithfulness(rows, batch):
    """Optional: ask the local model whether each answered (non-abstained) answer
    is faithful to the oracle passage. Catches plausible-but-unsupported answers
    that string overlap would miss. Reuses the Phase 2 Ollama connection."""
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "phase2_synthetic"))
    import sdg
    sdg.preflight()
    checked = faithful = 0
    for r in rows:
        if not r["answerable"] or r["abstained"]:
            continue
        oracle = r.get("_oracle", "")
        raw = sdg.chat([
            {"role": "system", "content": "You grade answer faithfulness. Reply ONLY JSON."},
            {"role": "user", "content":
                f"Passage:\n{oracle}\n\nAnswer: {r['pred']}\n\nIs the answer supported by "
                f'the passage, using no outside knowledge? Reply {{"faithful": true or false}}'},
        ], temperature=0.0, num_predict=40)
        obj = sdg.parse_json(raw) or {}
        checked += 1
        faithful += bool(obj.get("faithful"))
    return {"checked": checked, "faithful_rate": faithful / checked if checked else 0.0}


def eval_sentinel(model, tok, batch):
    probes = common.read_jsonl(SENTINEL_PATH)
    prompts = [[{"role": "user", "content": p["question"]}] for p in probes]
    raw = generate(model, tok, prompts, max_new_tokens=32, batch=batch)
    rows, ok = [], 0
    for p, o in zip(probes, raw):
        hit = any(a in o.lower() for a in p["answers"])
        ok += hit
        rows.append({"q": p["question"], "want": p["answers"], "got": o, "hit": hit})
    return {"n": len(probes), "score": ok / len(probes), "probes": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N gold rows")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--judge", action="store_true", help="add the LLM faithfulness pass")
    args = ap.parse_args()

    gold = common.read_jsonl(f"{common.DATA}/gold.jsonl")
    if args.limit:
        gold = gold[:args.limit]

    model, tok = load(args.adapter)
    print(f"model: {BASE_MODEL}" + (f" + adapter {args.adapter}" if args.adapter else "")
          + f"  dtype={pick_dtype()}  device={model.device}")

    print(f"\naxis 1, grounded QA on {len(gold)} gold rows:")
    task = eval_task(model, tok, gold, args.batch)
    print(f"  answerable  : EM={task['em']:.3f}  F1={task['f1']:.3f}  "
          f"acc@{F1_CORRECT}={task['answerable_accuracy']:.3f}  (n={task['n_answerable']})")
    print(f"  unanswerable: abstain_acc={task['abstention_accuracy']:.3f}  "
          f"hallucination={task['hallucination_rate']:.3f}  (n={task['n_unanswerable']})")
    print(f"  grounded_score (combined): {task['grounded_score']:.3f}")

    per_row = task.pop("rows")
    # attach oracle for the optional judge, without bloating the saved result
    for rec, g in zip(per_row, gold):
        rec["_oracle"] = g.get("oracle", "")
    if args.judge:
        print("\nLLM faithfulness judge on answered rows:")
        task["judge_faithfulness"] = judge_faithfulness(per_row, args.batch)
        print(f"  faithful_rate={task['judge_faithfulness']['faithful_rate']:.3f} "
              f"({task['judge_faithfulness']['checked']} checked)")
    for rec in per_row:
        rec.pop("_oracle", None)

    preds_path = f"{common.DATA}/preds_{args.name}.jsonl"
    common.write_jsonl(preds_path, per_row)
    misses = sum(1 for r in per_row if not r["correct"])
    print(f"  wrote {misses} misses / {len(per_row)} rows -> {preds_path}")

    print(f"\naxis 2, regression sentinel:")
    sent = eval_sentinel(model, tok, args.batch)
    print(f"  score={sent['score']:.3f} ({int(sent['score']*sent['n'])}/{sent['n']})")

    out = {"name": args.name, "base_model": BASE_MODEL, "adapter": args.adapter,
           "task": task, "sentinel": sent}
    path = f"{common.DATA}/result_{args.name}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
