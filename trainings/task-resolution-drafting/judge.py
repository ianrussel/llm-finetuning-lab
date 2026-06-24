"""LLM-as-judge for the open-ended task: turn free-text answer quality into a gate score.

For an open-ended task (draft a support reply) there is no single right answer, so macro-F1 does
not apply. Instead a STRONGER model judges, per held-out input, whether the fine-tuned model's
answer or the base model's answer is better. The fraction the fine-tuned model wins (ties count
half) is the win-rate, which the gate uses in place of task macro-F1.

Bias control: LLM judges have position bias, so each pair is judged in BOTH orders (candidate as A
and as B) and only counts as a win if it is preferred in both; one win + one tie counts as a half.
That cancels a judge that simply prefers whichever answer is shown first.

Calibrate the judge against a handful of human judgments (calibrate.py) before trusting it.

Default judge: Qwen2.5-7B-Instruct in 4-bit (stronger than the small models being trained, fits a
T4 loaded after the candidate is freed). Set cfg["judge"]["model"] to change it.
"""

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM = (
    "You are an impartial judge of customer-support replies. You are given the customer's ticket "
    "and two candidate replies, A and B. Decide which reply better resolves the ticket: more "
    "correct, helpful, and professional. Do not favour length or position. Briefly justify, then "
    "end with exactly one line: 'VERDICT: A', 'VERDICT: B', or 'VERDICT: tie'."
)
USER = "Ticket:\n{ticket}\n\nReply A:\n{a}\n\nReply B:\n{b}\n\nWhich reply is better?"


def _load_judge(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = None
    if torch.cuda.is_available():
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    return model.eval(), tok


def _verdict(text):
    """Parse the judge's VERDICT line; tolerate minor format drift. Returns 'A' / 'B' / 'tie'."""
    m = re.findall(r"verdict\s*:?\s*(a|b|tie)", text.lower())
    if m:
        return "tie" if m[-1] == "tie" else m[-1].upper()   # normalise to A / B / tie
    tail = text.lower().strip().splitlines()[-1] if text.strip() else ""
    if "tie" in tail:
        return "tie"
    if re.search(r"\bb\b", tail) and not re.search(r"\ba\b", tail):
        return "B"
    if re.search(r"\ba\b", tail) and not re.search(r"\bb\b", tail):
        return "A"
    return "tie"


def _gen_verdicts(model, tok, prompts, max_new=256, batch=4):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def _prompt(ticket, a, b):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(ticket=ticket, a=a, b=b)}]


def judge_pairwise(model_id, tickets, base_answers, cand_answers, batch=4, model_tok=None):
    """Win-rate of cand over base across tickets, judged in both A/B orders to cancel position bias.
    Pass model_tok=(model, tok) to reuse a loaded judge; otherwise it loads and frees one."""
    own = model_tok is None
    model, tok = _load_judge(model_id) if own else model_tok
    try:
        # order 1: cand=A, base=B ; order 2: base=A, cand=B
        p1 = [_prompt(t, c, b) for t, b, c in zip(tickets, base_answers, cand_answers)]
        p2 = [_prompt(t, b, c) for t, b, c in zip(tickets, base_answers, cand_answers)]
        v1 = [_verdict(x) for x in _gen_verdicts(model, tok, p1, batch=batch)]
        v2 = [_verdict(x) for x in _gen_verdicts(model, tok, p2, batch=batch)]
    finally:
        if own:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    verdicts, score = [], 0.0
    for a1, a2 in zip(v1, v2):
        cand1 = 1.0 if a1 == "A" else (0.5 if a1 == "tie" else 0.0)   # cand was A in order 1
        cand2 = 1.0 if a2 == "B" else (0.5 if a2 == "tie" else 0.0)   # cand was B in order 2
        pair = (cand1 + cand2) / 2.0                                  # average the two orders
        score += pair
        verdicts.append({"order1": a1, "order2": a2, "cand_score": pair})
    win_rate = score / len(tickets) if tickets else 0.0
    return {"win_rate": round(win_rate, 4), "n": len(tickets), "verdicts": verdicts}
