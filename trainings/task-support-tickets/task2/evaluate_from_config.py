"""Stage 3.4 (Evaluate): the two-axis measurement the gate consumes.

axis 1 (task)       : accuracy + macro-F1 on the held-out gold set (base vs fine-tuned).
axis 2 (regression) : the standard sentinel + a reasoning check + a tool-calling check.

Importable: evaluate(cfg, name, adapter=None) -> results dict (also written to
runs/result_<name>.json), so pipeline.py can score base and each candidate.

Run from this folder (GPU):
    ../../.venv/bin/python evaluate_from_config.py --name base
    ../../.venv/bin/python evaluate_from_config.py --name cand --adapter runs/helpdesk-resolution-replay
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common_p2 as c


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load(base, adapter):
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=pick_dtype())
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tok


def _gen(model, tok, prompts, max_new, batch):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        print(f"  {min(i + batch, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return outs


def _probe(model, tok, path, mode, max_new, batch):
    if not os.path.exists(path):
        return None
    probes = c.read_jsonl(path)
    raw = _gen(model, tok, [[{"role": "user", "content": p["question"]}] for p in probes], max_new, batch)
    return {"n": len(probes), "score": c.score_probes(raw, probes, mode)}


def evaluate(cfg, name, adapter=None, limit=0, batch=None):
    if batch is None:   # use the same eval batch as early stopping so best-epoch scores like the gate
        batch = cfg.get("train", {}).get("early_stopping", {}).get("eval_batch", 16)
    labels = c.load_labels(c.data_path(cfg, "labels"))
    gold = c.read_jsonl(c.data_path(cfg, "gold"))
    if limit:
        gold = gold[:limit]
    model, tok = _load(cfg["base_model"], adapter)
    print(f"[eval] {name}: base={cfg['base_model']}" + (f" + {adapter}" if adapter else ""))

    raw = _gen(model, tok, [r["messages"] for r in gold], 384, batch)
    g = [r["label"] for r in gold]
    p = [c.c_predict_label(o, labels) for o in raw]
    task = {
        "n": len(gold),
        "accuracy": sum(1 for a, b in zip(g, p) if a == b) / len(gold),
        "macro_f1": c.macro_f1(g, p, labels),
        "valid_label_rate": sum(1 for x in p if x is not None) / len(gold),
    }
    sentinel = _probe(model, tok, c.probe_path(cfg, "sentinel"), "any", 128, batch)
    reasoning = _probe(model, tok, c.probe_path(cfg, "reasoning_probes"), "any", 256, batch)
    tools = _probe(model, tok, c.probe_path(cfg, "tool_probes"), "all", 128, batch)

    res = {"name": name, "adapter": adapter, "task": task,
           "sentinel": sentinel, "reasoning": reasoning, "tools": tools}
    path = os.path.join(c.out_dir(cfg), f"result_{name}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[eval] {name}: macro_f1={task['macro_f1']:.3f} acc={task['accuracy']:.3f} "
          f"sentinel={sentinel and round(sentinel['score'],3)} "
          f"reasoning={reasoning and round(reasoning['score'],3)} "
          f"tools={tools and round(tools['score'],3)} -> {path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()
    cfg = c.load_config()
    evaluate(cfg, args.name, adapter=args.adapter, limit=args.limit, batch=args.batch)


if __name__ == "__main__":
    main()
