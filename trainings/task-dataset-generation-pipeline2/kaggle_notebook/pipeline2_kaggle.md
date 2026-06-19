# Part 2 (config-driven train + gate) on a free Kaggle GPU

Generation already happened in Part 1. Here you run the Part 2 loop on a Kaggle T4: a
config-driven LoRA run, the two-axis evaluation, and the acceptance gate (with one adjusted
re-run on failure). Self-contained, paste each block into its own Kaggle cell. Set
**Settings -> Accelerator -> GPU T4**.

The config is still the single source of truth, it just lives in the `CFG` dict in cell 2
instead of `config.yaml` (so the notebook needs nothing uploaded but the data).

## 0. Upload the data
Make a Kaggle Dataset named **`pipeline2-data`** from your Part 1 outputs:
`train_mix.jsonl`, `train_synth.jsonl`, `gold.jsonl`, `labels.txt` (from pipeline1 `data/`) and
`sentinel.jsonl`, `reasoning_probes.jsonl`, `tool_probes.jsonl` (from pipeline1 `eval/`). It
mounts at `/kaggle/input/pipeline2-data/`.

## 1. Install deps
```python
!pip -q install -U "transformers>=4.44" "trl>=0.9" peft bitsandbytes datasets accelerate
!pip -q uninstall -y torchao   # Kaggle ships an old torchao that recent peft rejects; we use bitsandbytes
print("deps installed")
```

## 2. Config (the single source of truth) + helpers
```python
import os, json, re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DATA = "/kaggle/input/pipeline2-data"
OUT  = "/kaggle/working"

CFG = {
    "name": "helpdesk-resolution",
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "lora": {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
    "train": {"epochs": 3, "lr": 2e-4, "max_len": 1408, "batch": 1, "grad_accum": 16},
    "guardrails": {"replay_mix": True},                 # train_mix vs train_synth
    "acceptance": {"min_task_gain_macro_f1": 0.05, "max_regression_drop": 0.05},
    "adjust_on_fail": {"force_replay": True, "lower_lr_factor": 0.5, "reduce_epochs_to": 2},
}
LABELS = [l.strip() for l in open(f"{DATA}/labels.txt") if l.strip()] or ["Done", "Won't Do"]

def read_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]
def normalize(s): return " ".join(s.lower().split())
def c_predict_label(out, labels):
    tail = out.rsplit("</think>", 1)[-1] if "</think>" in out else out
    by = {normalize(l): l for l in labels}
    lines = [x for x in tail.splitlines() if x.strip()]
    last = normalize(lines[-1]) if lines else ""
    if last in by: return by[last]
    low = normalize(tail); hits = [l for l in labels if normalize(l) in low]
    return max(hits, key=len) if hits else None
def macro_f1(gold, pred, labels):
    tot = 0.0
    for l in labels:
        tp = sum(1 for g,p in zip(gold,pred) if g==l and p==l)
        fp = sum(1 for g,p in zip(gold,pred) if g!=l and p==l)
        fn = sum(1 for g,p in zip(gold,pred) if g==l and p!=l)
        pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
        tot += 2*pr*rc/(pr+rc) if pr+rc else 0.0
    return tot/len(labels)
def score_probes(raw, probes, mode):
    ok = 0
    for o, p in zip(raw, probes):
        want = [a.lower() for a in p["answers"]]; low = o.lower()
        ok += (all(a in low for a in want) if mode=="all" else any(a in low for a in want))
    return ok/len(probes) if probes else 0.0
print("config ready; DATA:", os.listdir(DATA) if os.path.isdir(DATA) else "ADD THE DATASET")
```

## 3. Config-driven training (stage 3.2)
```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

def train_one(data_file, run_name, lr=None, epochs=None):
    t, lo = CFG["train"], CFG["lora"]
    lr = t["lr"] if lr is None else lr
    epochs = t["epochs"] if epochs is None else epochs
    tok = AutoTokenizer.from_pretrained(CFG["base_model"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(CFG["base_model"], quantization_config=quant,
        torch_dtype=torch.bfloat16, device_map="auto"); model.config.use_cache = False
    lora = LoraConfig(r=lo["rank"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=lo["target_modules"])
    ds = load_dataset("json", data_files=f"{DATA}/{data_file}", split="train")
    out = f"{OUT}/{run_name}"
    cfg = SFTConfig(output_dir=out, num_train_epochs=epochs, per_device_train_batch_size=t["batch"],
        gradient_accumulation_steps=t["grad_accum"], learning_rate=lr, lr_scheduler_type="cosine",
        warmup_ratio=0.05, logging_steps=10, save_strategy="epoch", bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=t["max_len"], packing=False, assistant_only_loss=True, seed=0, report_to="none")
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok)
    r = tr.train(); tr.save_model(out); tok.save_pretrained(out)
    print(f"[train] {run_name}: loss={r.training_loss:.4f} (data={data_file}, lr={lr}, epochs={epochs})")
    return out
```

## 4. Two-axis evaluation (stage 3.4)
```python
def _load(adapter=None):
    tok = AutoTokenizer.from_pretrained(CFG["base_model"]); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(CFG["base_model"], torch_dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel; m = PeftModel.from_pretrained(m, adapter)
    return m.to("cuda").eval(), tok

@torch.no_grad()
def _gen(m, tok, prompts, max_new, batch=16):
    outs = []
    for i in range(0, len(prompts), batch):
        ch = prompts[i:i+batch]
        txt = [tok.apply_chat_template(x, add_generation_prompt=True, tokenize=False) for x in ch]
        enc = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(ch)):
            outs.append(tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return outs

def _probe(m, tok, fname, mode, max_new):
    probes = read_jsonl(f"{DATA}/{fname}")
    raw = _gen(m, tok, [[{"role":"user","content":p["question"]}] for p in probes], max_new)
    return {"n": len(probes), "score": score_probes(raw, probes, mode)}

def evaluate(name, adapter=None):
    gold = read_jsonl(f"{DATA}/gold.jsonl")
    m, tok = _load(adapter)
    raw = _gen(m, tok, [r["messages"] for r in gold], 384)
    g = [r["label"] for r in gold]; p = [c_predict_label(o, LABELS) for o in raw]
    task = {"macro_f1": macro_f1(g, p, LABELS),
            "accuracy": sum(1 for a,b in zip(g,p) if a==b)/len(g),
            "valid": sum(1 for x in p if x is not None)/len(g)}
    res = {"name": name, "task": task,
           "sentinel": _probe(m, tok, "sentinel.jsonl", "any", 32),
           "reasoning": _probe(m, tok, "reasoning_probes.jsonl", "any", 256),
           "tools": _probe(m, tok, "tool_probes.jsonl", "all", 128)}
    json.dump(res, open(f"{OUT}/result_{name}.json","w"), indent=2)
    print(f"[eval] {name}: macroF1={task['macro_f1']:.3f} sentinel={res['sentinel']['score']:.3f} "
          f"reasoning={res['reasoning']['score']:.3f} tools={res['tools']['score']:.3f}")
    del m; torch.cuda.empty_cache(); return res
```

## 5. The gate (stage 3.4) + the orchestrated loop
```python
def gate(base, cand):
    a = CFG["acceptance"]
    task_gain = cand["task"]["macro_f1"] - base["task"]["macro_f1"]
    worst = max((base[k]["score"] - cand[k]["score"]) for k in ["sentinel","reasoning","tools"])
    accept = task_gain >= a["min_task_gain_macro_f1"] and worst <= a["max_regression_drop"]
    print(f"[gate] task_gain={task_gain:+.3f} (min {a['min_task_gain_macro_f1']:+.3f}) | "
          f"worst_drop={worst:.3f} (max {a['max_regression_drop']:.3f}) -> "
          f"{'ACCEPT' if accept else 'REJECT'}")
    return accept, task_gain, worst

base = evaluate("base")
replay = CFG["guardrails"]["replay_mix"]
name1 = f"{CFG['name']}-{'replay' if replay else 'noreplay'}"
adapter1 = train_one("train_mix.jsonl" if replay else "train_synth.jsonl", name1)
cand1 = evaluate(name1, adapter1)
accepted, *_ = gate(base, cand1)
final = adapter1 if accepted else None

if not accepted:                       # one adjusted re-run
    adj = CFG["adjust_on_fail"]
    lr = CFG["train"]["lr"] * adj["lower_lr_factor"]
    print(f"[pipeline] gate rejected; adjusted re-run (replay on, lr={lr}, epochs={adj['reduce_epochs_to']})")
    adapter2 = train_one("train_mix.jsonl", f"{CFG['name']}-adj", lr=lr, epochs=adj["reduce_epochs_to"])
    cand2 = evaluate(f"{CFG['name']}-adj", adapter2)
    ok2, *_ = gate(base, cand2)
    final = adapter2 if ok2 else None
print("\nACCEPTED:", final or "none")
```

## 6. (Optional) the guardrail experiment: no-replay vs replay
```python
nr = train_one("train_synth.jsonl", f"{CFG['name']}-noreplay")
res_nr = evaluate(f"{CFG['name']}-noreplay", nr)
print("\nregression axis, base vs no-replay vs replay (replay should hold the probes up):")
for tag, r in [("base", base), ("no-replay", res_nr), ("replay", cand1)]:
    print(f"  {tag:<9} macroF1={r['task']['macro_f1']:.3f}  sentinel={r['sentinel']['score']:.3f}  "
          f"reasoning={r['reasoning']['score']:.3f}  tools={r['tools']['score']:.3f}")
```

## 7. Download adapters + results
```python
import shutil
for n in [name1, f"{CFG['name']}-adj"]:
    if os.path.isdir(f"{OUT}/{n}"):
        shutil.make_archive(f"{OUT}/{n}", "zip", f"{OUT}/{n}")
print("Download the lora *.zip and result_*.json from the Output tab.")
```

Notes: this mirrors the local Part 2 scripts (`train_from_config` / `evaluate_from_config` /
`gate` / `pipeline`), with the config inlined as `CFG`. batch 1 + max_len 1408 fits a T4.
Two full training runs (candidate + adjusted, or no-replay + replay) take a while; keep the
dataset small. Copy the printed numbers into WRITEUP.md.
