# Track C training on a free Kaggle GPU

Generation runs locally (Ollama). Only training + eval run here. Paste each block into
its own Kaggle cell (or import into a notebook), set **Settings -> Accelerator -> GPU T4**.

## 0. Upload the data
Build the data locally first (phases 1-4). Then make a Kaggle Dataset named **`track-c-data`**
containing: `seed.jsonl`, `train_mix.jsonl`, `gold.jsonl` (from `data/`) and `sentinel.jsonl`,
`reasoning_probes.jsonl`, `tool_probes.jsonl` (from `eval/`). It mounts at `/kaggle/input/track-c-data/`.

## 1. Install deps
```python
!pip -q install -U "transformers>=4.44" "trl>=0.9" peft bitsandbytes datasets accelerate
!pip -q uninstall -y torchao   # Kaggle ships an old torchao that recent peft rejects; we use bitsandbytes
print("deps installed")
```

## 2. Config + inlined helpers
```python
import os, json, re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = "/kaggle/input/track-c-data"
OUT  = "/kaggle/working"
MAX_LEN, BATCH, GRAD_ACCUM, EPOCHS = 1408, 1, 16, 3
LABELS = ["Done", "Won't Do"]

def read_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]
def normalize(s): return " ".join(s.lower().split())
def c_predict_label(output, labels):
    tail = output.rsplit("</think>", 1)[-1] if "</think>" in output else output
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
print("config ready; DATA contents:", os.listdir(DATA) if os.path.isdir(DATA) else "ADD THE DATASET")
```

## 3. Train (control = seed, real = seed-synth)
```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

def train_one(data_file, name):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant,
        torch_dtype=torch.bfloat16, device_map="auto"); model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear")
    ds = load_dataset("json", data_files=f"{DATA}/{data_file}", split="train")
    out = f"{OUT}/lora-{name}"
    cfg = SFTConfig(output_dir=out, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_ratio=0.05, logging_steps=10, save_strategy="epoch", bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_LEN, packing=False, assistant_only_loss=True, seed=0, report_to="none")
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok)
    r = tr.train(); tr.save_model(out); tok.save_pretrained(out)
    print(f"{name}: final_train_loss={r.training_loss:.4f} -> {out}"); return out

train_one("seed.jsonl", "seed")
train_one("train_mix.jsonl", "seed-synth")
```

## 4. Evaluate base vs seed vs seed-synth (both axes)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(adapter=None):
    t = AutoTokenizer.from_pretrained(BASE_MODEL); t.padding_side = "left"
    if t.pad_token is None: t.pad_token = t.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel; m = PeftModel.from_pretrained(m, adapter)
    return m.to("cuda").eval(), t

@torch.no_grad()
def gen(m, t, prompts, max_new, batch=16):
    outs = []
    for i in range(0, len(prompts), batch):
        ch = prompts[i:i+batch]
        txt = [t.apply_chat_template(x, add_generation_prompt=True, tokenize=False) for x in ch]
        enc = t(txt, return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=t.pad_token_id)
        for j in range(len(ch)):
            outs.append(t.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return outs

def probe_score(m, t, fname, mode, max_new):
    rows = read_jsonl(f"{DATA}/{fname}")
    raw = gen(m, t, [[{"role":"user","content":r["question"]}] for r in rows], max_new)
    ok = 0
    for r, o in zip(rows, raw):
        ans = [a.lower() for a in r["answers"]]; low = o.lower()
        ok += (all(a in low for a in ans) if mode=="all" else any(a in low for a in ans))
    return ok/len(rows)

def evaluate(name, adapter=None):
    gold = read_jsonl(f"{DATA}/gold.jsonl")
    m, t = load_model(adapter)
    raw = gen(m, t, [r["messages"] for r in gold], 384)
    g = [r["label"] for r in gold]; p = [c_predict_label(o, LABELS) for o in raw]
    acc = sum(1 for a,b in zip(g,p) if a==b)/len(g); f1 = macro_f1(g, p, LABELS)
    sent = probe_score(m, t, "sentinel.jsonl", "any", 32)
    reason = probe_score(m, t, "reasoning_probes.jsonl", "any", 256)
    tools = probe_score(m, t, "tool_probes.jsonl", "all", 128)
    res = {"name":name, "acc":round(acc,3), "macro_f1":round(f1,3),
           "sentinel":round(sent,3), "reasoning":round(reason,3), "tools":round(tools,3)}
    json.dump(res, open(f"{OUT}/result_{name}.json","w"), indent=2); print(res)
    del m; torch.cuda.empty_cache(); return res

R = [evaluate("base"), evaluate("seed", f"{OUT}/lora-seed"), evaluate("seed-synth", f"{OUT}/lora-seed-synth")]
print("\nname           acc    macroF1  sentinel  reasoning  tools")
for r in R:
    print(f"{r['name']:<14} {r['acc']:.3f}  {r['macro_f1']:.3f}    {r['sentinel']:.3f}     {r['reasoning']:.3f}      {r['tools']:.3f}")
```

## 5. Download the adapters
```python
import shutil
for n in ["seed", "seed-synth"]:
    shutil.make_archive(f"{OUT}/lora-{n}", "zip", f"{OUT}/lora-{n}")
print("Download lora-*.zip and result_*.json from the Output tab.")
```

Note: batch 1 + max_len 1408 fits a T4. Generation (survey/gen/judge/filter/mix) stays local on
Ollama; only this training + eval runs in the cloud, so the numbers stay comparable to local.
