"""
Module 2: the "after". Same eval set and scoring as test_base.py, but loads
the base model glued to the LoRA adapter trained by train.py (./lora-out).
Compare the two summary lines to see what the fine-tune bought you.
"""

import json
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ADAPTER_DIR = "./lora-out"
SYSTEM = ("You are a support ticket classifier. Reply with ONLY a JSON object "
          "of the form {\"category\": one of [billing, technical, account, general], "
          "\"priority\": one of [low, medium, high]} and nothing else.")

model = AutoPeftModelForCausalLM.from_pretrained(ADAPTER_DIR)
tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)


def parse(answer):
    """Return (category, priority) if the answer is valid JSON, else (None, None)."""
    try:
        obj = json.loads(answer)
        return obj.get("category"), obj.get("priority")
    except Exception:
        return None, None


with open("data/eval.jsonl") as f:
    rows = [json.loads(line) for line in f if line.strip()]

valid_json = 0
exact = 0
for row in rows:
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    gold = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]          # only the new tokens
    answer = tok.decode(gen, skip_special_tokens=True).strip()

    cat, pri = parse(answer)
    g_cat, g_pri = parse(gold)
    if cat is not None:
        valid_json += 1
    if (cat, pri) == (g_cat, g_pri):
        exact += 1

    print(f"MSG : {user}")
    print(f"WANT: {gold}")
    print(f"GOT : {answer}")
    print("-" * 60)

n = len(rows)
print(f"valid JSON: {valid_json}/{n}    exact match: {exact}/{n}")
