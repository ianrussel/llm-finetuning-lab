"""
Module 3: the "after". Base model + the adapter trained on the synthetic set.
Same eval and scoring as test_base.py, so compare the summary lines directly,
and also against module 2's adapter (base + 24 seeds only).
"""

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from common import SYSTEM, read_jsonl, user_of, assistant_of, parse_label

ADAPTER_DIR = "./lora-out"

model = AutoPeftModelForCausalLM.from_pretrained(ADAPTER_DIR)
tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)

rows = read_jsonl("data/eval.jsonl")
valid_json = 0
category_ok = 0
exact = 0
for row in rows:
    user = user_of(row)
    gold = assistant_of(row)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    answer = tok.decode(gen, skip_special_tokens=True).strip()

    cat, pri = parse_label(answer)
    g_cat, g_pri = parse_label(gold)
    if cat is not None:
        valid_json += 1
    if cat == g_cat:
        category_ok += 1
    if (cat, pri) == (g_cat, g_pri):
        exact += 1

    print(f"MSG : {user}")
    print(f"WANT: {gold}")
    print(f"GOT : {answer}")
    print("-" * 60)

n = len(rows)
print(f"valid JSON: {valid_json}/{n}   category: {category_ok}/{n}   exact: {exact}/{n}")
