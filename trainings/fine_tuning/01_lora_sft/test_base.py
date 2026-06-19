from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(name)
tok = AutoTokenizer.from_pretrained(name)

msgs = [{"role": "user", "content": "What's the capital of France?"}]
# transformers 5.x returns a BatchEncoding (dict-like: input_ids + attention_mask),
# so unpack it with ** instead of passing it positionally.
inputs = tok.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
out = model.generate(**inputs, max_new_tokens=50)
print(tok.decode(out[0], skip_special_tokens=True))