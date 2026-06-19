from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# This ONE line loads the base model AND your adapter, glued together:
model = AutoPeftModelForCausalLM.from_pretrained("./lora-out")
tokenizer = AutoTokenizer.from_pretrained("./lora-out")

# Prompt it the SAME way as test_base.py (chat template), so the only
# difference between before/after is the adapter, not the prompting.
msgs = [{"role": "user", "content": "What's the capital of France?"}]
inputs = tokenizer.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
