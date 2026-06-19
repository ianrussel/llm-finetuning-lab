"""
Module 8: call the vLLM OpenAI-compatible server and pick a LoRA adapter per request.

Start the server first (see serve_vllm.sh), then: pip install openai, then run this.
The point: the SAME base model in memory answers as a different specialist depending on
which adapter you name in the `model` field. That is multi-LoRA serving.

    aipy client.py
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

SYSTEM = ("You are a support ticket classifier. Reply with ONLY a JSON object "
          "of the form {\"category\": one of [billing, technical, account, general], "
          "\"priority\": one of [low, medium, high]} and nothing else.")

ticket = "My invoice is higher than it was last month."


def ask(model_name, system, user):
    resp = client.chat.completions.create(
        model=model_name,            # <- selects base vs which LoRA adapter handles it
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=40, temperature=0,
    )
    return resp.choices[0].message.content.strip()


# Same request, routed to two different specialists on one base model:
print("via ticket-classifier adapter:")
print(" ", ask("ticket-classifier", SYSTEM, ticket))

print("\nvia the raw base model (no adapter):")
print(" ", ask("Qwen/Qwen2.5-0.5B-Instruct", SYSTEM, ticket))
