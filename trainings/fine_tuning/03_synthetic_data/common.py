"""Shared bits for module 3 (the classifier task, reused from module 2)."""

import json

SYSTEM = ("You are a support ticket classifier. Reply with ONLY a JSON object "
          "of the form {\"category\": one of [billing, technical, account, general], "
          "\"priority\": one of [low, medium, high]} and nothing else.")

CATEGORIES = {"billing", "technical", "account", "general"}
PRIORITIES = {"low", "medium", "high"}


def label_json(category, priority):
    """The exact assistant string we train the model to produce."""
    return json.dumps({"category": category, "priority": priority})


def build_row(user, category, priority):
    """One training example in the conversational messages format."""
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": label_json(category, priority)},
    ]}


def parse_label(text):
    """Return (category, priority) if text is valid JSON, else (None, None)."""
    try:
        obj = json.loads(text)
        return obj.get("category"), obj.get("priority")
    except Exception:
        return None, None


def user_of(row):
    """Pull the user message out of a messages row."""
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def assistant_of(row):
    """Pull the assistant (gold label) out of a messages row."""
    return next(m["content"] for m in row["messages"] if m["role"] == "assistant")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(s):
    """Lowercase and collapse whitespace, for dedup comparisons."""
    return " ".join(s.lower().split())


# Greeting / sign-off openers that small models tend to emit as separate lines.
_GREETING_SIGNOFF = (
    "hi", "hello", "hey", "dear", "greetings", "good morning", "good afternoon",
    "good evening", "to whom", "thanks", "thank you", "regards", "best regards",
    "sincerely", "cheers", "kind regards", "best,",
)


# Meta / preamble lines the model emits around the actual rewrites.
_PREAMBLE = (
    "here is", "here are", "here's", "below are", "the following", "sure",
    "certainly", "rewrite", "rewrites", "versions", "message:",
)


def looks_like_junk(text):
    """True if text is not a real ticket message: too short, a greeting/sign-off,
    or a meta/preamble line.

    Small local models wrap rewrites in email formatting ("Hi there,", "Hello,")
    and announce them ("Here is your message rewritten in five ways:"). Splitting on
    newlines turns those into bogus one-line "paraphrases", so we drop anything under
    4 words, starting with a greeting/preamble, or ending in a colon (a meta line).
    """
    s = text.strip()
    if len(s.split()) < 4:
        return True
    if s.endswith(":"):
        return True
    low = s.lower()
    return low.startswith(_GREETING_SIGNOFF) or low.startswith(_PREAMBLE)
