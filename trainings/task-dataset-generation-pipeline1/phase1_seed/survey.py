"""Phase 1, step 1: the LLM surveys the knowledge base and proposes relevant vs
leakage fields for the target task; the human confirms (the brief's step 1).

This is the gate before any generation. The local model is shown the available
fields (across the linked tables) and a few real serialized examples, and asked,
for resolution prediction, which fields are usable AT DECISION TIME versus which
LEAK the outcome. Its proposal is merged with a known-safe exclude list (union, so
a miss by the small model can never re-admit a known leak) and written to
data/field_survey.json with "confirmed": false. The human reviews the include/
exclude lists and flips confirmed=true; downstream steps refuse to run until then.

Run from the track root (Ollama up):
    ../../.venv/bin/python phase1_seed/survey.py
"""

import copy
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import common_c as common
import link
import serialize
import sdg

OUT = serialize.SURVEY_PATH
SAMPLE_PER_CLASS = 20

# The fields available across the linked tables, with one-line meanings (from FEATURES.md).
FIELD_CATALOG = [
    ("issue_type", "category of the ticket (Ticket, Service, Bug, ...)"),
    ("issue_priority", "priority level set on the ticket"),
    ("issue_contr_count", "number of users who worked on the issue"),
    ("issue_comments_count", "number of comments on the issue"),
    ("processing_steps", "number of workflow steps the issue passed through"),
    ("wf_<state> / wfe_<state>", "time spent in / number of times it passed each workflow state"),
    ("change_history", "ordered log of assignee and status changes over time"),
    ("issue_status", "the issue's FINAL status (closed, done, ...)"),
    ("issue_resolution", "the outcome label itself (Done / Won't Do / ...)"),
    ("issue_resolution_date", "timestamp the issue was resolved"),
    ("utterances", "the conversation text (only present for a small sample of issues)"),
]

SURVEY_SYSTEM = ("You are a data scientist planning a supervised model that predicts a "
                 "help-desk ticket's resolution outcome (Done vs Won't Do) from features "
                 "available BEFORE the outcome is known. You reply with ONLY a JSON object.")


def main():
    sdg.preflight()
    db = link.HelpDeskDB()
    rng = random.Random(0)

    # Balanced sample of real issues, serialized WITHOUT the leakage filter so the
    # model can see every field and judge it.
    full_view = copy.deepcopy(serialize.DEFAULT_SURVEY)
    full_view["leakage_exclude"] = []
    full_view["workflow"]["exclude_states"] = []
    full_view["history"]["exclude_terminal_status"] = []
    examples = []
    for label in common.LABELS:
        ids = db.ids_with_resolution(label)
        rng.shuffle(ids)
        for nid in ids[:3]:
            examples.append(f"[resolution={label}]\n" + serialize.serialize_issue(db.get_issue(nid), full_view))

    catalog = "\n".join(f"- {n}: {d}" for n, d in FIELD_CATALOG)
    raw = sdg.chat([
        {"role": "system", "content": SURVEY_SYSTEM},
        {"role": "user", "content":
            "Available fields:\n" + catalog +
            "\n\nA few real examples (with the outcome shown only for your reference):\n\n"
            + "\n\n".join(examples) +
            "\n\nClassify each field as 'predictive' (safe to use, known before the outcome) "
            "or 'leakage' (reveals or is determined by the outcome, would let the model cheat). "
            'Reply JSON: {"predictive": ["..."], "leakage": ["..."], "notes": "short"}'},
    ], temperature=0.0, num_predict=400)
    proposal = sdg.parse_json(raw) or {"predictive": [], "leakage": [], "notes": "unparsed"}

    # Merge: keep the safe defaults (deep copy so we never mutate DEFAULT_SURVEY),
    # union in anything the model flagged as leakage. Metadata fields go to
    # leakage_exclude; workflow states (wf_/wfe_ prefixed) go to workflow.exclude_states.
    survey = copy.deepcopy(serialize.DEFAULT_SURVEY)
    model_leak = [str(x).strip() for x in proposal.get("leakage", [])]
    survey["leakage_exclude"] = sorted(set(survey["leakage_exclude"]) |
                                       {x for x in model_leak if not x.startswith(("wf_", "wfe_"))})
    wf_leak = {x.split("_", 1)[1] for x in model_leak
               if x.startswith(("wf_", "wfe_")) and "_" in x}
    survey["workflow"]["exclude_states"] = sorted(set(survey["workflow"]["exclude_states"]) | wf_leak)
    survey["model_proposal"] = proposal
    survey["review_me"] = ("Fields/states in leakage_exclude, workflow.exclude_states, and "
                           "history.exclude_terminal_status are REMOVED from the model's input "
                           "so it cannot cheat. Everything else is used as a feature. Edit these "
                           "exclude lists if needed, then set confirmed=true.")
    survey["confirmed"] = False

    os.makedirs(common.DATA, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(survey, f, indent=2, ensure_ascii=False)

    print("model proposal:")
    print("  predictive (proposed to USE):", proposal.get("predictive"))
    print("  leakage (proposed to DROP)  :", proposal.get("leakage"))
    print("\nthe EXCLUDE lists written (everything not listed is used as a feature):")
    print(f"  leakage_exclude (metadata)       : {survey['leakage_exclude']}")
    print(f"  workflow.exclude_states          : {survey['workflow']['exclude_states']}")
    print(f"  history.exclude_terminal_status  : {survey['history']['exclude_terminal_status']}")
    print(f"\nwrote {OUT} with confirmed=false.")
    print("ACTION: review those three EXCLUDE lists in data/field_survey.json, edit if needed, "
          "set \"confirmed\": true, then run phase1_seed/build_seed.py.")


if __name__ == "__main__":
    main()
