"""Serialize one issue's LINKED context into a compact, leakage-aware prompt block.

Takes the joined record from link.HelpDeskDB.get_issue() and renders a short, readable
structured context: issue metadata, workflow time-in-state, and a pre-resolution
handling path from the change history. It is survey-aware: anything the field survey
marks as leakage (the resolution/status fields and outcome-coupled workflow states)
is never emitted, so the model cannot cheat its way to the label.

The default survey below is a safe starting point; phase1_seed/survey.py proposes a
refined one that the human confirms (data/field_survey.json).
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common_c as common

SURVEY_PATH = os.path.join(common.DATA, "field_survey.json")

# Metadata fields offered in the header, in display order. Anything also listed in
# the survey's leakage_exclude is dropped.
HEADER_FIELDS = [
    ("issue_type", "type"),
    ("issue_priority", "priority"),
    ("issue_contr_count", "contributors"),
    ("issue_comments_count", "comments"),
    ("processing_steps", "processing_steps"),
]

# Safe default: exclude the label itself, its timestamp, the final status, and the
# workflow states that ARE the outcome (a Won't Do ticket is the one that gets
# rejected/cancelled; a Done ticket the one that reaches done).
DEFAULT_SURVEY = {
    "task": "resolution_binary",
    "leakage_exclude": ["issue_resolution", "issue_resolution_date", "issue_status",
                        "last_change_date", "issue_created", "id", "issue_num",
                        "issue_reporter", "issue_assignee", "started", "ended"],
    "workflow": {"exclude_states": ["rejected", "cancelled", "done", "closed"], "top_k": 12},
    "history": {"exclude_terminal_status": ["resolved", "closed", "done", "rejected",
                                            "cancelled", "validation"]},
    "include_utterances": False,
    "confirmed": False,
}


def load_survey():
    """Return the confirmed survey if present, else the safe default."""
    if os.path.exists(SURVEY_PATH):
        with open(SURVEY_PATH) as f:
            return json.load(f)
    return dict(DEFAULT_SURVEY)


def require_confirmed_survey():
    """Hard gate for downstream generation/seed steps (mirrors sdg.preflight)."""
    s = load_survey()
    if not s.get("confirmed"):
        raise SystemExit(
            "field survey not confirmed. Run phase1_seed/survey.py, review "
            f"{SURVEY_PATH}, set \"confirmed\": true, then re-run.")
    return s


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def humanize(seconds):
    s = _num(seconds)
    if not s or s <= 0:
        return None
    if s < 90:
        return f"{int(s)}s"
    if s < 5400:
        return f"{s/60:.0f}m"
    if s < 172800:
        return f"{s/3600:.1f}h"
    return f"{s/86400:.1f}d"


def _wf_states(issue, exclude):
    """[(state, seconds, passes)] for nonzero workflow states not in exclude."""
    out = []
    for k in issue:
        if not k.startswith("wf_") or k == "wf_total_time":
            continue
        state = k[3:]
        if state in exclude:
            continue
        secs = _num(issue.get(k))
        if not secs or secs <= 0:
            continue
        passes = int(_num(issue.get("wfe_" + state)) or 0)
        out.append((state, secs, passes))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _handling_path(linked, exclude_terminal):
    """A compact pre-resolution handling summary from snapshots + change history,
    with terminal/resolving statuses removed so the outcome is not revealed."""
    turns = len(linked["snapshots"])
    statuses = []
    reopened = 0
    for h in linked["history"]:
        if h.get("field") != "status":
            continue
        v = (h.get("value") or "").strip()
        if v == "reopened":
            reopened += 1
        if v in exclude_terminal or not v:
            continue
        if not statuses or statuses[-1] != v:
            statuses.append(v)
    parts = []
    if turns:
        parts.append(f"handled across {turns} assignee turn(s)")
    if statuses:
        parts.append("status path: " + " -> ".join(statuses))
    if reopened:
        parts.append(f"reopened {reopened}x")
    return parts


def serialize_issue(linked, survey=None, max_utterances=12):
    """Render the linked context block. `linked` is link.get_issue(id) output."""
    survey = survey or load_survey()
    issue = linked["issue"]
    if issue is None:
        return "(issue not found)"
    exclude = set(survey.get("leakage_exclude", []))

    # Header
    head = []
    for col, label in HEADER_FIELDS:
        if col in exclude:
            continue
        val = issue.get(col)
        if val is None or str(val).strip() in ("", "nan"):
            continue
        if col in ("issue_contr_count", "issue_comments_count", "processing_steps"):
            val = int(_num(val) or 0)
        head.append(f"{label}={val}")
    lines = ["Ticket context:", "  " + ", ".join(head) if head else "  (no metadata)"]

    # Workflow time-in-state
    wf_ex = set(survey.get("workflow", {}).get("exclude_states", []))
    top_k = survey.get("workflow", {}).get("top_k", 12)
    states = _wf_states(issue, wf_ex)[:top_k]
    if states:
        rendered = ", ".join(
            f"{s}={humanize(secs)}" + (f"(x{passes})" if passes > 1 else "")
            for s, secs, passes in states)
        lines.append("  workflow time-in-state: " + rendered)
        total = humanize(issue.get("wf_total_time"))
        if total:
            lines.append(f"  total handling time: {total}")
    else:
        lines.append("  workflow time-in-state: none recorded")

    # Pre-resolution handling path
    path = _handling_path(linked, set(survey.get("history", {}).get("exclude_terminal_status", [])))
    if path:
        lines.append("  " + "; ".join(path))

    # Optional conversation text (off for the resolution task)
    if survey.get("include_utterances") and linked["utterances"]:
        lines.append("  conversation (excerpt):")
        for u in linked["utterances"][:max_utterances]:
            body = (u.get("actionbody") or "").strip()
            if body:
                lines.append(f"    [{u.get('author_role','?')}] {body}")

    return "\n".join(lines)


if __name__ == "__main__":
    import link
    db = link.HelpDeskDB()
    print(serialize_issue(db.get_issue(1004364)))
    print("\n--- with utterances ---")
    s = dict(DEFAULT_SURVEY); s["include_utterances"] = True
    print(serialize_issue(db.get_issue(1004364), s, max_utterances=6))
