"""Load and link the help-desk knowledge base by issue id.

The corpus is four related CSVs (see FEATURES.md). They link on the issue id:
  issues.csv               one row per issue            key: id
  issues_snapshot.csv      per-assignee turn snapshots  key: id   (many per issue)
  issues_change_history.csv raw status/assignee log      key: issueid -> id
  sample_utterances.csv    masked conversation text      key: issueid -> id (sample only)

`HelpDeskDB.get_issue(id)` returns the issue row plus its ordered snapshots, change
history, and utterances, so the rest of the pipeline can pull a ticket's full linked
context in one call. Run this file directly for a self-check against EXAMPLE.md.
"""

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common_c as common

# issues.csv has 58 wide columns; raise the field-size limit to be safe on big rows.
csv.field_size_limit(10 ** 7)


def _nid(x):
    """Canonical issue id as an int-string ('1004364.0' -> '1004364')."""
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_int(x, default=0):
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return default


class HelpDeskDB:
    def __init__(self, raw=common.RAW):
        self.issues = {}
        for r in _read(os.path.join(raw, "issues.csv")):
            self.issues[_nid(r["id"])] = r

        self.snapshots = {}
        snap_path = os.path.join(raw, "issues_snapshot.csv")
        if os.path.exists(snap_path):
            for r in _read(snap_path):
                self.snapshots.setdefault(_nid(r["id"]), []).append(r)
            for k in self.snapshots:
                self.snapshots[k].sort(key=lambda r: _to_int(r.get("turn")))

        self.history = {}
        hist_path = os.path.join(raw, "issues_change_history.csv")
        if os.path.exists(hist_path):
            for r in _read(hist_path):
                self.history.setdefault(_nid(r["issueid"]), []).append(r)
            for k in self.history:
                self.history[k].sort(key=lambda r: r.get("created", ""))

        self.utterances = {}
        utt_path = os.path.join(raw, "sample_utterances.csv")
        if os.path.exists(utt_path):
            for r in _read(utt_path):
                self.utterances.setdefault(_nid(r["issueid"]), []).append(r)
            for k in self.utterances:
                self.utterances[k].sort(
                    key=lambda r: (_to_int(r.get("comment_seq")), _to_int(r.get("utr_seq"))))

    def get_issue(self, issue_id):
        nid = _nid(issue_id)
        return {
            "id": nid,
            "issue": self.issues.get(nid),
            "snapshots": self.snapshots.get(nid, []),
            "history": self.history.get(nid, []),
            "utterances": self.utterances.get(nid, []),
        }

    def ids_with_resolution(self, label):
        """Issue ids whose resolution equals `label` (exact)."""
        return [nid for nid, r in self.issues.items()
                if (r.get("issue_resolution") or "").strip() == label]

    def __len__(self):
        return len(self.issues)


def _selfcheck():
    db = HelpDeskDB()
    print(f"loaded issues={len(db.issues)} snapshots_keys={len(db.snapshots)} "
          f"history_keys={len(db.history)} utterance_keys={len(db.utterances)}")
    assert len(db.issues) == len(set(db.issues)), "issue id not unique"
    ex = db.get_issue(1004364)  # the worked example in EXAMPLE.md
    iss = ex["issue"]
    print("\n=== issue 1004364 (compare to EXAMPLE.md) ===")
    print(f"  resolution={iss['issue_resolution']!r} status={iss['issue_status']!r} "
          f"contributors={iss['issue_contr_count']} comments={iss['issue_comments_count']}")
    print(f"  snapshots (turns): {len(ex['snapshots'])} -> assignees "
          f"{[s['issue_assignee'] for s in ex['snapshots']]}")
    print(f"  change-history events: {len(ex['history'])}")
    for h in ex["history"]:
        print(f"    {h['created']}  {h['field']}={h['value']}")
    print(f"  utterances: {len(ex['utterances'])} "
          f"(comments={len(set(u['comment_seq'] for u in ex['utterances']))})")
    print("  expected per EXAMPLE.md: Done/closed, 2 contributors, 6 comments, "
          "2 snapshot turns (4hghq, 4sii), 5 history events")


if __name__ == "__main__":
    _selfcheck()
