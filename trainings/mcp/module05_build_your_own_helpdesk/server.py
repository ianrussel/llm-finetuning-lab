"""Module 05: a real MCP server over the help-desk dataset.

This exposes the issues.csv from the Track C pipeline through MCP, so a host (Claude
Code, or the Inspector) can look up tickets and stats without you writing a query each
time. It is the same shape as a server you would build over a real company database:
load data once, expose a few well-named tools and a summary resource.

Data path resolution: set HELPDESK_CSV to the issues.csv you want, otherwise it defaults
to ../../task dataset generation pipeline/issues.csv relative to this file. If the file is
absent, the tools return a clear message instead of crashing.
"""

import csv
import logging
import os
import sys
from collections import Counter

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("helpdesk-mcp")

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.normpath(
    os.path.join(_HERE, "..", "..", "task dataset generation pipeline", "issues.csv"))
CSV_PATH = os.environ.get("HELPDESK_CSV", DEFAULT_CSV)

mcp = FastMCP("helpdesk")
csv.field_size_limit(10 ** 7)

# Load issues once, indexed by a canonical id ('1004364.0' -> '1004364').
_ISSUES = {}


def _nid(x):
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def _load():
    if _ISSUES or not os.path.exists(CSV_PATH):
        return
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            _ISSUES[_nid(row.get("id"))] = row
    log.info("loaded %d issues from %s", len(_ISSUES), CSV_PATH)


_load()


@mcp.tool()
def get_ticket(issue_id: str) -> dict:
    """Look up one help-desk ticket by its id and return its key fields.

    Args:
        issue_id: The ticket id (e.g. 1004364).
    """
    if not _ISSUES:
        return {"error": f"dataset not loaded; set HELPDESK_CSV (looked at {CSV_PATH})"}
    row = _ISSUES.get(_nid(issue_id))
    if not row:
        return {"error": f"no ticket with id {issue_id}"}
    return {
        "id": _nid(row.get("id")),
        "type": row.get("issue_type"),
        "priority": row.get("issue_priority"),
        "status": row.get("issue_status"),
        "resolution": row.get("issue_resolution"),
        "contributors": row.get("issue_contr_count"),
        "comments": row.get("issue_comments_count"),
        "total_handling_seconds": row.get("wf_total_time"),
        "processing_steps": row.get("processing_steps"),
    }


@mcp.tool()
def resolution_counts() -> dict:
    """Count tickets by resolution outcome (Done, Won't Do, Duplicate, ...)."""
    if not _ISSUES:
        return {"error": f"dataset not loaded; set HELPDESK_CSV (looked at {CSV_PATH})"}
    c = Counter((r.get("issue_resolution") or "(none)") for r in _ISSUES.values())
    return dict(c.most_common())


@mcp.tool()
def find_tickets(resolution: str, limit: int = 5) -> list:
    """Return up to `limit` ticket ids whose resolution matches.

    Args:
        resolution: Exact resolution value, e.g. "Won't Do" or "Done".
        limit: Maximum ids to return (default 5).
    """
    if not _ISSUES:
        return [f"dataset not loaded; set HELPDESK_CSV (looked at {CSV_PATH})"]
    out = [nid for nid, r in _ISSUES.items()
           if (r.get("issue_resolution") or "").strip() == resolution]
    return out[:max(1, limit)]


@mcp.resource("helpdesk://summary")
def summary() -> str:
    """A read-only overview of the loaded dataset: size and resolution distribution."""
    if not _ISSUES:
        return f"Dataset not loaded. Set HELPDESK_CSV (looked at {CSV_PATH})."
    c = Counter((r.get("issue_resolution") or "(none)") for r in _ISSUES.values())
    dist = ", ".join(f"{k}={v}" for k, v in c.most_common())
    return f"{len(_ISSUES)} tickets loaded from {os.path.basename(CSV_PATH)}. Resolution: {dist}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
