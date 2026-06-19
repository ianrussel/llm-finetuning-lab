# Module 05: build a real server over your own data

Goal: put everything together on real data. This server exposes the help-desk `issues.csv`
(the Track C dataset) through MCP, so a host can look up tickets and stats by asking, instead
of you writing a query each time. It is the same shape as a server over a company database.

## The code

`server.py` loads `issues.csv` once (indexed by ticket id) and exposes:
- tool `get_ticket(issue_id)` -> key fields for one ticket,
- tool `resolution_counts()` -> counts by resolution outcome,
- tool `find_tickets(resolution, limit)` -> ids with a given resolution,
- resource `helpdesk://summary` -> dataset size + resolution distribution.

Data path: it reads `HELPDESK_CSV` if set, else defaults to
`../../task dataset generation pipeline/issues.csv` relative to the file. If the CSV is
missing, the tools return a clear message rather than crashing, so you always get feedback.

## Run it in the Inspector

From the `mcp/` folder:

```
npx @modelcontextprotocol/inspector python module05_build_your_own_helpdesk/server.py
```

If your `issues.csv` lives elsewhere, pass it through the environment:

```
HELPDESK_CSV="/abs/path/to/issues.csv" npx @modelcontextprotocol/inspector python module05_build_your_own_helpdesk/server.py
```

Then:
1. **Resources** -> read `helpdesk://summary` to confirm it loaded (you should see ~66,691
   tickets and the resolution distribution).
2. **Tools** -> call `get_ticket` with issue_id=1004364 (the worked example from the dataset),
   `resolution_counts()`, and `find_tickets` with resolution="Won't Do", limit=3.

## Connect it to Claude Code

```
claude mcp add helpdesk --env HELPDESK_CSV="/abs/path/to/issues.csv" -- python /ABSOLUTE/PATH/TO/trainings/mcp/module05_build_your_own_helpdesk/server.py
```

(Point `command` at the venv python that has `mcp` installed if needed.) Then in a session:

```
/mcp                      # confirm "helpdesk" is connected
How many tickets are Won't Do?      # Claude calls resolution_counts()
Show me ticket 1004364.             # Claude calls get_ticket(1004364)
```

There is a ready `.mcp.json` entry for this server in module 04's `mcp.json.example`.

## Things to try (and where to go next)

- Add a tool `tickets_by_priority()` mirroring `resolution_counts()`.
- Add a templated resource `helpdesk://ticket/{issue_id}` that returns the same data as
  `get_ticket` but as a read-only resource, and notice the control difference (the app pulls a
  resource; the model calls a tool).
- The real power: join the other tables. Track C's `link.py` already builds a ticket's full
  linked context (snapshots + change history + utterances). A natural next server wraps that as
  a `get_ticket_context(issue_id)` tool, so a host can pull a ticket's whole story in one call.

## What to take away

- A useful MCP server is often just "load data once, expose a few well-named tools + a summary
  resource." The hard part is naming and shaping the outputs, not the protocol.
- Env vars (here `HELPDESK_CSV`) are how you configure a server per deployment without changing
  code, and how you keep secrets out of the code.
- You now have the full loop: write a server, test it in the Inspector, connect it to Claude
  Code, and use it in conversation. That is MCP end to end.
