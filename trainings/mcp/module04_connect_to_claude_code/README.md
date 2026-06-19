# Module 04: connect a server to Claude Code

Goal: take the module 03 server out of the Inspector and into a real host. Once registered,
Claude Code discovers its tools and can call them during a conversation. This is the payoff,
your server becomes something the model can use.

## Two ways to register a server

### A) The CLI (`claude mcp add`)

Stdio (local) server, the kind you have been building:

```
claude mcp add primitives-demo -- python /ABSOLUTE/PATH/TO/trainings/mcp/module03_resources_and_prompts/server.py
```

Notes:
- Everything after `--` is the exact command Claude Code will run to launch your server.
- Use an ABSOLUTE path. Claude Code runs from different working directories, so a relative
  path usually fails.
- If your server needs a venv, point at that venv's python, e.g.
  `.../ai_training/.venv/bin/python /ABSOLUTE/PATH/.../server.py`, so the `mcp` package is found.
- Pass environment variables with `--env KEY=value` before the `--`.

Scopes (where the registration is saved) with `--scope`:
- `local` (default): just you, just this project.
- `project`: shared with teammates via a checked-in `.mcp.json`.
- `user`: just you, across all your projects.

Remote (HTTP) server:

```
claude mcp add --transport http some-remote https://mcp.example.com/mcp
```

### B) A project `.mcp.json` file

For project scope, servers live in `.mcp.json` at the project root (commit it to share with the
team). See `mcp.json.example` in this folder for the format; replace the `/ABSOLUTE/PATH/...`
placeholders with real paths. Copy it to your project root as `.mcp.json`:

```
cp mcp.json.example /path/to/your/project/.mcp.json   # then edit the paths
```

Each entry needs `type` (`stdio` or `http`), and for stdio a `command` + `args` (+ optional
`env`); for http a `url`.

## Verify it loaded

List from the shell:

```
claude mcp list
```

Or inside a Claude Code session:

```
/mcp
```

You should see `primitives-demo` connected with its tool, resources, and prompt. Status symbols:
`done`/check = connected, `!` = needs authentication, `x` = failed to connect. Use
`claude mcp get primitives-demo` to print the exact launch command if it failed (usually a
wrong path or a python that lacks the `mcp` package).

## Use it

In a session, just ask in plain language:

```
Use the primitives-demo server to format "Quarterly Report" as a level 2 heading.
```

Claude discovers the tool, asks permission the first time, and calls it. In the transcript the
call is namespaced by server, like `primitives-demo__format_heading`, so you can see which
server provided it. Resources and prompts from the server are available too (prompts show up as
slash-style options you can invoke).

## Troubleshooting

- "Failed to connect": run the exact command from `claude mcp get <name>` in your shell and read
  the error. Usually the path is wrong or the python has no `mcp` installed.
- "No tools appear": the server started but registered nothing, often a missing env var. Check
  with the Inspector first (module 02/03), then add the env var in the CLI or `.mcp.json`.
- First-run timeout: a cold `npx`/`uv` can be slow. Retry, or raise the timeout with
  `MCP_TIMEOUT=60000 claude`.

## What to take away

- `claude mcp add ... -- <command>` registers a stdio server; `.mcp.json` does it per project.
- `/mcp` and `claude mcp list` show status; tools are namespaced by server.
- Absolute paths and the right python (with `mcp` installed) are the two things people get wrong.

Next: module 05 builds a server over your real help-desk data and connects it here.
