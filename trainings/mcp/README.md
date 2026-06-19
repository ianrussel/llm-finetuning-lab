# MCP deep dive (hands-on)

A self-paced course to fully understand the Model Context Protocol (MCP) by building it.
The goal is durable mastery of one thing: how an AI app (a host like Claude Code) connects
to external tools and data through a small, standard protocol, and how to write a server
that exposes your own tools, resources, and prompts.

You learn by doing. Each module has a short concept page and a runnable server you test
with the MCP Inspector, then wire into Claude Code.

## What MCP is, in one line

MCP is a standard wire protocol (JSON-RPC over stdio or HTTP) that lets any AI host talk
to any tool server, so you write an integration once and any MCP-aware app can use it. The
"USB-C port for AI apps" framing: one connector instead of a custom cable per app.

## Setup (once)

Python 3.10+ and the official SDK. From this folder:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You also need Node (for the Inspector, run via npx) and, for module 4, the Claude Code CLI.

## The path

| Module | What you build / learn | Run it with |
|--------|------------------------|-------------|
| 01 | Concepts: host/client/server, the 3 primitives, transports, JSON-RPC | (reading) |
| 02 | Your first server: two tools, no network | MCP Inspector |
| 03 | Add a resource and a prompt (all three primitives) | MCP Inspector |
| 04 | Connect a server to Claude Code (CLI + .mcp.json + /mcp) | Claude Code |
| 05 | Build a real server over the help-desk dataset | Inspector + Claude Code |

Work them in order. Each module folder has its own README with the exact commands and the
runnable `server.py`. Track your progress in PROGRESS.md.

## How to test any server in this course

Two ways, both used throughout:

1. MCP Inspector (a browser UI that talks to your server directly, no LLM needed):
   ```
   npx @modelcontextprotocol/inspector python module02_first_server/server.py
   ```
   Open the URL it prints, go to the Tools tab, call a tool, watch the JSON-RPC in Logs.

2. Claude Code (module 4 onward): register the server, then `/mcp` to confirm it loaded and
   ask Claude to use a tool.

## The mental model to hold

- A **host** (Claude Code) runs one **client** per **server** and aggregates their tools.
- A **server** exposes three kinds of capability: **tools** (the model calls them, they act),
  **resources** (read-only context the app pulls in), **prompts** (templates the user invokes).
- Messages are **JSON-RPC 2.0**; a connection starts with an **initialize** handshake where
  each side declares what it supports.
- **stdio** transport for local servers (subprocess, fast), **HTTP** for remote/shared ones.

Reference details and current commands were gathered from modelcontextprotocol.io and
code.claude.com; see module 01 for the conceptual map and each module for runnable code.
