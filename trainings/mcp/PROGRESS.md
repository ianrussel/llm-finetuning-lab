# MCP deep dive: progress

My hands-on log for learning the Model Context Protocol by building servers. The goal is to
be able to explain host/client/server, the three primitives, and the transports, and to
write and connect my own MCP server from scratch.

Status: [ ] not started, [~] in progress, [x] done

| Module | What | Status |
|--------|------|--------|
| 01 | Concepts (architecture, primitives, transports, JSON-RPC) | [ ] |
| 02 | First server: two tools, tested in the Inspector | [ ] |
| 03 | Add a resource and a prompt | [ ] |
| 04 | Connect a server to Claude Code | [ ] |
| 05 | Real server over the help-desk dataset | [ ] |

## Setup
- [ ] Python 3.10+ venv created, `pip install -r requirements.txt`
- [ ] Node available (for `npx @modelcontextprotocol/inspector`)
- [ ] Claude Code CLI available (for module 4)

## Module 01: Concepts
- [ ] Can explain host vs client vs server in my own words
- [ ] Can say when to use a tool vs a resource vs a prompt
- [ ] Can explain stdio vs HTTP transport and when each is used
- [ ] Understand the initialize/capabilities handshake at a high level

Notes:

## Module 02: First server
- [ ] Ran server.py under the MCP Inspector
- [ ] Called both tools from the Tools tab and saw the result + raw JSON-RPC
- [ ] Understood how the type hints + docstring became the tool schema

Notes:

## Module 03: Resources and prompts
- [ ] Read the static and templated resources in the Inspector Resources tab
- [ ] Invoked the prompt with arguments in the Prompts tab
- [ ] Can articulate the control difference (model vs app vs user controlled)

Notes:

## Module 04: Connect to Claude Code
- [ ] Registered the server with `claude mcp add`
- [ ] Confirmed it with `/mcp` (tools/resources/prompts listed)
- [ ] Asked Claude to use a tool and watched the namespaced call
- [ ] Tried a project-scoped `.mcp.json`

Notes:

## Module 05: Real server (help-desk)
- [ ] Pointed the server at issues.csv and looked up a ticket via a tool
- [ ] Exposed a resource (e.g. resolution counts) and read it
- [ ] Connected it to Claude Code and asked a data question through the tool
- [ ] Noted what I would add next (more tables, the linked context from Track C)

Notes:

## What surprised me / open questions
-
