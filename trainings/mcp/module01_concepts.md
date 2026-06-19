# Module 01: Concepts

Read this once, then keep it open while you build modules 02 to 05. The whole protocol fits
in a few ideas.

## The problem MCP solves

Before MCP, connecting an AI app to an external system (Slack, a database, your docs) meant a
custom integration per app and per system. N apps times M systems is a lot of glue, rebuilt
everywhere. MCP makes it N plus M: a system exposes one MCP server, and any MCP-aware app can
use it. One standard connector instead of a custom cable per pair.

## Three roles: host, client, server

- **Host** is the AI application that owns the conversation and coordinates everything. Claude
  Code, Claude Desktop, an IDE assistant, or your own agent.
- **Client** is a connector inside the host. The host creates one client per server, and each
  client holds exactly one connection to one server.
- **Server** is a program that exposes capabilities (tools, resources, prompts). It can run
  locally as a subprocess or remotely over HTTP.

So a host with three servers runs three clients and merges all their tools into one set the
model can pick from. You write servers; the host and clients already exist.

```
Host (Claude Code)
  client A  ── stdio ──>  server A (local python)
  client B  ── http  ──>  server B (remote SaaS)
  client C  ── stdio ──>  server C (local node)
```

## The three primitives (and who controls each)

A server exposes exactly three kinds of capability. The difference that matters is who decides
when each is used.

| Primitive | What it is | Controlled by | Use it for |
|-----------|-----------|---------------|------------|
| **Tool** | A function the model can call that DOES something | the model (with user approval) | actions and side effects: write a record, send a message, query an API, compute |
| **Resource** | Read-only data the app pulls in as context | the application/host | files, schemas, docs, a knowledge base, current state |
| **Prompt** | A reusable, parameterized instruction template | the user (invoked explicitly) | guided workflows: "plan a trip", "review this PR" |

A way to remember it: tools are verbs the model chooses, resources are nouns the app supplies,
prompts are recipes the user starts. Most learning servers begin with tools because they are
the most active and the easiest to see working.

Each primitive has list + use operations under the hood: `tools/list` + `tools/call`,
`resources/list` + `resources/read`, `prompts/list` + `prompts/get`.

## Transports: how bytes move

- **stdio** (standard input/output): the host launches the server as a subprocess and they
  exchange newline-delimited JSON over stdin/stdout. Local only, fastest, one client to one
  server. This is the default for everything you build locally in this course. One hard rule:
  on a stdio server, stdout is reserved for protocol messages, so all your logging must go to
  stderr or a file. A stray `print()` to stdout corrupts the stream.
- **streamable HTTP**: the host talks to a server over HTTP (with optional server-sent events
  for streaming). The server can run anywhere and serve many clients. Used for remote and
  shared/SaaS servers, with normal auth (OAuth, bearer tokens).

Pick stdio for local tools on your machine, HTTP for something hosted and shared.

## It is JSON-RPC 2.0 underneath

Every message is a JSON-RPC 2.0 object. Requests carry an `id` and get a matching response;
notifications have no `id` and get none. You rarely write this by hand (the SDK does it), but
seeing it once makes the Inspector logs readable.

A connection always starts with an **initialize handshake**:
1. the client sends `initialize` with its protocol version and capabilities,
2. the server replies with its protocol version and which primitives it offers,
3. the client sends an `initialized` notification, and normal calls begin.

That handshake is just version negotiation plus "here is what each side can do." After it, the
client can call `tools/list`, `resources/read`, and so on.

## Security, briefly (it matters from day one)

- Tools can act, so a host asks the user before running a new tool, and you should validate
  every tool input (never trust a path or query blindly).
- Keep secrets in environment variables, never in code or tool descriptions (descriptions are
  shown to the model and logged).
- Only connect servers you trust; a malicious server can expose harmful tools.

## Check yourself before module 02

You should be able to answer:
- What is the difference between a host, a client, and a server?
- When would you make something a tool versus a resource versus a prompt?
- Why must a stdio server never print to stdout?
- What does the initialize handshake accomplish?

If those are clear, go build one. Module 02 is a runnable server with two tools.
