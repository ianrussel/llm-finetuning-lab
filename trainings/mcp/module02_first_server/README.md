# Module 02: your first MCP server

Goal: write a server that exposes two tools, then call them yourself in the MCP Inspector.
This is the smallest useful thing in MCP, and it teaches the core pattern: a decorated
Python function becomes a tool the model can call.

## The code

`server.py` defines two tools:
- `add(a, b)` returns the sum.
- `text_stats(text)` returns character/word/line counts and the reversed string.

The whole server is the `@mcp.tool()` decorator plus `mcp.run(transport="stdio")`. FastMCP
reads each function's name, type hints, and docstring and builds the tool schema for you, so
the function signature IS the API contract the model sees. That is why the type hints and the
docstring matter: they are not decoration, they are the interface.

## Run it in the Inspector

From the `mcp/` folder, with your venv active and deps installed:

```
npx @modelcontextprotocol/inspector python module02_first_server/server.py
```

This launches your server as a subprocess and opens a browser UI. Then:
1. Wait for "Connected".
2. Open the **Tools** tab. You should see `add` and `text_stats` with the argument fields the
   SDK generated from your type hints.
3. Call `add` with a=2, b=3 and confirm you get 5.
4. Call `text_stats` with some text and read the result.
5. Open the **Logs** tab and look at the raw JSON-RPC: the `tools/list` response (your schemas)
   and the `tools/call` request/response. This is the protocol from module 01, made concrete.

## Things to try (to build intuition)

- Change a docstring or a type hint, restart the Inspector, and watch the tool schema change.
- Add a `print("hi")` to a tool, call it, and watch it BREAK the stdio stream. Then change it
  to `log.info("hi")` (stderr) and see it work again. This is the one stdio rule from module 01.
- Add a third tool of your own (for example `multiply(a, b)`), restart, and call it.

## What to take away

- A tool = an `@mcp.tool()` function; its signature and docstring become the schema.
- You can fully test a server with the Inspector, no LLM required.
- stdout is sacred on stdio servers; log to stderr.

Next: module 03 adds the other two primitives, a resource and a prompt.
