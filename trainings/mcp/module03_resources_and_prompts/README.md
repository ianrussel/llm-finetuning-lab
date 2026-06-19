# Module 03: resources and prompts

Goal: see all three primitives in one server and feel the difference between them. Module 02
was tools only; here you add a resource (read-only context) and a prompt (a user-invoked
template), so you have the full vocabulary.

## The code

`server.py` exposes:
- a **tool** `format_heading(text, level)` (the model calls it to do something),
- a **static resource** `guide://about` (fixed read-only text),
- a **templated resource** `greeting://{name}` (the `{name}` in the URI becomes an argument),
- a **prompt** `summarize(text, style)` (a template the user invokes; it returns instructions).

The point is the control difference from module 01, now visible:
- the model decides to call the **tool**,
- the app reads a **resource** to add context,
- the user picks a **prompt** to start a workflow.

## Run it in the Inspector

From the `mcp/` folder:

```
npx @modelcontextprotocol/inspector python module03_resources_and_prompts/server.py
```

Then explore three tabs:
1. **Tools** -> call `format_heading` with text="Hello", level=2, see `## Hello`.
2. **Resources** -> read `guide://about` (static). For the templated one, read
   `greeting://Ian` (or any name) and see the personalized text. Templated resources show as
   a URI template you fill in.
3. **Prompts** -> select `summarize`, give it some `text` and a `style` (try `bullets`), and
   see the rendered instruction string it produces. Note the prompt does not summarize; it
   returns the instructions a host would hand to the model. The user starts it; the model then
   acts on it.

## Things to try

- Read `greeting://` with different names and notice it is the same resource, parameterized.
- Change the `summarize` styles or add a new one, restart, and re-invoke.
- Ask yourself for each primitive: who triggers it, and does it act or just supply text?

## What to take away

- Tools act (model-driven), resources supply context (app-driven), prompts template a
  workflow (user-driven). Same server, three different control models.
- Templated URIs (`scheme://{param}`) are how you expose parameterized read-only data.

Next: module 04 takes a server out of the Inspector and into Claude Code.
