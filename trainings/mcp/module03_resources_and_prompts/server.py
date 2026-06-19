"""Module 03: all three primitives in one server, a tool, resources, and a prompt.

  tool      @mcp.tool()      the model calls it to DO something (here: format text)
  resource  @mcp.resource()  read-only context the host pulls in (static + templated)
  prompt    @mcp.prompt()    a user-invoked template that returns instructions

Still offline and stdio. Test each in its own Inspector tab (see README).
"""

import logging
import sys

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("primitives-demo")

mcp = FastMCP("primitives-demo")


# --- TOOL: the model calls this to perform an action ------------------------
@mcp.tool()
def format_heading(text: str, level: int = 1) -> str:
    """Format text as a Markdown heading.

    Args:
        text: The heading text.
        level: Heading level 1-6 (default 1).
    """
    level = max(1, min(6, level))
    return f"{'#' * level} {text}"


# --- RESOURCE (static): read-only context, fixed URI ------------------------
@mcp.resource("guide://about")
def about() -> str:
    """A short description of what this server provides. The host can read this to
    understand the server without calling a tool."""
    return ("primitives-demo exposes one tool (format_heading), two resources "
            "(this guide and a templated greeting), and one prompt (summarize). "
            "It is a teaching server for MCP's three primitives.")


# --- RESOURCE (templated): the {name} in the URI becomes a function arg ------
@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """A personalized greeting resource. Reading greeting://Ian returns a greeting
    for Ian. Templated resources are how you expose parameterized read-only data
    (a file by path, a record by id, a forecast by city).

    Args:
        name: Who to greet.
    """
    return f"Hello, {name}. This text came from a templated MCP resource."


# --- PROMPT: a user-invoked template that returns instructions --------------
@mcp.prompt()
def summarize(text: str, style: str = "concise") -> str:
    """A reusable summarization workflow the user can invoke by name.

    Args:
        text: The text to summarize.
        style: 'concise', 'bullets', or 'eli5' (default concise).
    """
    how = {
        "concise": "in 2 to 3 sentences",
        "bullets": "as 3 to 5 short bullet points",
        "eli5": "in simple language a beginner would understand",
    }.get(style, "in 2 to 3 sentences")
    return f"Summarize the following text {how}.\n\nTEXT:\n{text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
