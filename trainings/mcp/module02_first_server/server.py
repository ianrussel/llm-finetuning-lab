"""Module 02: your first MCP server. Two tools, no network, runnable offline.

A tool is just an async function decorated with @mcp.tool(). FastMCP turns the
function name, type hints, and docstring into the tool's JSON schema automatically,
so the model knows the tool's name, arguments, and what it does. Run it under the
MCP Inspector (see this module's README) and call the tools from the Tools tab.

Rule for stdio servers: never print to stdout (it carries the JSON-RPC messages).
Log to stderr instead, as configured below.
"""

import logging
import sys

from mcp.server.fastmcp import FastMCP

# Logging goes to stderr so it never corrupts the stdout JSON-RPC stream.
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hello-mcp")

# The name shows up in the host's /mcp output and in logs.
mcp = FastMCP("hello-mcp")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum.

    Args:
        a: The first number.
        b: The second number.
    """
    log.info("add(%s, %s)", a, b)        # stderr, safe
    return a + b


@mcp.tool()
def text_stats(text: str) -> dict:
    """Return basic statistics about a piece of text.

    Args:
        text: The text to analyze.
    """
    words = text.split()
    return {
        "characters": len(text),
        "words": len(words),
        "lines": text.count("\n") + 1 if text else 0,
        "reversed": text[::-1],
    }


if __name__ == "__main__":
    # stdio transport: the host launches this file as a subprocess and talks to it
    # over stdin/stdout. This is the default for local servers.
    mcp.run(transport="stdio")
