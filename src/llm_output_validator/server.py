"""MCP server exposing llm-output-validator as an AI response validator.

Day-1 spike: a single hello-world tool wired up through FastMCP, proven with
an in-memory client test before any real validation logic is built on top.
See ARCHITECTURE.md for the full design and CLAUDE.md for the ship gates
this file is built under.
"""

from __future__ import annotations

import sys

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
except ImportError as exc:  # pragma: no cover - exercised via missing-extra path
    raise ImportError(
        "The MCP server requires the 'mcp' extra. Install with: "
        "pip install 'llm-output-validator[mcp]'"
    ) from exc

mcp = FastMCP("llm_validator_mcp")

_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


@mcp.tool(name="llmval_ping", annotations=_READ_ONLY)
def llmval_ping() -> str:
    """Health-check tool. Returns a fixed string to confirm the server is wired up.

    This is a day-1 spike placeholder — it will be replaced by
    ``llmval_validate_response`` once the generic checks and profile runner
    land (see the ship gates in CLAUDE.md).
    """
    return "llm_validator_mcp: ok"


def main() -> None:
    """Entry point for the ``llm-validate-mcp`` console script.

    Runs over stdio only. Per the security guidance in ARCHITECTURE.md, this
    process must never write anything to stdout outside the MCP protocol
    itself — logging, if added later, goes to stderr.
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(0)
