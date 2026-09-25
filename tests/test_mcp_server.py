"""Day-1 spike test: prove the MCP server is wired up correctly before any
real validation logic is built on top of it.

Uses an in-memory client/server connection (no subprocess, no stdio pipe)
per the MCP SDK's testing pattern. This settles the mcp SDK pin and the
anyio test setup early, as required by the ship gates in CLAUDE.md.
"""

from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from llm_output_validator.server import mcp

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_tool_list_includes_ping() -> None:
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools.tools}
        assert "llmval_ping" in names


async def test_ping_tool_returns_ok() -> None:
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("llmval_ping", {})
        assert result.isError is not True
        text_blocks = [block.text for block in result.content if block.type == "text"]
        assert any("ok" in text for text in text_blocks)


async def test_ping_tool_annotations_are_read_only() -> None:
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        tools = await client.list_tools()
        ping = next(tool for tool in tools.tools if tool.name == "llmval_ping")
        assert ping.annotations is not None
        assert ping.annotations.readOnlyHint is True
        assert ping.annotations.destructiveHint is False
        assert ping.annotations.idempotentHint is True
        assert ping.annotations.openWorldHint is False


async def test_ping_tool_has_no_input_parameters() -> None:
    # llmval_ping takes no arguments; this pins that shape so later tools
    # (llmval_validate_response, added in PR-A/PR-B) can be contrasted
    # against it deliberately rather than by accident.
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        tools = await client.list_tools()
        ping = next(tool for tool in tools.tools if tool.name == "llmval_ping")
        assert ping.inputSchema.get("properties", {}) == {}
