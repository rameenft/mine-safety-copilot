"""MCP server over an in-process client: same tools, schemas and guardrails as the agent."""

import asyncio

import pytest
from conftest import ACC

pytest.importorskip("mcp")
from mcp import Client

from mine_copilot.agent.tools import TOOL_SPECS
from mine_copilot.mcp_server import build_server


def _session(tools, fn):
    async def go():
        async with Client(build_server(tools)) as client:
            return await fn(client)
    return asyncio.run(go())


def test_lists_the_agents_tools_as_read_only(tools):
    listed = _session(tools, lambda c: c.list_tools()).tools
    assert [t.name for t in listed] == [s["name"] for s in TOOL_SPECS]
    assert all(t.annotations.read_only_hint for t in listed)
    stats = next(t for t in listed if t.name == "accident_stats")
    assert stats.input_schema["properties"]["group_by"]["enum"]  # allow-list survives


def test_call_tool_returns_same_result_as_agent_tool(tools):
    res = _session(tools, lambda c: c.call_tool("accident_stats", {"severity": "fatal"}))
    assert not res.is_error
    assert res.structured_content["total_matching"] == int((ACC["SEVERITY"] == "fatal").sum())


def test_bad_input_is_a_tool_error_not_a_crash(tools):
    res = _session(tools, lambda c: c.call_tool("accident_stats", {"group_by": "mine_owner"}))
    assert res.is_error and "group_by must be one of" in res.content[0].text
