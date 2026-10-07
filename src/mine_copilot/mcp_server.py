"""MCP server exposing the copilot's three tools to any MCP client (Claude Desktop, Claude Code, ...).

Usage: python -m mine_copilot.mcp_server   (stdio transport; needs `pip install -e ".[mcp]"`)

The tools, their JSON schemas and their guardrails (allow-listed filters, read-only DB, expired
sections hidden) are the same objects the agent uses. What does NOT come along: the agent loop's
post-answer grounding check. Over MCP the client's model writes the answer, so the agent's rules
are passed as server instructions, but nothing enforces them.
"""

import asyncio
import json

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from mine_copilot.agent.loop import SYSTEM
from mine_copilot.agent.tools import TOOL_SPECS, Tools, call_tool

READ_ONLY = types.ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True,
                                  openWorldHint=False)


def build_server(tools: Tools | None = None) -> Server:
    tools = tools or Tools()

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=s["name"], description=s["description"], inputSchema=s["parameters"],
                       annotations=READ_ONLY)
            for s in TOOL_SPECS])

    async def run_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        result = call_tool(tools, params.name, dict(params.arguments or {}))
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result, default=str))],
            structuredContent=result, isError="error" in result)

    return Server("mine-safety-copilot", instructions=SYSTEM,
                  on_list_tools=list_tools, on_call_tool=run_tool)


async def main() -> None:
    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
