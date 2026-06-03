"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools."""

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


class MCPToolWrapper(Tool):
    """一个适配器，将外部 MCP (Model Context Protocol) 服务器提供的工具封装成系统原生的 Tool 对象，以便被大语言模型（LLM）统一调用。
    Wraps a single MCP server tool as a nanobot Tool."""

    def __init__(self, session, server_name: str, tool_def, tool_timeout: int = 30):
        self._session = session
        self._original_name = tool_def.name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._tool_timeout = tool_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters
    # 不包含真实的业务逻辑，而是通过 self._session.call_tool(...) 将参数原封不动地转发给底层的 MCP 服务器执行
    async def execute(self, **kwargs: Any) -> str:
        from mcp import types
        try:
            # 超时熔断机制：使用 asyncio.wait_for 包装了远程调用。如果 MCP 服务器响应时间超过了 self._tool_timeout（默认 30 秒），
            # 会捕获 TimeoutError 并返回一个超时提示字符串，从而防止整个智能体系统因外部服务卡死而永久阻塞
            result = await asyncio.wait_for(
                self._session.call_tool(self._original_name, arguments=kwargs),
                timeout=self._tool_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("MCP tool '{}' timed out after {}s", self._name, self._tool_timeout)
            return f"(MCP tool call timed out after {self._tool_timeout}s)"
        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts) or "(no output)"


async def connect_mcp_servers(
    mcp_servers: dict, registry: ToolRegistry, stack: AsyncExitStack
) -> None:
    """负责连接所有配置的 MCP服务器，并将其提供的工具注册到智能体系统中
    Connect to configured MCP servers and register their tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    for name, cfg in mcp_servers.items():
        try:
            # 标准输入输出 (Stdio)
            if cfg.command:
                params = StdioServerParameters(
                    command=cfg.command, args=cfg.args, env=cfg.env or None
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            # HTTP 流式传输 (SSE)
            elif cfg.url:
                from mcp.client.streamable_http import streamable_http_client
                # Always provide an explicit httpx client so MCP HTTP transport does not
                # inherit httpx's default 5s timeout and preempt the higher-level tool timeout.
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=cfg.headers or None,
                        follow_redirects=True,
                        timeout=None,
                    )
                )
                # 强依赖于 AsyncExitStack (stack)，确保在系统关闭时，
                # 所有打开的流（read/write）和客户端会话（ClientSession）都能被一并自动且优雅地关闭。
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(cfg.url, http_client=http_client)
                )
            else:
                logger.warning("MCP server '{}': no command or url configured, skipping", name)
                continue

            session = await stack.enter_async_context(ClientSession(read, write))
            # 连接成功后，通过 await session.initialize() 完成握手。
            await session.initialize()
            # 获取该服务器暴露的所有可用工具
            tools = await session.list_tools()
            # 遍历抓取到的每一个工具定义（tool_def），使用 MCPToolWrapper 适配器将其包装成智能体可以理解的标准 Tool。
            for tool_def in tools.tools:
                wrapper = MCPToolWrapper(session, name, tool_def, tool_timeout=cfg.tool_timeout)
                # 调用 registry.register(wrapper) 注入到全局工具库中，供 LLM 分析和调用
                registry.register(wrapper)
                logger.debug("MCP: registered tool '{}' from server '{}'", wrapper.name, name)

            logger.info("MCP server '{}': connected, {} tools registered", name, len(tools.tools))
        except Exception as e:
            logger.error("MCP server '{}': failed to connect: {}", name, e)
