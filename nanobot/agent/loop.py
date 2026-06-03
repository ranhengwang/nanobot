"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """为特定的工具（Tools）动态注入当前对话的路由上下文信息。
        Update context for all tools that need routing info."""
        # 当系统收到一条新消息准备开始处理时，Agent 需要让某些特定的工具知道“当前正在服务哪个渠道的哪个用户”
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                # 便大模型调用发送消息工具时，底层知道该把消息发送到哪个聊天窗口或具体回复哪条消息
                message_tool.set_context(channel, chat_id, message_id)

        # 尝试从工具注册表（self.tools）中获取特定的工具，如果存在，则调用它们的 set_context 方法注入上下文：
        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """移除部分具有推理能力（Reasoning）的模型在回复正文中嵌入的 <think>...</think> 内部思考片段
        Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """将 Agent 发起的工具调用列表（tool_calls）格式化为一个简短的提示字符串
        Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            # 提取传入参数字典（tc.arguments）中的第一个参数的值作为代表展示
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            # 如果这个值是纯字符串，判断它的长度，根据长度返回不同的东西
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """负责执行大语言模型（LLM）的 “思考 -> 调用工具 -> 观察结果 -> 再思考” 的完整迭代循环
        Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        # 通过一个 while 循环不断与 LLM 进行多轮交互，直至得出最终结论或达到最大防死循环的迭代次数
        while iteration < self.max_iterations:
            iteration += 1
            # 将当前累计的 messages（包含系统提示、历史记忆、用户当前输入及之前的工具执行结果）以及系统当前支持的 tools 定义发给 LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # 如果有工具调用，那么就解析 LLM 的回复，提取工具调用信息（工具名称和参数），然后起执行对应的工具
            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                # 将 LLM 这次“决定调用某个工具”的动作作为一条 Assistant 消息追加到上下文中。
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                # 遍历 LLM 请求的所有工具，在本地环境实际执行（self.tools.execute）。
                # 得到结果后，使用 add_tool_result 追加到上下文的 messages 列表中
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = self._strip_think(response.content)
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """不断从消息总线上监听用户输入，获取一个输入后，驱动agent处理流程，然后将结果投递回总线，供通信渠道返回给用户。
        Run the agent loop, processing messages from the bus."""
        self._running = True
        # 初始化连接配置好的 MCP 外部工具服务器
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                # 阻塞监听：使用 asyncio.wait_for 从消息总线 (self.bus.consume_inbound()) 尝试拉取一条入站消息
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    # 成功获取消息后，将其交由 _process_message(msg) 进行完整的上下文构建、LLM 思考与工具调用流程。
                    response = await self._process_message(msg)
                    if response is not None:
                        # 处理完毕后得到 response（出站消息），使用 self.bus.publish_outbound(response) 将结果投递回总线，供具体的通信渠道返回给用户。
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id, content="", metadata=msg.metadata or {},
                        ))
                except Exception as e:
                    logger.error("Error processing message: {}", e)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

# 在异步环境中，同一个会话（session_key）可能会同时触发多次历史记忆压缩与保存操作（
# 比如用户连续快速发送多条消息，或者执行 /new 命令清理记忆时）。.
# 为了防止多个后台任务同时读写同一个会话的历史记录导致数据错乱或丢失，需要使用异步锁（asyncio.Lock）来保证操作的排他性/串行化

    def _get_consolidation_lock(self, session_key: str) -> asyncio.Lock:
        """获取或创建会话专属的锁"""
        lock = self._consolidation_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._consolidation_locks[session_key] = lock
        return lock

    # 当记忆整合操作完成后，系统会调用此方法
    def _prune_consolidation_lock(self, session_key: str, lock: asyncio.Lock) -> None:
        """清理闲置的锁（防止内存泄漏）
        Drop lock entry if no longer in use."""

        if not lock.locked():
            self._consolidation_locks.pop(session_key, None)

    async def _process_message(
        self,
        msg: InboundMessage,    #入站消息对象，包含内容、渠道、发送者等信息
        session_key: str | None = None,     #可选的会话键，用于覆盖消息自带的会话标识
        on_progress: Callable[[str], Awaitable[None]] | None = None,    #可选的进度回调，用于流式推送中间结果
    ) -> OutboundMessage | None:
        """消息处理方法，负责将一条入站消息转化为智能体的最终回复。
        Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        # 系统消息处理：有些消息（如定时任务触发、MCP 工具调用等）并非来自用户直接输入，
        # 而是系统内部事件或外部工具触发的。这类消息通常会被标记为 channel="system"，
        # 并且它们的 chat_id 会包含来源信息（格式为 "channel:chat_id"）。
        # 在处理这类消息时，Agent 需要解析出真正的渠道和聊天 ID，以便正确构建上下文并回复到正确的位置。
        # 这些系统就类似于qq、飞书这些
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            # 创建或获取一个临时会话（session）来处理这个系统消息，确保工具调用和记忆整合等功能正常工作
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            # 注入工具上下文
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            # 构建消息列表（含历史）
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            # 运行LLM循环处理消息，得到最终回复内容和所有交互消息
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            # 保存本轮历史记录到会话中，并将回复投递回总线
            self._save_turn(session, all_msgs, 1 + len(history))
            # session落盘存入到session的json文件里
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # 普通用户消息处理：
        # 对于来自用户输入的消息，Agent 首先需要确定它属于哪个会话（session_key），
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        # 内置快捷命令：如果用户输入以 "/" 开头，Agent 会将其视为命令进行特殊处理。
        cmd = msg.content.strip().lower()
        # 新开一个会话：当用户输入 "/new" 命令时，Agent 需要清理当前会话的历史记忆，开始一个全新的对话流程。
        if cmd == "/new":
            # 获取会话专属异步锁
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    # 将 session.last_consolidated 之后的消息快照出来
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        # 创建临时 Session，将快照写入
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        # 调用 _consolidate_memory(archive_all=True)
                        # ├── 失败 → 返回错误提示，不清除会话
                        # └── 成功 → session.clear() + save + invalidate
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        # 直接返回可用命令列表
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")

        # 后台记忆整合触发
        unconsolidated = len(session.messages) - session.last_consolidated
        # 未整合消息数 ≥ memory_window 阈值
        # 且当前没有正在进行的整合任务
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._get_consolidation_lock(session.key)

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    self._prune_consolidation_lock(session.key, lock)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            # Python 的 asyncio.create_task 返回的 Task 如果没有强引用，可能被垃圾回收器提前销毁导致任务中断。
            self._consolidation_tasks.add(_task)
        # 注入工具上下文 + 初始化轮次
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                # 重置"本轮是否已发送消息"标志
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        # 将历史记录、当前消息、媒体附件组装成 LLM 所需的 messages 列表。
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            # 多模态支持
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True        # 标记为进度消息（非最终结果）
            meta["_tool_hint"] = tool_hint      # 标记是否为工具调用提示
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))
        # 进入 _run_agent_loop，执行完整的 思考 → 调用工具 → 观察结果 → 再思考 迭代循环。
        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        # 保存历史 & 决定返回值
        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    _TOOL_RESULT_MAX_CHARS = 500

    def _save_turn(self, 
                   session: Session,    #当前会话对象，用于存储和管理对话历史
                   messages: list[dict],    #本轮 LLM 交互产生的完整消息列表
                   skip: int        #跳过前 N 条已存入历史的消息，只保存本轮新增的部分
                   ) -> None:
        """将新一轮的消息保存到会话中，针对工具调用结果进行截断处理以防过长
        Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            # 针对于每一条消息，构建一个新的字典 entry，包含原消息的所有键值对，但排除掉 "reasoning_content" 键（如果存在的话）。
            entry = {k: v for k, v in m.items() if k != "reasoning_content"}
            # 如果这条消息是一个工具调用结果（entry.get("role") == "tool"）且其内容是字符串类型，那么就检查内容长度是否超过 _TOOL_RESULT_MAX_CHARS 的限制。
            if entry.get("role") == "tool" and isinstance(entry.get("content"), str):
                content = entry["content"]
                if len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            entry.setdefault("timestamp", datetime.now().isoformat())
            # 将处理后的消息条目追加到会话的 messages 列表中，并更新 session 的 updated_at 时间戳为当前时间。
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """将记忆整合的具体逻辑委托给 MemoryStore 类处理
        Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await MemoryStore(self.workspace).consolidate(   #以当前工作目录创建 MemoryStore 实例
            session, self.provider, self.model,
            archive_all=archive_all,        #是否归档全部消息
            memory_window=self.memory_window,   #记忆窗口大小
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """绕过消息总线（MessageBus）的直接调用入口，
        专为 CLI 命令行交互和定时任务（Cron）场景设计，
        不需要经过消息队列的发布/订阅流程。
        Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        # 将纯文本 content 包装成标准消息对象
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        # 走完整的消息处理流程（历史 + LLM + 工具调用）
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
