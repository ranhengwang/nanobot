"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
    
    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """负责将多个信息源（agent的身份、BOOTSTRAP_FILES里储的md文件、长期记忆、skills）拼装成一个完整的 System Prompt，最终发送给 LLM
        注意，这里没有注入工具的description，因为工具调用的细节（如参数）会在后续的 messages 中通过 tool_calls 传递给 LLM，而不是直接放在 system prompt 中。
        Build the system prompt from bootstrap files, memory, and skills.
        
        Args:
            skill_names: Optional list of skills to include.
        
        Returns:
            Complete system prompt.
        """
        # 执行流程
        #     parts = []
        # │
        # ├─ 1. _get_identity()          → Agent 身份 + 运行环境信息
        # ├─ 2. _load_bootstrap_files()  → 读取工作区引导文件
        # ├─ 3. memory.get_memory_context() → 注入长期记忆
        # ├─ 4. skills.get_always_skills()  → 始终加载的技能（完整内容）
        # └─ 5. skills.build_skills_summary() → 可用技能摘要（按需加载）
        # return "\n\n---\n\n".join(parts)

        parts = []
        
        # Core identity
        parts.append(self._get_identity())
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        
        # Memory context
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """生成 Agent 核心身份 System Prompt 的方法
        它动态收集运行时环境信息，然后拼接成一段结构化的 Markdown 文本，作为 System Prompt 的第一部分注入给 LLM
        Get the core identity section."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        workspace_path = str(self.workspace.expanduser().resolve())
        # 让 LLM 知道宿主机环境（macOS/Linux/Windows + 架构）
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
# nanobot 🐈               ← 身份定义
## Current Time            ← 当前时间 + 时区
## Runtime                 ← 操作系统 + Python版本
## Workspace               ← 工作区路径及关键文件位置
## Tool Call Guidelines    ← 工具调用行为规范（如先读后写）
## Memory                  ← 长期记忆的读写位置

# HISTORY.md 是 nanobot 的历史日志文件

        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant. 

## Current Time
{now} ({tz})

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel.

## Tool Call Guidelines
- Before calling tools, you may briefly state your intent (e.g. "Let me check that"), but NEVER predict or describe the expected result before receiving it.
- Before modifying a file, read it first to confirm its current content.
- Do not assume a file or directory exists — use list_dir or read_file to verify.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.

## Memory
- Remember important facts: write to {workspace_path}/memory/MEMORY.md
- Recall past events: grep {workspace_path}/memory/HISTORY.md"""
    
    def _load_bootstrap_files(self) -> str:
        """read读取BOOTSTRAP_FILES下的几个.md文档
        Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        构建发送给 LLM（大语言模型）的完整对话消息列表。它将 System Prompt、历史对话以及用户当前的新消息，按照模型要求的格式按顺序组装起来
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        system_prompt = self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        # 当前用户消息（如果有媒体附件，则转换为内嵌图片格式）
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """构建用户消息内容的辅助方法，支持纯文本和图文混合两种格式。
        如果存在媒体文件路径，会将其转换为 base64 编码的内嵌图片格式，最终返回一个包含文本和图片块的列表；如果没有媒体，则直接返回文本字符串。
        Build user message content with optional base64-encoded images."""
        if not media:
            return text
        
        images = []
        for path in media:
            p = Path(path)
            # 根据文件扩展名自动推断 MIME 类型
            mime, _ = mimetypes.guess_type(path)
            # 过滤非图片文件
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            # 将图片二进制转为 base64 字符串，内嵌在 URL 中
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],     #当前的上下文消息列表
        tool_call_id: str,      #工具调用的唯一 ID（LLM 在调用工具时生成，工具结果会通过这个 ID 关联回对应的调用）
        tool_name: str,     #被调用的工具名称
        result: str     #工具执行后输出的结果
    ) -> list[dict[str, Any]]:
        """
        将工具执行的结果追加到对话消息列表中。
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],     #当前的对话上下文消息列表
        content: str | None,        #助手回复的普通文本内容
        tool_calls: list[dict[str, Any]] | None = None,     #助手发起的工具调用请求列表
        reasoning_content: str | None = None,       #推理思考过程的内容
    ) -> list[dict[str, Any]]:
        """
        将 AI 助手（assistant）的回复 追加到对话上下文的消息列表中
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant"}

        # Always include content — some providers (e.g. StepFun) reject
        # assistant messages that omit the key entirely.
        msg["content"] = content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Include reasoning content when provided (required by some thinking models)
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content

        messages.append(msg)
        return messages
