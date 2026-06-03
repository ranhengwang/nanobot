"""Skills loader for agent capabilities."""

import json
import os
import re
import shutil
from pathlib import Path

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillsLoader:
    """
    Loader for agent skills.
    
    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
    """
    
    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
    
    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        扫描并列出 Agent 当前可用的所有技能（由包含 SKILL.md 的文件夹定义）
        List all available skills.
        
        Args:
            filter_unavailable: If True, filter out skills with unmet requirements.
        
        Returns:
            List of skill info dicts with 'name', 'path', 'source'.
        """
        skills = []
        
        # Workspace skills (highest priority)
        # 首先在工作区目录（self.workspace_skills）下查找。
        # 遍历所有子目录，如果子目录包含 SKILL.md 文件，则将其记录下来，并将 source 标记为 "workspace"。
        if self.workspace_skills.exists():
            for skill_dir in self.workspace_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "workspace"})
        
        # Built-in skills
        # 检查全局内置的技能目录（self.builtin_skills）
        # 同样查找子目录中的 SKILL.md
        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists() and not any(s["name"] == skill_dir.name for s in skills):
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "builtin"})
        
        # Filter by requirements
        # 如果要求过滤掉不可用技能
        if filter_unavailable:
            return [s for s in skills if self._check_requirements(self._get_skill_meta(s["name"]))]
        return skills
    
    def load_skill(self, name: str) -> str | None:
        """
        给一个skill名，去工作区和内置目录里找对应的技能文件（SKILL.md），如果找到就返回它的内容，否则返回 None
        Load a skill by name.
        
        Args:
            name: Skill name (directory name).
        
        Returns:
            Skill content or None if not found.
        """
        # Check workspace first
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            return workspace_skill.read_text(encoding="utf-8")
        
        # Check built-in
        if self.builtin_skills:
            builtin_skill = self.builtin_skills / name / "SKILL.md"
            if builtin_skill.exists():
                return builtin_skill.read_text(encoding="utf-8")
        
        return None
    
    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        加载指定的技能内容（移除了顶部的元数据信息），并将其格式化为一个完整的 Markdown 字符串，以便将其注入到 Agent 的系统上下文
        Load specific skills for inclusion in agent context.
        
        Args:
            skill_names: List of skill names to load.
        
        Returns:
            Formatted skills content.
        """
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                # 调用 _strip_frontmatter(content) 移除文件顶部的 YAML 元数据（由 --- 包裹的部分）
                content = self._strip_frontmatter(content)
                parts.append(f"### Skill: {name}\n\n{content}")
        
        return "\n\n---\n\n".join(parts) if parts else ""
    
    def build_skills_summary(self) -> str:
        """
        该方法用于构建所有技能的摘要信息（包括技能名称、描述、路径和可用性等），并将其格式化为 XML 结构
        Build a summary of all skills (name, description, path, availability).
        
        This is used for progressive loading - the agent can read the full
        skill content using read_file when needed.
        
        Returns:
            XML-formatted skills summary.
        """
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""
        # XML 转义助手：定义了 escape_xml 内部函数，用于将 &, <, > 等特殊字符转义，防止破坏 XML 结构
        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        lines = ["<skills>"]
        # 构建 XML 结构：
        #     以 <skills> 标签开始。
        #     遍历每个技能，提取并转义 name、path和 description。
        #     检查该技能的依赖项（available 状态）
        for s in all_skills:
            name = escape_xml(s["name"])
            path = s["path"]
            desc = escape_xml(self._get_skill_description(s["name"]))
            skill_meta = self._get_skill_meta(s["name"])
            available = self._check_requirements(skill_meta)
            
            lines.append(f"  <skill available=\"{str(available).lower()}\">")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")
            
            # Show missing requirements for unavailable skills
            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    lines.append(f"    <requires>{escape_xml(missing)}</requires>")
            
            lines.append(f"  </skill>")
        # 拼接 </skills> 闭合标签，将所有行组合成一个完整的多行 XML 字符串
        lines.append("</skills>")
        
        return "\n".join(lines)
    
    def _get_missing_requirements(self, skill_meta: dict) -> str:
        """获取并格式化某个技能缺失的依赖项，返回一个描述性字符串。
        Get a description of missing requirements."""
        missing = []
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                missing.append(f"CLI: {b}")
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)
    
    def _get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter."""
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name  # Fallback to skill name
    
    def _strip_frontmatter(self, content: str) -> str:
        """从 Markdown 内容中移除顶部的 YAML Frontmatter 元数据块
        Remove YAML frontmatter from markdown content."""
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end():].strip()
        return content
    
    def _parse_nanobot_metadata(self, raw: str) -> dict:
        """将提取出的原始字符串解析为 JSON 格式的元数据字典，主要处理技能文件（SKILL.md）Frontmatter 中定义的特定配置
        Parse skill metadata JSON from frontmatter (supports nanobot and openclaw keys)."""
        try:
            # 将传入的字符串解析为py对象
            data = json.loads(raw)
            # 如果解析结果是一个字典，它会优先获取 "nanobot" 键下对应的值。
            # 为了向下/跨项目兼容，如果找不到 "nanobot"，它会尝试回退获取 "openclaw" 键下的值。
            # 如果都没找到，默认返回空字典 {}。
            return data.get("nanobot", data.get("openclaw", {})) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _check_requirements(self, skill_meta: dict) -> bool:
        """检查一个技能的运行依赖要求是否已满足，决定该技能在当前系统环境下是否可用
        Check if skill requirements are met (bins, env vars)."""
        # 从传入的技能元数据 skill_meta 中提取 requires（依赖项）字典，如果未定义则默认为空字典 {}
        requires = skill_meta.get("requires", {})
        # 检查命令行工具
        # 遍历技能要求的所有命令行工具名称，使用 Python 标准库中的 shutil.which() 检查该工具是否已安装并存在于系统的 PATH 环境变量中。
        for b in requires.get("bins", []):
            if not shutil.which(b):
                return False
        # 检查环境变量
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True
    
    def _get_skill_meta(self, name: str) -> dict:
        """Get nanobot metadata for a skill (cached in frontmatter)."""
        meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(meta.get("metadata", ""))
    
    def get_always_skills(self) -> list[str]:
        """获取所有标记为 always=true 且当前系统环境满足依赖要求的技能名称列表
        Get skills marked as always=true that meet requirements."""
        result = []
        for s in self.list_skills(filter_unavailable=True):
            meta = self.get_skill_metadata(s["name"]) or {}
            skill_meta = self._parse_nanobot_metadata(meta.get("metadata", ""))
            if skill_meta.get("always") or meta.get("always"):
                result.append(s["name"])
        return result
    
    def get_skill_metadata(self, name: str) -> dict | None:
        """
        从 Markdown 格式的技能文件（SKILL.md）顶部提取 YAML Frontmatter 元数据，并将其解析为 Python 字典
        主要是要得到skill-name、description、requires（依赖项）等信息，这些信息可以用来判断技能是否可用，以及在构建系统提示时提供技能描述等。

        Get metadata from a skill's frontmatter.
        
        Args:
            name: Skill name.
        
        Returns:
            Metadata dict or None.
        """
        content = self.load_skill(name)
        if not content:
            return None
        # 检查文本是否以 --- 开头
        if content.startswith("---"):
            # 使用正则表达式 ^---\n(.*?)\n--- 提取被两个 --- 包裹的元数据文本块。
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                # Simple YAML parsing
                metadata = {}
                for line in match.group(1).split("\n"):
                    # 寻找包含冒号 : 的行，按第一个冒号将其拆分为 key 和 value。
                    if ":" in line:
                        key, value = line.split(":", 1)
                        # 去除键值对前后的空白字符和多余的引号 " 或 '
                        metadata[key.strip()] = value.strip().strip('"\'')
                return metadata
        
        return None
