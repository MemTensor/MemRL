"""
mem0 集成核心模块（mem0_core）。

该包与 `memrl.service` / MemOS 解耦，仅提供：
- 基于 `mem0` Python SDK 的轻量客户端封装（本地 / 自托管）。
- 跨 benchmark 统一的 Experience 抽象（一次任务 + 轨迹）。
- 以 Experience 为中心的 Mem0Store（写入 / 检索 + 钩子式结构化日志）。
- 将检索结果格式化为可直接注入 LLM 上下文的辅助函数。

BigCodeBench、LifelongBench 等基准只需依赖本包，不需要了解 MemOS 内部细节。
"""

from .config import Mem0Config
from .types import Experience, RetrievedMemory
from .client import Mem0Client
from .store import Mem0Store
from .formatting import format_memories_for_llm

__all__ = [
    "Mem0Config",
    "Experience",
    "RetrievedMemory",
    "Mem0Client",
    "Mem0Store",
    "format_memories_for_llm",
]
