"""
selfrag_core
============

基于主 LLM 的 Self-RAG 风格记忆基线核心模块。

模块职责：
- 提供统一的 SelfRAGExperience 抽象，用于描述「一次任务 + 执行轨迹」；
- 使用简单向量检索（SelfRAGStore）从历史 Experience 构造候选记忆集合；
- 使用主 LLM（SelfRAGClient）基于 Prompt 决定是否检索、选哪些候选、如何打标签；
- 将最终选中的 Experience 格式化为可注入 system prompt 的 memory_context 文本。

注意：
- 这里的 Self-RAG 是“算法思想”的实现，而不是论文原始模型的复刻；
- 不依赖 llama_index / SelfRAGPack，仅依赖本项目现有的 LLM / Embedding 提供方。
"""

from .config import SelfRAGConfig
from .store import SelfRAGStore
from .client import SelfRAGClient
from .formatting import build_memory_context
from .types import (
    SelfRAGExperience,
    RetrievedCandidate,
    SelfRAGDecision,
    SelfRAGDocDecision,
)

__all__ = [
    "SelfRAGConfig",
    "SelfRAGStore",
    "SelfRAGClient",
    "build_memory_context",
    "SelfRAGExperience",
    "RetrievedCandidate",
    "SelfRAGDecision",
    "SelfRAGDocDecision",
]

