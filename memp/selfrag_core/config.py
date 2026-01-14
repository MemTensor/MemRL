from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SelfRAGConfig:
    """
    Self-RAG 基线的最小配置。

    说明：
    - 与 mem0 一样，只保留当前实验真正需要的字段；
    - Self-RAG 在本项目中指“由主 LLM 驱动的检索决策层”，而不是论文中的特定模型实现。
    """

    # 记忆作用域标识：用于隔离不同实验（benchmark/split/subset）
    user_id: str = "selfrag_default_user"

    # 检索相关配置
    top_k: int = 5
    # 预留：是否允许在决策失败时退回纯相似度 Top-K
    fallback_on_error: bool = True

    # 索引持久化根目录（相对项目根目录）
    index_root: str = ".selfrag"

    # self-rag 决策相关配置
    retrieval_mode: str = "adaptive"  # "adaptive" | "always" | "never"

    # 主 LLM 调用的最大重试次数（例如 JSON 解析失败时）
    max_decision_retries: int = 2

    # 透传给底层 embedding provider 的配置（如 model/base_url 等）
    embedding_cfg: Dict[str, str] = field(default_factory=dict)

