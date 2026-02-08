from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SelfRAGExperience:
    """
    Self-RAG 使用的统一经验抽象。

    设计尽量与 mem0_core.Experience 对齐：
    - 一条 Experience = 一个 task + 对应的执行轨迹；
    - 轨迹中不包含 memory_context，仅包含该任务本身的行为信息。
    """

    benchmark: str
    task_id: str
    phase: str  # "train" / "val"
    success: bool
    task_text: str
    trajectory: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedCandidate:
    """
    底层向量检索返回的候选经验视图。
    """

    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfRAGDocDecision:
    """
    LLM 决策后，针对单条候选经验的决策结果。
    """

    id: str
    task_id: Optional[str] = None
    selected: bool = False
    relevance: Optional[float] = None
    support: Optional[float] = None
    utility: Optional[int] = None  # 1-5 之间的整数，表示主观有用性
    raw_labels: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfRAGDecision:
    """
    一次 self-rag 决策的整体结果。
    """

    should_retrieve: bool
    docs: List[SelfRAGDocDecision] = field(default_factory=list)
    # LLM 的原始输出文本，便于调试与审计
    raw_output: str = ""
    # 若决策过程中出现错误，则在此记录错误信息（不会抛出到上层）
    error: Optional[str] = None

