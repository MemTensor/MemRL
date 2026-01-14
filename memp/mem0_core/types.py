from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Experience:
    """
    统一描述一次“可写入 mem0 的经验”的数据结构。

    在 mem0 中，每条经验被表示为一段最小对话：
    - user 消息：任务描述 / 指令（task_text）
    - assistant 消息：轨迹 / 输出（trajectory）

    各个 benchmark 的适配层负责构造该结构，并在 metadata 中写入：
    - benchmark 名称
    - task_id / split / epoch 等信息
    - 成功与否、奖励等标签
    """

    benchmark: str
    task_id: str
    phase: str  # e.g. "train" | "val"
    success: bool
    task_text: str
    trajectory: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedMemory:
    """
    mem0 返回的单条记忆项在本仓库中的统一视图。

    该结构屏蔽了 mem0 v0.x 与 v1.x 在返回格式上的差异，只保留评测中真正需要的字段。
    """

    id: str
    memory: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional extra fields copied from mem0 responses when present
    # 可选信息：从 mem0 原始结果中直接拷贝，便于后续筛选 / 诊断
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

