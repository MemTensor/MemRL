from __future__ import annotations

from typing import Iterable, List

from .types import RetrievedMemory


def format_memories_for_llm(
    memories: Iterable[RetrievedMemory],
    *,
    budget_tokens: int | None,
    header: str = "# Retrieved Memories from mem0\n",
) -> str:
    """
    将 mem0 检索到的记忆格式化为一段可直接注入 LLM 的上下文文本。

    设计意图：
    - 风格尽量与 BigCodeBench / LifelongBench 现有记忆注入方式保持一致，方便横向对比；
    - 每条记忆生成一个编号小节，并在可能的情况下展示 outcome / task_id；
    - 按 budget_tokens 粗略控制整体长度（budget_tokens≈token 数，内部按 1 token ≈ 4 字符估算）。
    """
    mem_list: List[RetrievedMemory] = list(memories)
    if not mem_list:
        return ""

    parts: List[str] = [header.rstrip(), ""]

    for idx, mem in enumerate(mem_list, 1):
        meta = mem.metadata or {}
        outcome = str(meta.get("outcome", "") or "").lower()
        if not outcome:
            # Fall back to success flag if present.
            if meta.get("success") is True or meta.get("outcome_success") is True:
                outcome = "success"
            elif meta.get("success") is False or meta.get("outcome_success") is False:
                outcome = "failure"
            else:
                outcome = "unknown"
        task_id = str(meta.get("task_id", "") or "")

        # 按用户当前实验需求：不对 memory 文本做截断，原样提供给 LLM。
        # budget_tokens 参数保留用于向后兼容，但在 mem0 基线模式下不再生效。
        content = mem.memory or ""

        title = f"## Example {idx} [{outcome.upper()}]"
        if task_id:
            title += f" (task_id={task_id})"
        parts.append(title)
        parts.append(content)
        parts.append("")

    return "\n".join(parts)
