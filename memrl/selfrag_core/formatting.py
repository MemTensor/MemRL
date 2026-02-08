from __future__ import annotations

from typing import List, Tuple

from .types import RetrievedCandidate, SelfRAGDecision


def build_memory_context(
    candidates: List[RetrievedCandidate],
    decision: SelfRAGDecision,
    *,
    header: str = "# Retrieved Experiences from Self-RAG\n",
) -> Tuple[str, int]:
    """
    根据 self-rag 决策结果，构造最终要注入给主 LLM 的 memory_context 文本。

    返回：(memory_context, num_selected)
    """
    if not candidates or not decision.should_retrieve:
        return "(no memory context)", 0

    selected_ids = {d.id for d in decision.docs if d.selected}
    if not selected_ids:
        return "(no memory context)", 0

    # 按原始候选顺序输出，便于日志与调试
    sections: List[str] = []
    sections.append(header.rstrip())

    idx = 1
    for cand in candidates:
        if cand.id not in selected_ids:
            continue
        meta = cand.metadata or {}
        task_id = meta.get("task_id")
        success = meta.get("success")
        outcome = "SUCCESS" if success is True else "FAILURE" if success is False else "UNKNOWN"
        sections.append(f"\n## Example {idx} [{outcome}] (task_id={task_id})")
        sections.append(cand.text.strip())
        idx += 1

    if idx == 1:
        return "(no memory context)", 0

    return "\n".join(sections), idx - 1

