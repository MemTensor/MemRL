"""
BigCodeBench adapter: wraps memp LLM + MemoryService for code generation.

This keeps the BigCodeBench-specific prompting logic isolated, while allowing
the runner to inject a MemoryService instance (for retrieval + RL updates).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def extract_code_from_response(text: str) -> str:
    """Best-effort extraction of Python code from an LLM response."""
    if not text:
        return ""

    # Prefer fenced python blocks
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()

    # Fallback: strip leading/trailing non-code noise
    return text.strip()


class MempBCBDecoder:
    """Generate code for BCB prompts with optional memory augmentation."""

    def __init__(
        self,
        name: str,
        llm_provider: Any,
        mem_service: Any | None,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 1280,
        system_prompt: str = "",
        retrieve_k: int = 5,
        memory_budget_tokens: int = 2000,
    ) -> None:
        self.name = name
        self._llm = llm_provider
        self._mem = mem_service
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self._sys_prompt = str(system_prompt or "")
        self._k = int(retrieve_k)
        self._budget = int(memory_budget_tokens)

        # per-prompt traces (used by runner for logging + RL updates)
        self._last_retrieval: Dict[str, Any] = {}
        self._last_retrievals: List[Dict[str, Any]] = []

    @property
    def last_retrieval(self) -> Dict[str, Any]:
        return self._last_retrieval

    @property
    def last_retrievals(self) -> List[Dict[str, Any]]:
        return self._last_retrievals

    def _format_memory_context(self, candidates: List[Dict[str, Any]]) -> str:
        if not candidates:
            return ""

        # Very rough token budgeting by characters (keeps dependency surface small).
        budget_chars = max(0, int(self._budget) * 4)
        lines: List[str] = ["[Retrieved Memory Context]"]
        used = 0
        for idx, c in enumerate(candidates, start=1):
            content = c.get("content") or ""
            mem_id = c.get("memory_id") or c.get("id") or "unknown"
            block = f"\n### Memory {idx} (id={mem_id})\n{content}\n"
            if used + len(block) > budget_chars:
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines).strip() + "\n"

    def _get_retrieve_threshold(self) -> float:
        """Align with other benchmarks: use mem_service.rl_config.sim_threshold (fallback tau)."""
        try:
            rl_cfg = getattr(self._mem, "rl_config", None)
            if rl_cfg is None:
                return 0.0
            return float(getattr(rl_cfg, "sim_threshold", getattr(rl_cfg, "tau", 0.0)))
        except Exception:
            return 0.0

    def _retrieve_memory(self, query: str) -> Tuple[str, List[str], Dict[str, Any], Any]:
        if not self._mem:
            return "", [], {}, None

        # Align with current service ecosystem: retrieve_query drives selection.
        thr = self._get_retrieve_threshold()
        ret = self._mem.retrieve_query(query, k=self._k, threshold=thr)
        if isinstance(ret, tuple):
            ret_result, topk_queries = ret
        else:
            ret_result, topk_queries = ret, None

        selected = (ret_result or {}).get("selected", []) if ret_result else []
        if not isinstance(selected, list):
            selected = []
        selected_ids = [
            str(c.get("memory_id") or c.get("id"))
            for c in selected
            if isinstance(c, dict) and (c.get("memory_id") or c.get("id"))
        ]
        simmax = 0.0
        try:
            simmax = max(
                (float(c.get("similarity", 0.0) or 0.0) for c in selected), default=0.0
            )
        except Exception:
            simmax = 0.0

        return (
            self._format_memory_context(selected),
            selected_ids,
            {
                "mode": "retrieve_query",
                "k": self._k,
                "threshold": thr,
                "simmax": simmax,
                "selected_count": len(selected_ids),
            },
            topk_queries,
        )

    def _generate_single(self, prompt: str, mem_context: str) -> str:
        sys_prompt = self._sys_prompt
        if mem_context:
            sys_prompt = (sys_prompt + "\n\n" + mem_context).strip()

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self._llm.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        except Exception as e:
            logger.error("LLM generation failed: %s", e, exc_info=True)
            return ""

        return extract_code_from_response(resp)

    def codegen(self, prompts: List[str], do_sample: bool = False, num_samples: int = 1) -> List[List[str]]:
        # BigCodeBench expects List[List[str]] (k samples per prompt).
        out: List[List[str]] = []
        self._last_retrievals = []

        for prompt in prompts:
            mem_context, selected_ids, trace, topk_queries = self._retrieve_memory(prompt)
            code = self._generate_single(prompt, mem_context)
            info = {
                "prompt": prompt,
                "selected_ids": selected_ids,
                "trace": trace,
                "retrieved_topk_queries": topk_queries,
                "num_retrieved": len(selected_ids),
                "memory_context": mem_context,
            }
            self._last_retrieval = info
            self._last_retrievals.append(info)
            out.append([code])

        return out
