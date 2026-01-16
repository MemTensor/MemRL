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
        retrieve_threshold: float = 0.2,
        rl_enabled: bool = False,
        memory_budget_tokens: int = 2000,
    ) -> None:
        self.name = name
        self._llm = llm_provider
        self._mem = mem_service
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self._sys_prompt = str(system_prompt or "")
        self._k = int(retrieve_k)
        self._threshold = float(retrieve_threshold)
        self._rl = bool(rl_enabled)
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

    def _retrieve_memory(self, query: str) -> Tuple[str, List[str], Dict[str, Any]]:
        if not self._mem:
            return "", [], {}

        if self._rl:
            vd = self._mem.retrieve_value_aware(query, k=self._k, threshold=self._threshold)
            candidates = list(vd.get("candidates") or [])
            simmax = float(vd.get("simmax", 0.0) or 0.0)

            # Select top-k candidates by similarity (value-aware ordering can be added later).
            try:
                candidates.sort(key=lambda x: float(x.get("similarity", 0.0) or 0.0), reverse=True)
            except Exception:
                pass
            selected = [c for c in candidates if float(c.get("similarity", 0.0) or 0.0) >= self._threshold][: self._k]
            selected_ids = [str(c.get("memory_id") or c.get("id")) for c in selected if (c.get("memory_id") or c.get("id"))]
            return (
                self._format_memory_context(selected),
                selected_ids,
                {"mode": "rl", "k": self._k, "threshold": self._threshold, "simmax": simmax, "retrieved_count": len(candidates)},
            )

        # RL-off: similarity-only retrieval
        candidates = self._mem.retrieve(query, k=self._k, threshold=self._threshold)
        selected_ids = [str(c.get("memory_id") or c.get("id")) for c in candidates if (c.get("memory_id") or c.get("id"))]
        simmax = 0.0
        try:
            simmax = max((float(c.get("similarity", 0.0) or 0.0) for c in candidates), default=0.0)
        except Exception:
            pass
        return (
            self._format_memory_context(candidates),
            selected_ids,
            {"mode": "similarity", "k": self._k, "threshold": self._threshold, "simmax": simmax, "retrieved_count": len(candidates)},
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
            mem_context, selected_ids, trace = self._retrieve_memory(prompt)
            code = self._generate_single(prompt, mem_context)
            info = {
                "prompt": prompt,
                "selected_ids": selected_ids,
                "trace": trace,
                "num_retrieved": len(selected_ids),
                "memory_context": mem_context,
            }
            self._last_retrieval = info
            self._last_retrievals.append(info)
            out.append([code])

        return out

