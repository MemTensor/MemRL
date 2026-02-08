"""
BigCodeBench (BCB) multi-epoch runner for MemRL.

This runner implements the same high-level structure used by other benchmarks:
  - multi-epoch loop
  - per-epoch train then val
  - retrieval via MemoryService.retrieve_query (dict_memory + RL threshold)
  - train writes memories via MemoryService.add_memories (keeps dict_memory in sync)
  - value-driven Q updates via MemoryService.update_values (best-effort)
  - per-epoch snapshots via MemoryService.save_checkpoint_snapshot(target_ck_dir, ckpt_id)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memrl.bigcodebench_eval.bcb_adapter import extract_code_from_response
from memrl.bigcodebench_eval.eval_utils import (
    ensure_bigcodebench_on_path,
    run_untrusted_check_with_hard_timeout,
    sanitize_code,
)
from memrl.bigcodebench_eval.task_wrappers import get_prompt, load_bcb_data, split_dataset, write_samples

logger = logging.getLogger(__name__)

# Default system prompt for memory-augmented code generation (aligned to memory_rl).
DEFAULT_SYSTEM_PROMPT = """You are an expert Python programmer solving BigCodeBench coding tasks.

You may receive a [Retrieved Memory Context] block with past experiences from similar problems.
These are **references for learning**, not guaranteed solutions:
- [MEMORY TYPE] SUCCESS_PROCEDURE: A successful approach from a similar task—learn the implementation pattern.
- [MEMORY TYPE] FAILURE_REFLECTION: A failed attempt with lessons—avoid similar mistakes.

Use the memories as inspiration, but always analyze your current task independently and
adapt your approach based on its specific requirements. Generate clean, correct Python code.

Hard constraints for BigCodeBench:
- Do NOT change the required function signature, return type, or required exception types/messages.
- Do NOT wrap specific exceptions into generic ones; keep the exact exception class and message if specified.
- Import every module you use; remove unused imports; do not rely on implicit imports.
- Avoid broad try/except (e.g., `except Exception`) unless the task explicitly requires it.
- Avoid any network calls or extra file I/O beyond what the task specifies.
- Keep code deterministic: no randomness, time-based logic, or unnecessary logging.
"""


@dataclass
class BCBSelection:
    subset: str = "hard"  # hard|full
    split: str = "instruct"  # instruct|complete
    train_ratio: float = 0.7
    seed: int = 42
    split_file: Optional[str] = None
    data_path: Optional[str] = None


class BCBRunner:
    def __init__(
        self,
        *,
        root: Path,
        selection: BCBSelection,
        llm: Any,
        memory_service: Any,
        output_dir: str,
        model_name: str,
        num_epochs: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 1280,
        retrieve_k: int = 5,
        # BigCodeBench uses a dedicated similarity threshold knob (separate from RL tau).
        # If None, falls back to rl_config.sim_threshold (or rl_config.tau).
        retrieve_threshold: Optional[float] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory_budget_tokens: int = 2000,
        bcb_repo: Optional[str] = None,
        untrusted_hard_timeout_s: float = 120.0,
        eval_timeout_s: float = 60.0,
    ) -> None:
        self.root = Path(root)
        self.sel = selection
        self.llm = llm
        self.mem = memory_service
        self.output_dir = os.path.abspath(output_dir)
        self.model_name = str(model_name)
        self.num_epochs = int(num_epochs)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.retrieve_k = int(retrieve_k)
        self.retrieve_threshold = (
            None if retrieve_threshold is None else float(retrieve_threshold)
        )
        self.system_prompt = str(system_prompt or "")
        # NOTE: In memory_rl this is called "budget_tokens" but is used as a rough character budget.
        self.memory_budget_tokens = int(memory_budget_tokens)
        self.bcb_repo = bcb_repo
        self.untrusted_hard_timeout_s = float(untrusted_hard_timeout_s)
        self.eval_timeout_s = float(eval_timeout_s)

        ensure_bigcodebench_on_path(self.bcb_repo)

        self._problems: Dict[str, Dict[str, Any]] = {}
        self._train_ids: List[str] = []
        self._val_ids: List[str] = []

    def _get_retrieve_threshold(self) -> float:
        """BCB threshold knob (aligned to memory_rl)."""
        if self.retrieve_threshold is not None:
            return float(self.retrieve_threshold)
        try:
            rl_cfg = getattr(self.mem, "rl_config", None)
            if rl_cfg is None:
                return 0.0
            return float(getattr(rl_cfg, "sim_threshold", getattr(rl_cfg, "tau", 0.0)))
        except Exception:
            return 0.0

    def _format_memory_context(
        self, selected_mems: List[Dict[str, Any]]
    ) -> str:
        # Align with memory_rl BCB adapter formatting.
        if not selected_mems:
            return ""

        parts: List[str] = ["# Relevant Code Examples from Memory\n"]

        for i, c in enumerate(selected_mems, 1):
            meta_obj = c.get("metadata")
            meta: Dict[str, Any] = {}
            if meta_obj is not None:
                try:
                    if hasattr(meta_obj, "model_dump"):
                        meta = meta_obj.model_dump()  # type: ignore[assignment]
                    elif isinstance(meta_obj, dict):
                        meta = meta_obj
                except Exception:
                    meta = {}

            outcome = meta.get("outcome", "unknown")
            task_id = meta.get("task_id", "")

            mem_item = c.get("memory_item")
            task_desc = ""
            try:
                task_desc = str(getattr(mem_item, "memory", "") or "")
            except Exception:
                task_desc = ""

            raw_content = c.get("content", c.get("full_content", "")) or ""
            content = self._coerce_bcb_memory_content(
                raw_content=raw_content,
                outcome=outcome,
                task_description=task_desc,
            )
            if not content:
                continue

            # Truncate if needed (memory_rl uses a rough per-entry budget).
            if len(content) > self.memory_budget_tokens // len(selected_mems):
                content = content[: self.memory_budget_tokens // len(selected_mems)] + "..."

            parts.append(f"## Example {i} [{outcome.upper()}]")
            if task_id:
                parts.append(f"Task: {task_id}")
            parts.append(content)
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _coerce_bcb_memory_content(
        *,
        raw_content: str,
        outcome: str,
        task_description: str,
    ) -> str:
        """
        BCB-only prompt alignment:
        - memory_rl stores full_content using [MEMORY TYPE]/[TASK]/... blocks.
        - Other benchmarks must not be affected, so we coerce at *injection time*
          for BCB (even if the stored full_content is in legacy "Task: ..." style).
        """
        text = str(raw_content or "").strip()
        if not text:
            return ""

        # If it's already in the memory_rl format, keep as-is.
        if "[MEMORY TYPE]" in text.upper():
            return text

        out = str(outcome or "unknown").strip().lower()
        is_failure = out in {"failure", "fail", "failed", "0", "false", "no"}

        if is_failure:
            # Legacy adjustment content often looks like:
            # "TASK REFLECTION:\nTask: ...\n\nReflection: ...".
            # Prefer a line that begins with "Reflection:"; avoid matching "TASK REFLECTION:" header.
            m = re.search(r"(?is)(?:^|\n)reflection\s*:\s*(.*)$", text)
            reflection = (m.group(1) if m else text).strip()
            td = task_description.strip() if task_description else ""
            if not td:
                # Fall back to whatever we have.
                td = ""
            return (
                "[MEMORY TYPE] FAILURE_REFLECTION\n"
                "[TASK]\n"
                f"{td}\n\n"
                "[REFLECTION]\n"
                f"{reflection}"
            ).strip()

        # Success path: treat legacy body as execution trajectory.
        body = text
        m = re.match(r"(?is)^task\\s*:\\s*.*?\\n\\n(.*)$", text)
        if m:
            body = (m.group(1) or "").strip()
        td = task_description.strip() if task_description else ""
        return (
            "[MEMORY TYPE] SUCCESS_PROCEDURE\n"
            "[TASK]\n"
            f"{td}\n\n"
            "[EXECUTION TRAJECTORY]\n"
            f"{body}"
        ).strip()

    @staticmethod
    def _trajectory_from_raw_or_fallback(
        *,
        raw_response: str,
        prompt: str,
        code: str,
        eval_res: Dict[str, Any],
        retrieval: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Align with memory_rl BigCodeBench:
        - Prefer trajectory = model raw output when available
        - Otherwise fall back to a structured, step-style trajectory blob
        """
        if raw_response:
            return raw_response

        steps: List[str] = []
        # Step 1: Original task prompt
        steps.append("[STEP 1] TASK PROMPT")
        steps.append(prompt or "")

        # Step 2: Memory retrieval info (if any)
        if retrieval:
            trace = retrieval.get("trace", {}) or {}
            steps.append("")
            steps.append("[STEP 2] MEMORY RETRIEVAL")
            steps.append(f"mode: {trace.get('mode', 'similarity')}")
            steps.append(
                f"retrieved_count: {trace.get('retrieved_count', retrieval.get('num_retrieved', 0))}"
            )
            steps.append(f"simmax: {trace.get('simmax', 0.0)}")
            steps.append(f"selected_memory_ids: {retrieval.get('selected_ids', [])}")

        # Step 3: Generated code
        steps.append("")
        steps.append("[STEP 3] GENERATED CODE")
        steps.append("```python")
        steps.append(code or "")
        steps.append("```")

        # Step 4: Evaluation result
        steps.append("")
        steps.append("[STEP 4] EVALUATION RESULT")
        status = eval_res.get("status", "UNKNOWN")
        steps.append(f"status: {status}")
        error_msg = eval_res.get("error", "")
        if error_msg:
            steps.append("error:")
            steps.append(str(error_msg))

        return "\n".join(steps)

    def _generate_raw(self, prompt: str, *, memory_context: str = "") -> str:
        messages: List[Dict[str, str]] = []

        system_parts: List[str] = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        if memory_context:
            system_parts.append(memory_context)
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        messages.append({"role": "user", "content": prompt})
        try:
            resp = self.llm.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception:
            logger.warning("LLM generation failed for BCB prompt", exc_info=True)
            return ""
        return resp or ""

    def _generate_code(self, prompt: str, *, memory_context: str = "") -> str:
        return extract_code_from_response(self._generate_raw(prompt, memory_context=memory_context))

    # -------------------------- I/O helpers --------------------------

    @staticmethod
    def _save_json(path: str, obj: Any) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

    # -------------------------- evaluation --------------------------

    def _evaluate_one(self, *, task: Dict[str, Any], code: str) -> Dict[str, Any]:
        """Evaluate one solution using official BigCodeBench untrusted_check."""
        task_id = str(task.get("task_id", "unknown"))
        entry_point = str(task.get("entry_point", "task_func"))
        test_code = str(task.get("test", "") or "")

        if not test_code:
            return {"task_id": task_id, "status": "SYNTAX_OK", "error": "no_test_code"}

        # quick syntax check
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {"task_id": task_id, "status": "SYNTAX_ERROR", "error": str(e)}

        # sanitize for evaluation robustness (best-effort)
        clean_code = sanitize_code(code, entry_point, bcb_repo=self.bcb_repo)

        from bigcodebench.eval import PASS, FAIL, TIMEOUT  # type: ignore

        stat, details, err, hard_timed_out = run_untrusted_check_with_hard_timeout(
            code=clean_code,
            test_code=test_code,
            entry_point=entry_point,
            max_as_limit=30 * 1024,
            max_data_limit=30 * 1024,
            max_stack_limit=10,
            min_time_limit=1.0,
            gt_time_limit=float(self.eval_timeout_s),
            hard_timeout_s=float(self.untrusted_hard_timeout_s),
            bcb_repo=self.bcb_repo,
        )

        if hard_timed_out:
            return {"task_id": task_id, "status": "TIMEOUT", "error": err or "hard_timeout"}
        if err:
            return {"task_id": task_id, "status": "RUNTIME_ERROR", "error": err}
        if stat == PASS:
            return {"task_id": task_id, "status": "PASS"}
        if stat == TIMEOUT:
            return {"task_id": task_id, "status": "TIMEOUT", "error": "timeout"}
        if stat == FAIL:
            # Keep details small; they can be very long.
            return {"task_id": task_id, "status": "FAIL", "error": str(details)[:500] if details else "fail"}
        return {"task_id": task_id, "status": "UNKNOWN", "error": str(stat)}

    # -------------------------- phases --------------------------

    def _run_phase(
        self,
        *,
        epoch: int,
        phase: str,
        task_ids: List[str],
        epoch_dir: str,
        update_memory: bool,
    ) -> Dict[str, Any]:
        assert phase in {"train", "val"}
        phase_dir = os.path.join(epoch_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)

        samples: List[Dict[str, Any]] = []
        retrieval_logs: List[Dict[str, Any]] = []

        pass_count = 0
        total = len(task_ids)

        # Buffered memory + Q updates (mini-batch-like), aligned with other runners.
        pending_task_descriptions: List[str] = []
        pending_trajectories: List[str] = []
        pending_successes: List[bool] = []
        pending_retrieved_ids: List[List[str]] = []
        pending_retrieved_queries: List[Optional[List[Tuple[str, float]]]] = []
        pending_metadatas: List[Dict[str, Any]] = []

        def _flush_memory_updates() -> None:
            if not update_memory or not pending_task_descriptions or self.mem is None:
                return
            try:
                self.mem.update_values(
                    [float(s) for s in pending_successes], pending_retrieved_ids
                )
            except Exception:
                logger.debug("BCB Q update failed (batch)", exc_info=True)
            try:
                self.mem.add_memories(
                    task_descriptions=pending_task_descriptions,
                    trajectories=pending_trajectories,
                    successes=pending_successes,
                    retrieved_memory_queries=pending_retrieved_queries,
                    retrieved_memory_ids_list=pending_retrieved_ids,
                    metadatas=pending_metadatas,
                )
            except Exception:
                logger.warning("BCB add_memories failed (batch)", exc_info=True)
            finally:
                pending_task_descriptions.clear()
                pending_trajectories.clear()
                pending_successes.clear()
                pending_retrieved_ids.clear()
                pending_retrieved_queries.clear()
                pending_metadatas.clear()

        for idx, task_id in enumerate(task_ids, start=1):
            task = self._problems[task_id]
            prompt = get_prompt(task, split=self.sel.split)
            selected_ids: List[str] = []
            retrieved_topk_queries: Optional[List[Tuple[str, float]]] = None
            mem_context = ""
            retrieval_trace: Dict[str, Any] = {}

            if self.mem is not None and self.retrieve_k > 0:
                try:
                    thr = self._get_retrieve_threshold()
                    ret = self.mem.retrieve_query(prompt, k=self.retrieve_k, threshold=thr)
                    if isinstance(ret, tuple):
                        ret_result, retrieved_topk_queries = ret
                    else:
                        ret_result, retrieved_topk_queries = ret, None

                    selected_mems = (ret_result or {}).get("selected", []) if ret_result else []
                    if not isinstance(selected_mems, list):
                        selected_mems = []

                    selected_ids = [
                        str(m.get("memory_id") or m.get("id"))
                        for m in selected_mems
                        if isinstance(m, dict) and (m.get("memory_id") or m.get("id"))
                    ]
                    mem_context = self._format_memory_context(selected_mems)
                    try:
                        retrieval_trace = {
                            "mode": "retrieve_query",
                            "retrieved_count": len(selected_ids),
                            "simmax": float((ret_result or {}).get("simmax", 0.0) or 0.0),
                        }
                    except Exception:
                        retrieval_trace = {
                            "mode": "retrieve_query",
                            "retrieved_count": len(selected_ids),
                            "simmax": 0.0,
                        }
                except Exception:
                    logger.debug("BCB retrieval failed for %s", task_id, exc_info=True)

            raw_response = self._generate_raw(prompt, memory_context=mem_context)
            code = extract_code_from_response(raw_response)

            retrieval_logs.append(
                {
                    "task_id": task_id,
                    "epoch": epoch,
                    "phase": phase,
                    "selected_ids": selected_ids,
                    "retrieved_topk_queries": retrieved_topk_queries,
                    "threshold": self._get_retrieve_threshold(),
                }
            )

            eval_res = self._evaluate_one(task=task, code=code)
            ok = eval_res.get("status") == "PASS"
            pass_count += 1 if ok else 0

            sample = {
                "task_id": task_id,
                "solution": code,
                "prompt": prompt,
                "raw_response": raw_response,
                "epoch": epoch,
                "phase": phase,
                "model": self.model_name,
                "status": eval_res.get("status"),
                "error": eval_res.get("error"),
            }
            samples.append(sample)

            if update_memory:
                meta = {
                    "source_benchmark": "bigcodebench",
                    "success": bool(ok),
                    # Fields used by memory_rl prompt formatting
                    "task_id": task_id,
                    "outcome": "success" if ok else "failure",
                    "outcome_success": bool(ok),
                    "entry_point": str(task.get("entry_point", "")) if isinstance(task, dict) else "",
                    "libs": (task.get("libs") if isinstance(task, dict) else None),
                    "source": "conversation",
                    "eval_status": eval_res.get("status"),
                    "eval_error": eval_res.get("error"),
                    "bcb_epoch": epoch,
                    "phase": phase,
                    "model": self.model_name,
                }
                retrieval_for_traj = None
                if selected_ids or retrieval_trace:
                    retrieval_for_traj = {
                        "selected_ids": list(selected_ids),
                        "num_retrieved": len(selected_ids),
                        "trace": retrieval_trace,
                    }
                pending_task_descriptions.append(prompt)
                pending_trajectories.append(
                    self._trajectory_from_raw_or_fallback(
                        raw_response=raw_response,
                        prompt=prompt,
                        code=code,
                        eval_res=eval_res,
                        retrieval=retrieval_for_traj,
                    )
                )
                pending_successes.append(bool(ok))
                pending_retrieved_ids.append(list(selected_ids))
                pending_retrieved_queries.append(retrieved_topk_queries)
                pending_metadatas.append(meta)
                if len(pending_task_descriptions) >= 25:
                    _flush_memory_updates()

            if idx % 25 == 0 or idx == total:
                logger.info("[bcb] epoch %d %s %d/%d pass=%d", epoch, phase, idx, total, pass_count)

        _flush_memory_updates()

        samples_path = os.path.join(phase_dir, "samples.jsonl")
        write_samples(samples, samples_path)
        self._save_json(
            os.path.join(phase_dir, "metrics.json"),
            {
                "epoch": epoch,
                "phase": phase,
                "subset": self.sel.subset,
                "split": self.sel.split,
                "model": self.model_name,
                "total": total,
                "pass": pass_count,
                "pass@1": (pass_count / total) if total else None,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # store retrieval traces (useful for debugging)
        write_samples(retrieval_logs, os.path.join(phase_dir, "memory_retrieval.jsonl"))

        return {
            "total": total,
            "pass": pass_count,
            "pass@1": (pass_count / total) if total else None,
            "samples_path": samples_path,
        }

    # -------------------------- public API --------------------------

    def run(self) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)

        # load problems + split once
        self._problems = load_bcb_data(subset=self.sel.subset, data_path=self.sel.data_path)
        self._train_ids, self._val_ids = split_dataset(
            self._problems,
            train_ratio=self.sel.train_ratio,
            seed=self.sel.seed,
            split_file=self.sel.split_file,
        )

        run_cfg = {
            "subset": self.sel.subset,
            "split": self.sel.split,
            "train_ratio": self.sel.train_ratio,
            "seed": self.sel.seed,
            "num_epochs": self.num_epochs,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "retrieve_k": self.retrieve_k,
            # Aligned threshold knob across benchmarks:
            "retrieve_threshold": self._get_retrieve_threshold(),
            "bcb_repo": self.bcb_repo,
            "created_at": datetime.now().isoformat(),
        }
        self._save_json(os.path.join(self.output_dir, "run_config.json"), run_cfg)

        epoch_summaries: List[Dict[str, Any]] = []
        for epoch in range(1, self.num_epochs + 1):
            epoch_dir = os.path.join(self.output_dir, f"epoch{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            train_res = self._run_phase(
                epoch=epoch,
                phase="train",
                task_ids=self._train_ids,
                epoch_dir=epoch_dir,
                update_memory=True,
            )
            val_res = self._run_phase(
                epoch=epoch,
                phase="val",
                task_ids=self._val_ids,
                epoch_dir=epoch_dir,
                update_memory=False,
            )

            # per-epoch snapshot
            try:
                self.mem.save_checkpoint_snapshot(epoch_dir, ckpt_id=str(epoch))
            except Exception:
                logger.warning("Failed to save checkpoint snapshot for epoch %d", epoch, exc_info=True)

            epoch_summary = {"epoch": epoch, "train": train_res, "val": val_res}
            self._save_json(os.path.join(epoch_dir, "epoch_summary.json"), epoch_summary)
            epoch_summaries.append(epoch_summary)

        # final snapshot (best-effort)
        try:
            self.mem.save_checkpoint_snapshot(self.output_dir, ckpt_id="final")
        except Exception:
            logger.warning("Failed to save final snapshot", exc_info=True)

        final = {
            "output_dir": self.output_dir,
            "epochs": epoch_summaries,
        }
        self._save_json(os.path.join(self.output_dir, "summary.json"), final)
        return final
