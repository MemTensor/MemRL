"""
BigCodeBench (BCB) multi-epoch runner for MemRL.

This runner implements the same high-level structure used by other benchmarks:
  - multi-epoch loop
  - per-epoch train then val
  - retrieval via MemoryService.retrieve_query (dict_memory + RL threshold)
  - train writes memories via MemoryService.add_memories (keeps dict_memory in sync)
  - value-driven Q updates via MemoryService.update_values
  - per-epoch snapshots via MemoryService.save_checkpoint_snapshot(target_ck_dir, ckpt_id)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memp.bigcodebench_eval.bcb_adapter import extract_code_from_response
from memp.bigcodebench_eval.eval_utils import (
    ensure_bigcodebench_on_path,
    run_untrusted_check_with_hard_timeout,
    sanitize_code,
)
from memp.bigcodebench_eval.task_wrappers import get_prompt, load_bcb_data, split_dataset, write_samples

logger = logging.getLogger(__name__)


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
        self.bcb_repo = bcb_repo
        self.untrusted_hard_timeout_s = float(untrusted_hard_timeout_s)
        self.eval_timeout_s = float(eval_timeout_s)

        ensure_bigcodebench_on_path(self.bcb_repo)

        self._problems: Dict[str, Dict[str, Any]] = {}
        self._train_ids: List[str] = []
        self._val_ids: List[str] = []

    def _get_retrieve_threshold(self) -> float:
        """Align with other benchmarks: use rl_config.sim_threshold (fallback to tau)."""
        try:
            rl_cfg = getattr(self.mem, "rl_config", None)
            if rl_cfg is None:
                return 0.0
            return float(getattr(rl_cfg, "sim_threshold", getattr(rl_cfg, "tau", 0.0)))
        except Exception:
            return 0.0

    def _format_memory_context(
        self, selected_mems: List[Dict[str, Any]], *, budget_tokens: int = 2000
    ) -> str:
        if not selected_mems:
            return ""
        budget_chars = max(0, int(budget_tokens) * 4)
        lines: List[str] = ["[Retrieved Memory Context]"]
        used = 0
        for idx, m in enumerate(selected_mems, start=1):
            mem_id = m.get("memory_id") or m.get("id") or "unknown"
            content = m.get("content") or ""
            sim = m.get("similarity", None)
            header = f"\n### Memory {idx} (id={mem_id}"
            if sim is not None:
                try:
                    header += f", sim={float(sim):.3f}"
                except Exception:
                    pass
            header += ")\n"
            block = header + str(content).strip() + "\n"
            if used + len(block) > budget_chars:
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines).strip() + "\n"

    def _generate_code(self, prompt: str, *, memory_context: str = "") -> str:
        messages: List[Dict[str, str]] = []
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
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
        return extract_code_from_response(resp or "")

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

        # Buffered memory updates (mini-batch-like), consistent with other runners.
        pending_task_descriptions: List[str] = []
        pending_trajectories: List[str] = []
        pending_successes: List[bool] = []
        pending_retrieved_ids: List[List[str]] = []
        pending_retrieved_queries: List[Optional[List[Tuple[str, float]]]] = []
        pending_metadatas: List[Dict[str, Any]] = []

        def _flush_memory_updates() -> None:
            if not update_memory or not pending_task_descriptions:
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
                        str(m.get("memory_id"))
                        for m in selected_mems
                        if isinstance(m, dict) and m.get("memory_id")
                    ]
                    mem_context = self._format_memory_context(selected_mems)
                except Exception:
                    logger.debug("BCB retrieval failed for %s", task_id, exc_info=True)

            code = self._generate_code(prompt, memory_context=mem_context)

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
                "epoch": epoch,
                "phase": phase,
                "model": self.model_name,
                "status": eval_res.get("status"),
                "error": eval_res.get("error"),
            }
            samples.append(sample)

            if update_memory:
                traj = "\n".join(
                    [
                        f"[BCB] epoch={epoch} phase={phase} task_id={task_id}",
                        "[PROMPT]",
                        prompt,
                        "[GENERATED CODE]",
                        "```python",
                        code,
                        "```",
                        "[EVAL]",
                        json.dumps({"status": eval_res.get("status"), "error": eval_res.get("error")}, ensure_ascii=False),
                    ]
                )
                # Align with other benchmarks: use add_memories() so dict_memory stays consistent.
                try:
                    rl_cfg = getattr(self.mem, "rl_config", None)
                    q_init_pos = float(getattr(rl_cfg, "q_init_pos", 0.0)) if rl_cfg else 0.0
                    q_init_neg = float(getattr(rl_cfg, "q_init_neg", 0.0)) if rl_cfg else 0.0
                except Exception:
                    q_init_pos, q_init_neg = 0.0, 0.0

                meta = {
                    "source_benchmark": "bigcodebench",
                    "success": bool(ok),
                    "q_value": (q_init_pos if ok else q_init_neg),
                    "q_visits": 0,
                    "q_updated_at": datetime.now().isoformat(),
                    "last_used_at": datetime.now().isoformat(),
                    "reward_ma": 0.0,
                    "task_id": task_id,
                    "bcb_epoch": epoch,
                    "phase": phase,
                    "model": self.model_name,
                }

                pending_task_descriptions.append(prompt)
                pending_trajectories.append(traj)
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
