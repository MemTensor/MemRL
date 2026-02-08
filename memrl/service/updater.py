"""
Updaters for different update strategies in the Memp system.

This module mirrors builders.py and retrievers.py patterns and provides:
- BaseUpdater (abstract)
- VanillaUpdater / ValidationUpdater / AdjustmentUpdater (concrete)
- get_updater factory

Key goals:
- Use MemOS text_mem.add/update to ensure real memory_id is available
- Support Adjustment in append and inplace modes
- Unify metadata fields (task_description, strategies, confidence, updated_at, etc.)
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from memos.mem_os.main import MOS
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata

from .strategies import UpdateStrategy, StrategyConfiguration
from .builders import get_builder

logger = logging.getLogger(__name__)


# ------------ Helper structures ------------

@dataclass
class AdjustmentConfig:
    mode: str = "append"  # "append" | "inplace"
    confidence_factor: float = 0.8  # reduce confidence for adjustment


def _now_iso() -> str:
    return datetime.now().isoformat()


def _build_standard_metadata(
    *,
    base: Optional[Dict[str, Any]],
    task_description: str,
    strategies: StrategyConfiguration,
    confidence: float,
    extra: Optional[Dict[str, Any]] = None,
) -> TextualMemoryMetadata:
    """Compose TextualMemoryMetadata with unified fields.

    Note: TextualMemoryMetadata(model_config.extra="allow") permits extra fields.
    """
    meta: Dict[str, Any] = dict(base or {})
    meta.setdefault("updated_at", _now_iso())
    meta.setdefault("memory_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    meta["task_description"] = task_description
    meta["strategy_build"] = strategies.build.value
    meta["strategy_retrieve"] = strategies.retrieve.value
    meta["strategy_update"] = strategies.update.value
    meta["confidence"] = confidence
    if extra:
        meta.update(extra)
    return TextualMemoryMetadata(**meta)


def _get_text_mem(mos: MOS, user_id: str, mem_cube_id: Optional[str]) -> Any:
    if mem_cube_id is None:
        # fallback to user's first accessible cube
        cubes = mos.user_manager.get_user_cubes(user_id)
        if not cubes:
            raise ValueError(f"No mem cube accessible for user {user_id}")
        mem_cube_id = cubes[0].cube_id
    if mem_cube_id not in mos.mem_cubes:
        raise ValueError(f"MemCube '{mem_cube_id}' is not loaded. Please register.")
    text_mem = mos.mem_cubes[mem_cube_id].text_mem
    if text_mem is None:
        raise ValueError("Textual memory is not initialized")
    return text_mem


# ------------ Base class ------------

def mem_add_with_retry(text_mem, item, max_retries=5, base_delay=2.0):
    """
        Retry for embedding queries
    """
    for attempt in range(1, max_retries + 1):
        try:
            text_mem.add([item])
            return None
        except Exception as e:
            logger.warning(
                f"[Retry {attempt}/{max_retries}] Textmem add 失败: {e}"
            )
            if attempt == max_retries:
                logger.error("达到最大重试次数，仍未成功。")
                raise

            sleep_time = base_delay * (2 ** (attempt - 1))
            time.sleep(sleep_time)

class BaseUpdater(ABC):
    def __init__(
        self,
        mos: MOS,
        num_workers: int,
        user_id: str,
        strategies: StrategyConfiguration,
        llm: Any,
        *,
        default_cube_id: Optional[str] = None,
        memory_confidence: float = 100.0,
        adjustment_config: Optional[AdjustmentConfig] = None,
    ) -> None:
        self.mos = mos
        self.num_workers = num_workers
        self.user_id = user_id
        self.strategies = strategies
        self.llm = llm
        self.default_cube_id = default_cube_id
        self.memory_confidence = memory_confidence
        self.adjustment_config = adjustment_config or AdjustmentConfig()

    @abstractmethod
    def prepare_update_op(
        self,
        task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        [NEW] The "thinking" phase. Prepares the memory operation without writing to the DB.
        This method is thread-safe and can be run in parallel.
        It must return a dictionary representing the write operation.
        The dictionary must include an "op" key ('add', 'update', 'noop') and a "task_description" key.
        """
        ...

    def execute_update_op(self, op: Dict) -> Optional[str]:
        """
        [NEW] The "writing" phase. Executes a prepared operation dictionary.
        This method is fast and should be run serially to prevent race conditions.
        """
        if not op or op.get("op") == "noop":
            return None

        text_mem = _get_text_mem(self.mos, self.user_id, self.default_cube_id)
        
        op_type = op["op"]
        if op_type == "add":
            item = op["item"]
            mem_add_with_retry(text_mem, item)
            return str(item.id)
        elif op_type == "update":
            mem_id = op["id"]
            data = op["data"]
            text_mem.update(mem_id, data)
            return str(mem_id)
        
        logger.warning(f"Unknown operation type '{op_type}' in execute_update_op.")
        return None

    def update(
        self,
        task_description: str,
        trajectory: str,
        success: bool,
        retrieved_memory_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """A single, synchronous update operation for convenience and testing."""
        op = self.prepare_update_op(task_description, trajectory, success, retrieved_memory_ids, metadata)
        return self.execute_update_op(op)

    def update_batch(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        successes: List[bool],
        retrieved_ids_list: List[List[str]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Optional[str]]:
        """
        Hybrid parallel/serial batch update using lists of data.
        """
        num_tasks = len(task_descriptions)
        logger.info(f"Starting hybrid parallel update for {num_tasks} memories...")
        
        metadatas = metadatas or [None] * num_tasks
        
        ops_to_execute = [None] * num_tasks

        # --- Phase 1: Parallel "Thinking" (prepare_update_op) ---
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_index = {
                executor.submit(self.prepare_update_op, td, traj, success, r_ids, meta): i
                for i, (td, traj, success, r_ids, meta) in enumerate(zip(
                    task_descriptions, trajectories, successes, retrieved_ids_list, metadatas
                ))
            }

            for future in tqdm(as_completed(future_to_index), total=num_tasks, desc="Updating memories (Parallel Processing)"):
                index = future_to_index[future]
                try:
                    op = future.result()
                    ops_to_execute[index] = op
                except Exception as e:
                    logger.error(f"Failed to prepare update for task '{task_descriptions[index]}': {e}", exc_info=True)

        # --- Phase 2: Serial "Writing" (execute_update_op) ---
        logger.info(f"Executing {len(ops_to_execute)} prepared memory operations serially...")
        results = []
        for op in tqdm(ops_to_execute, desc="Updating memories (Serial Writing)"):
            if op:
                task_desc = op.get("task_description", "unknown_task")
                try:
                    mem_id = self.execute_update_op(op)
                    results.append((task_desc, mem_id))
                except Exception as e:
                    logger.error(f"Failed to execute update for task '{task_desc}': {e}", exc_info=True)
                    results.append((task_desc, None))

        return results


    # utilities usable by subclasses
    def _add_new_memory(self, task_description: str, full_content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Add memory where embedding uses only task_description, and full content is in metadata."""
        text_mem = _get_text_mem(self.mos, self.user_id, self.default_cube_id)
        item = TextualMemoryItem(
            memory=task_description,  # retrieval key only
            metadata=_build_standard_metadata(
                base=metadata,
                task_description=task_description,
                strategies=self.strategies,
                confidence=self.memory_confidence,
                extra={"type": "procedure", "source": "conversation", "full_content": full_content},
            ),
        )
        text_mem.add([item])
        return str(item.id)

    def _generate_reflection(self, task_description: str, failed_trajectory: str) -> str:
        prompt = f"""
Task: {task_description}

Failed trajectory:
{failed_trajectory}

This task failed. Analyze what went wrong and suggest improvements for future similar tasks.
Focus on:
1. Incorrect assumptions
2. Steps to improve
3. What to avoid next time

Provide a brief reflection:
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.llm.generate(messages, temperature=0.3)
        except Exception as e:
            logger.warning(f"LLM reflection generation failed: {e}")
            return "Reflection not available due to LLM error."


# ------------ Concrete updaters ------------

class VanillaUpdater(BaseUpdater):
    def prepare_update_op(
        self, task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Prepare op to always add a new memory."""
        builder = get_builder(self.strategies.build, self.llm)
        memory_body = builder.build(task_description, trajectory)
        memory_content = f"Task: {task_description}\n\n{memory_body}"
        
        item = TextualMemoryItem(
            memory=task_description,
            metadata=_build_standard_metadata(
                base=metadata, task_description=task_description, strategies=self.strategies,
                confidence=self.memory_confidence,
                extra={"type": "procedure", "source": "conversation", "full_content": memory_content},
            ),
        )
        return {"op": "add", "item": item, "task_description": task_description}

class ValidationUpdater(BaseUpdater):
    def prepare_update_op(
        self, task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Prepare op to add memory only if successful."""
        if not success:
            return {"op": "noop", "task_description": task_description}
        
        # If successful, logic is identical to VanillaUpdater
        builder = get_builder(self.strategies.build, self.llm)
        memory_body = builder.build(task_description, trajectory)
        memory_content = f"Task: {task_description}\n\n{memory_body}"
        
        item = TextualMemoryItem(
            memory=task_description,
            metadata=_build_standard_metadata(
                base=metadata, task_description=task_description, strategies=self.strategies,
                confidence=self.memory_confidence,
                extra={"type": "procedure", "source": "conversation", "full_content": memory_content},
            ),
        )
        return {"op": "add", "item": item, "task_description": task_description}

class AdjustmentUpdater(BaseUpdater):
    def prepare_update_op(
        self, task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Prepare op: on success, add new; on failure, prepare an adjustment op."""
        if success:
            builder = get_builder(self.strategies.build, self.llm)
            memory_body = builder.build(task_description, trajectory)
            memory_content = f"Task: {task_description}\n\n{memory_body}"
            item = TextualMemoryItem(
                memory=task_description,
                metadata=_build_standard_metadata(
                    base=metadata, task_description=task_description, strategies=self.strategies,
                    confidence=self.memory_confidence,
                    extra={"type": "procedure", "source": "conversation", "full_content": memory_content},
                )
            )
            return {"op": "add", "item": item, "task_description": task_description}

        # Failure path
        reflection = self._generate_reflection(task_description, trajectory)
        mode = (self.adjustment_config.mode or "append").lower()

        if mode == "inplace":
            return self._prepare_inplace_adjust(task_description, trajectory, reflection, retrieved_memory_ids or [], metadata)
        
        return self._prepare_append_adjust(task_description, trajectory, reflection, retrieved_memory_ids or [], metadata)

    def _prepare_append_adjust(self, task_description: str, failed_trajectory: str, reflection: str, 
                               related_ids: List[str], metadata: Optional[Dict[str, Any]]) -> Dict:
        """Prepares an 'add' operation for a new reflection memory."""
        adjustment_content = (
            f"TASK REFLECTION:\nTask: {task_description}\n\n"
            f"What went wrong:\n{reflection}\n\nFailed approach:\n{failed_trajectory}\n"
        )
        meta = _build_standard_metadata(
            base=metadata, task_description=task_description, strategies=self.strategies,
            confidence=self.memory_confidence * self.adjustment_config.confidence_factor,
            extra={
                "type": "adjustment", "source": "conversation", "source_detail": "reflection", 
                "related_memory_ids": related_ids, "full_content": adjustment_content,
            },
        )
        item = TextualMemoryItem(memory=task_description, metadata=meta)
        return {"op": "add", "item": item, "task_description": task_description}

    def _prepare_inplace_adjust(self, task_description: str, failed_trajectory: str, reflection: str,
                                related_ids: List[str], metadata: Optional[Dict[str, Any]]) -> Dict:
        """Prepares an 'update' operation to modify an existing memory."""
        if not related_ids:
            return {"op": "noop", "task_description": task_description}
        
        text_mem = _get_text_mem(self.mos, self.user_id, self.default_cube_id)
        mem_id_to_update = related_ids[0]  # Update the most relevant memory

        try:
            old_item = text_mem.get(mem_id_to_update)
        except Exception as e:
            logger.warning(f"Could not find memory {mem_id_to_update} to update. Skipping. Error: {e}")
            return {"op": "noop", "task_description": task_description}

        old_meta = getattr(old_item, "metadata", None)
        old_meta_dict = old_meta.model_dump() if hasattr(old_meta, "model_dump") else dict(old_meta or {})
        prev_full = old_meta_dict.get("full_content", f"Task: {old_item.memory}\n\n(Original content unavailable)")

        new_full_content = (
            f"{prev_full}\n\n--- ADJUSTMENT NOTE ({_now_iso()}) ---\n"
            f"A similar task failed: {task_description}\n\n"
            f"Reflection:\n{reflection}\n"
        )
        
        new_meta = _build_standard_metadata(
            base={**old_meta_dict, **(metadata or {})}, task_description=old_item.memory, # Keep original task desc in meta
            strategies=self.strategies,
            confidence=(old_meta_dict.get("confidence") or self.memory_confidence) * self.adjustment_config.confidence_factor,
            extra={"full_content": new_full_content},
        )

        return {
            "op": "update",
            "id": mem_id_to_update,
            "data": {"id": mem_id_to_update, "memory": old_item.memory, "metadata": new_meta.model_dump()},
            "task_description": task_description
        }


# ------------ Factory ------------

def get_updater(
    strategy: UpdateStrategy,
    *,
    mos: MOS,
    user_id: str,
    strategies: StrategyConfiguration,
    llm: Any,
    num_workers: int = 32,
    default_cube_id: Optional[str] = None,
    memory_confidence: float = 100.0,
    adjustment_mode: str = "append",
    adjustment_confidence_factor: float = 0.8,
) -> BaseUpdater:
    cfg = AdjustmentConfig(mode=adjustment_mode, confidence_factor=adjustment_confidence_factor)
    if strategy == UpdateStrategy.VANILLA:
        return VanillaUpdater(mos, num_workers, user_id, strategies, llm, default_cube_id=default_cube_id, memory_confidence=memory_confidence, adjustment_config=cfg)
    if strategy == UpdateStrategy.VALIDATION:
        return ValidationUpdater(mos, num_workers, user_id, strategies, llm, default_cube_id=default_cube_id, memory_confidence=memory_confidence, adjustment_config=cfg)
    return AdjustmentUpdater(mos, num_workers, user_id, strategies, llm, default_cube_id=default_cube_id, memory_confidence=memory_confidence, adjustment_config=cfg)

