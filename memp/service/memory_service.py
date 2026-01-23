"""
Core MemoryService implementation for the Memp procedural memory system.

This module provides the central MemoryService class that integrates with MemOS
to implement the Build/Retrieve/Update memory management strategies.
"""

import logging
import random
import time
import math
import statistics
from datetime import datetime, timezone

from collections import defaultdict

from typing import List, Dict, Any, Tuple, Optional


# Hotfix for memos.utils missing `timed` decorator in some versions
# We patch it early before importing other memos submodules.
try:
    import memos.utils as _memos_utils  # type: ignore

    if not hasattr(_memos_utils, "timed"):
        import functools

        def _timed(
            func=None,
            *,
            name: str | None = None,
            logger: logging.Logger | None = None,
            level: int = logging.INFO,
        ):
            """Simple timing decorator compatible with memos' expected API.

            Usage:
              @_timed
              def f(...): ...

              or

              @_timed(name="custom")
              def g(...): ...
            """
            if func is None:
                # Called as @_timed(...)
                return lambda real_func: _timed(
                    real_func, name=name, logger=logger, level=level
                )

            log = logger or logging.getLogger("memos.timed")

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000.0
                    try:
                        log.log(
                            level,
                            f"[timed]{' ' + name if name else ''} {func.__qualname__} took {duration_ms:.2f} ms",
                        )
                    except Exception:
                        # Best-effort logging; never break the call path
                        pass

            return wrapper

        _memos_utils.timed = _timed  # type: ignore[attr-defined]
except Exception:
    # If anything goes wrong, fail open without blocking imports
    pass

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from memos.configs.mem_os import MOSConfig
from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.mem_os.main import MOS
from memos.mem_cube.general import GeneralMemCube
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata

from .strategies import (
    BuildStrategy,
    RetrieveStrategy,
    UpdateStrategy,
    StrategyConfiguration,
)
from .keyer import AveFactKeyer, SimpleKeyer
from .procedural_memory import ProceduralMemory
from .builders import get_builder
from .retrievers import get_retriever
from .updater import get_updater
from ..utils.task_id import extract_task_id

# Configure logger
logger = logging.getLogger(__name__)
from ..providers.base import BaseLLM, BaseEmbedder

from .value_driven import RLConfig, ValueAwareSelector, QValueUpdater, MemoryCurator


def _now_iso() -> str:
    return datetime.now().isoformat()


def get_embedding_with_retry(embed_func, text, max_retries=5, base_delay=2.0):
    """
    Retry for embedding queries
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = embed_func(text)
            return result

        except Exception as e:
            logger.warning(
                f"[Retry {attempt}/{max_retries}] Embedding API 调用失败: {e}"
            )
            if attempt == max_retries:
                logger.error("达到最大重试次数，仍未成功。")
                raise

            sleep_time = base_delay * (2 ** (attempt - 1))
            time.sleep(sleep_time)


def _meta_to_dict(meta: Any) -> Dict[str, Any]:
    """Convert TextualMemoryMetadata or dict-like to a plain dict safely."""
    if meta is None:
        return {}
    try:
        if hasattr(meta, "model_extra"):
            return dict(meta.model_dump())
    except Exception:
        pass
    try:
        if isinstance(meta, dict):
            return dict(meta)
    except Exception:
        pass
    # best effort fallback
    try:
        return dict(getattr(meta, "__dict__", {}))
    except Exception:
        return {}


def _parse_datetime(value: Any) -> datetime | None:
    """Best-effort parse of various datetime string formats."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Try python's ISO parser first
        try:
            return datetime.fromisoformat(text)
        except Exception:
            pass
        # Fallback patterns
        patterns = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in patterns:
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
    return None


def _resolve_snapshot_dirs(
    snapshot_root: str, meta: Optional[Dict[str, Any]]
) -> tuple[str, str, int]:
    """Resolve snapshot cube/qdrant directories from optional snapshot_meta.json.

    IMPORTANT: meta["qdrant_dir"] may be None if qdrant copy failed during save.
    Treat None/empty as "missing" and fall back to <snapshot_root>/qdrant.
    """
    import os

    cube_dir = os.path.join(snapshot_root, "cube")
    qdrant_dir = os.path.join(snapshot_root, "qdrant")
    checkpoint_id = 0

    if isinstance(meta, dict):
        try:
            cube_dir = meta.get("cube_dir") or cube_dir
        except Exception:
            pass
        try:
            qdrant_dir = meta.get("qdrant_dir") or qdrant_dir
        except Exception:
            pass
        try:
            checkpoint_id = int(meta.get("checkpoint_id", checkpoint_id))
        except Exception:
            checkpoint_id = 0

    return cube_dir, qdrant_dir, checkpoint_id


def _coerce_success(value: Any) -> Optional[bool]:
    """Parse a best-effort boolean success flag.

    Returns:
        True / False if value can be reliably interpreted, otherwise None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return None
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
        # Treat "unknown"/"n/a" and other strings as unknown
        return None
    return None


class MemoryService:
    """
    Core memory service implementing the Memp procedural memory algorithms.

    This service orchestrates the three-phase memory management:
    1. Build: Construct procedural memories from task trajectories
    2. Retrieve: Find relevant memories for new tasks
    3. Update: Learn from new experiences and adjust existing memories

    The service integrates with MemOS for storage and retrieval operations.
    """

    def __init__(
        self,
        mos_config_path: str,
        llm_provider: BaseLLM,
        embedding_provider: BaseEmbedder,
        strategy_config: Optional[StrategyConfiguration] = None,
        user_id: str = "memp_user",
        num_workers: int = 32,
        **kwargs,
    ):
        """
        Initialize the memory service.

        Args:
            mos_config_path: Path to MemOS configuration file
            llm_provider: LLM provider for text generation and keyword extraction
            embedding_provider: Embedding provider for vector operations
            strategy_config: Strategy configuration (defaults to main combination)
            user_id: User ID for multi-tenant memory management
            **kwargs: Additional configuration parameters
        """
        # Set strategy configuration
        self.strategy_config = (
            strategy_config or StrategyConfiguration.main_combination()
        )
        self.user_id = user_id
        self.num_workers = num_workers
        import threading

        self.db_max_concurrency: int = int(kwargs.get("db_max_concurrency", 4))
        self._db_gate = threading.BoundedSemaphore(self.db_max_concurrency)
        # Simple in-memory cache for loaded memories to reduce DB roundtrips
        self._mem_cache: dict[str, Any] = {}
        self._mem_cache_max_size: int = int(kwargs.get("mem_cache_max_size", 10000))
        # Lightweight Q-value cache: {mem_id: q_value} for fast Q-value updates
        self._q_cache: dict[str, float] = {}
        self._q_cache_max_size: int = int(kwargs.get("q_cache_max_size", 1000000))
        # Store providers
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

        # Similarity normalization constants (default from corpus stats; override via kwargs if needed)
        self.sim_norm_mean: float = float(kwargs.get("sim_norm_mean", 0.1856827586889267))
        self.sim_norm_std: float = float(kwargs.get("sim_norm_std", 0.09407906234264374)) or 1.0
        # Optional: disable z-score normalization in hybrid scoring.
        # This is intentionally plumbed via kwargs so only selected runners (e.g., LLB)
        # change behavior without affecting other benchmarks by default.
        self.use_z_score_normalization: bool = bool(kwargs.get("use_z_score_normalization", True))
        # Optional: LLB-only Phase-B behavior to reduce repeated same-task memories.
        self.dedup_by_task_id: bool = bool(kwargs.get("dedup_by_task_id", False))

        # Initialize MemOS
        try:
            self.mos_config = MOSConfig.from_json_file(mos_config_path)
            self.mos = MOS(self.mos_config)

            # Create user if doesn't exist
            self.mos.create_user(user_id=self.user_id)

            import os

            base_root = os.path.abspath("./results/mem_cubes")
            os.makedirs(base_root, exist_ok=True)
            # timestamped cube dir for historical isolation
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._base_root = base_root
            self._cube_timestamp = ts_str
            cube_dir = os.path.join(base_root, self.user_id, ts_str)
            self._cube_dir = cube_dir

            os.makedirs(cube_dir, exist_ok=True)
            extractor_llm_cfg = {
                "backend": self.mos_config.chat_model.backend,
                "config": self.mos_config.chat_model.config.model_dump(),
            }
            try:
                embedder_conf_obj = self.mos_config.mem_reader.config.embedder.config
                embedder_model_name = embedder_conf_obj.model_name_or_path
                embedder_provider = getattr(embedder_conf_obj, "provider", "openai")
                embedder_base_url = getattr(embedder_conf_obj, "base_url", None)
                embedder_api_key = getattr(embedder_conf_obj, "api_key", None)
            except Exception:
                embedder_conf_obj = None
                embedder_model_name = (
                    getattr(self.embedding_provider, "model", None)
                    or "text-embedding-3-large"
                )
                embedder_provider = "openai"
                embedder_base_url = None
                embedder_api_key = None

            embedder_cfg = {
                "backend": "universal_api",
                "config": {
                    "provider": embedder_provider,
                    "model_name_or_path": embedder_model_name,
                    "api_key": embedder_api_key
                    or self.mos_config.chat_model.config.api_key,
                    "base_url": embedder_base_url
                    or self.mos_config.chat_model.config.api_base,
                },
            }

            qdrant_dir = os.path.abspath(
                os.path.join(base_root, "..", "qdrant", self.user_id, ts_str)
            )
            os.makedirs(qdrant_dir, exist_ok=True)
            self._qdrant_dir = qdrant_dir
            vector_db_cfg = {
                "backend": "qdrant",
                "config": {
                    "collection_name": f"memp_{self.user_id}_{ts_str}",
                    "vector_dimension": 3072,
                    "distance_metric": "cosine",
                    "path": qdrant_dir,
                },
            }
            cube_cfg = GeneralMemCubeConfig(
                user_id=self.user_id,
                text_mem={
                    "backend": "general_text",
                    "config": {
                        "extractor_llm": extractor_llm_cfg,
                        "embedder": embedder_cfg,
                        "vector_db": vector_db_cfg,
                    },
                },
                act_mem={"backend": "uninitialized", "config": {}},
                para_mem={"backend": "uninitialized", "config": {}},
            )
            # Use a stable mem_cube_id and prefer reusing an existing cube from disk
            if os.listdir(cube_dir):
                try:
                    cube = GeneralMemCube.init_from_dir(cube_dir)
                    print("reuse existing cube")
                except Exception:
                    cube = GeneralMemCube(cube_cfg)
                    cube.dump(cube_dir)
            else:
                cube = GeneralMemCube(cube_cfg)
                cube.dump(cube_dir)

            stable_id = f"cube_{self.user_id}_{ts_str}"
            self.mos.register_mem_cube(
                cube, mem_cube_id=stable_id, user_id=self.user_id
            )
            self.default_cube_id = stable_id  # remember for add() calls
            # no delete_all on init; timestamp ensures isolation

        except Exception as e:
            raise RuntimeError(f"Failed to initialize MemOS: {e}")

        # Configuration parameters (must be set before _init_key_generators)
        self.max_keywords = kwargs.get("max_keywords", 8)
        self.memory_confidence = kwargs.get("memory_confidence", 100.0)
        self.add_similarity_threshold = kwargs.get("add_similarity_threshold", 0.90)
        # Value-driven config and components (optional)
        self.enable_value_driven: bool = bool(kwargs.get("enable_value_driven", True))
        self.rl_config: Optional[RLConfig] = kwargs.get("rl_config", None)
        if self.enable_value_driven and self.rl_config is None:
            self.rl_config = RLConfig()

        # Initialize key generators for different retrieval strategies
        self._init_key_generators()

        if self.enable_value_driven and self.rl_config is not None:
            try:
                self.dict_memory = {}
                self.query_embeddings = {}
                self._value_selector = ValueAwareSelector(self.rl_config)
                self._q_updater = QValueUpdater(
                    self.mos,
                    self.user_id,
                    self.rl_config,
                    default_cube_id=self.default_cube_id,
                )
                self._curator = MemoryCurator(
                    self.mos,
                    self.user_id,
                    self.rl_config,
                    default_cube_id=self.default_cube_id,
                    q_updater=self._q_updater,
                )
                self.weight_sim = self.rl_config.weight_sim
                self.weight_q = self.rl_config.weight_q
            except Exception:
                # Fail gracefully; callers can still use baseline flows
                self.enable_value_driven = False

    def _init_key_generators(self) -> None:
        """Initialize key generators for retrieval strategies."""
        self.avefact_keyer = AveFactKeyer(
            self.llm_provider, self.embedding_provider, max_keywords=self.max_keywords
        )
        self.simple_keyer = SimpleKeyer(self.embedding_provider)

    def build_memory(
        self,
        task_description: str,
        trajectory: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build procedural memory from task trajectory using the configured build strategy.

        Args:
            task_description: Natural language description of the task
            trajectory: Detailed step-by-step trajectory of task execution
            metadata: Optional additional metadata

        Returns:
            Memory ID of the created memory

        Raises:
            RuntimeError: If memory building fails
        """
        try:
            # Extract source benchmark from metadata
            source_benchmark = (metadata or {}).get("source_benchmark", "unknown")
            raw_success = (metadata or {}).get("success", None)
            success = _coerce_success(raw_success)

            # Use Builder pattern to generate memory content
            builder = get_builder(self.strategy_config.build, self.llm_provider)
            memory_content = builder.build(task_description, trajectory)

            # Create procedural memory based on build strategy
            if self.strategy_config.build == BuildStrategy.TRAJECTORY:
                procedural_memory = ProceduralMemory.create_trajectory_memory(
                    task_description=task_description,
                    trajectory=memory_content,  # Use builder output
                    build_strategy=self.strategy_config.build,
                    retrieve_strategy=self.strategy_config.retrieve,
                    update_strategy=self.strategy_config.update,
                    source_benchmark=source_benchmark,
                    confidence_score=self.memory_confidence,
                )

            elif self.strategy_config.build == BuildStrategy.SCRIPT:
                procedural_memory = ProceduralMemory.create_script_memory(
                    task_description=task_description,
                    script=memory_content,  # Use builder output
                    trajectory=trajectory,  # Keep original trajectory for reference
                    build_strategy=self.strategy_config.build,
                    retrieve_strategy=self.strategy_config.retrieve,
                    update_strategy=self.strategy_config.update,
                    source_benchmark=source_benchmark,
                    confidence_score=self.memory_confidence,
                )

            else:  # BuildStrategy.PROCEDURALIZATION
                # For proceduralization, the builder returns combined content
                # Extract script from the combined content for metadata
                lines = memory_content.split("\n")
                script_start = None
                trajectory_start = None

                for i, line in enumerate(lines):
                    if line.strip() == "SCRIPT:":
                        script_start = i + 1
                    elif line.strip() == "TRAJECTORY:":
                        trajectory_start = i + 1
                        break

                if script_start is not None and trajectory_start is not None:
                    script = "\n".join(
                        lines[script_start : trajectory_start - 2]
                    ).strip()
                else:
                    # Fallback: generate script separately if parsing fails
                    script = self.llm_provider.generate_script(trajectory)

                procedural_memory = ProceduralMemory.create_procedural_memory(
                    task_description=task_description,
                    script=script,
                    trajectory=trajectory,
                    build_strategy=self.strategy_config.build,
                    retrieve_strategy=self.strategy_config.retrieve,
                    update_strategy=self.strategy_config.update,
                    source_benchmark=source_benchmark,
                    confidence_score=self.memory_confidence,
                )

            # Generate retrieval keys if needed
            if self.strategy_config.retrieve == RetrieveStrategy.AVEFACT:
                keywords = self.llm_provider.extract_keywords(
                    task_description, self.max_keywords
                )
                avefact_vector = self.avefact_keyer.generate_key(task_description)
                procedural_memory.memp_metadata.avefact_keywords = keywords
                procedural_memory.memp_metadata.avefact_vector = avefact_vector
            elif self.strategy_config.retrieve == RetrieveStrategy.QUERY:
                query_vector = self.simple_keyer.generate_key(task_description)
                procedural_memory.memp_metadata.query_vector = query_vector

            # Write using text_mem.add with retrieval key = task_description (embedding),
            # and full content stored in metadata.full_content
            full_content = (
                f"Task: {task_description}\n\n{procedural_memory.memory_content}"
            )

            # Get textual memory from the default cube
            mem_cube_id = getattr(self, "default_cube_id", None)
            if mem_cube_id is None or mem_cube_id not in self.mos.mem_cubes:
                raise RuntimeError(
                    "MemCube is not registered for the user or default_cube_id is missing"
                )
            text_mem = self.mos.mem_cubes[mem_cube_id].text_mem
            if text_mem is None:
                raise RuntimeError("Textual memory is not initialized in the MemCube")

            # Build metadata and item
            base_meta = {
                "type": "procedure",
                "source": "conversation",
                "source_benchmark": source_benchmark,
                "success": success,
                "strategy_build": self.strategy_config.build.value,
                "strategy_retrieve": self.strategy_config.retrieve.value,
                "strategy_update": self.strategy_config.update.value,
                "confidence": self.memory_confidence,
                "full_content": full_content,
            }
            # Initialize value fields when enabled
            if (
                getattr(self, "enable_value_driven", False)
                and getattr(self, "rl_config", None) is not None
            ):
                # User-selected behavior: unknown success defaults to q_init_pos.
                is_success = True if success is None else bool(success)
                base_meta |= {
                    "q_value": (
                        float(self.rl_config.q_init_pos)
                        if is_success
                        else float(self.rl_config.q_init_neg)
                    ),
                    "q_visits": 0,
                    "q_updated_at": datetime.now().isoformat(),
                    "last_used_at": datetime.now().isoformat(),
                    "reward_ma": 0.0,
                }
            item = TextualMemoryItem(
                memory=task_description,  # only task as retrieval key
                metadata=TextualMemoryMetadata(**base_meta),
            )

            text_mem.add([item])
            return str(item.id)

        except Exception as e:
            raise RuntimeError(f"Failed to build memory: {e}")

    def retrieve_value_aware(
        self, task_description: str, k: Optional[int] = None, threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Value-aware retrieval with ε-greedy re-ranking and unknown detection.

        Returns a dict with keys: action (memory_id or None), selected, candidates, simmax.
        """
        # Fallback to baseline retrieve if disabled
        try:
            candidates = self.retrieve(task_description, k=k, threshold=threshold)
            if (
                not getattr(self, "enable_value_driven", False)
                or getattr(self, "_value_selector", None) is None
            ):
                # No value layer; return top-1 greedy by similarity as selected (if any)
                if not candidates:
                    return {
                        "action": None,
                        "selected": None,
                        "candidates": [],
                        "simmax": 0.0,
                    }
                best = max(candidates, key=lambda x: float(x.get("similarity", 0.0)))
                return {
                    "action": best.get("memory_id"),
                    "selected": best,
                    "candidates": candidates,
                    "simmax": float(best.get("similarity", 0.0)),
                }
            # Use value-aware selection
            return self._value_selector.select(candidates, self.rl_config.topk)
        except Exception as e:
            raise RuntimeError(f"Value-aware retrieval failed: {e}")

    def retrieve(
        self,
        task_description: str,
        k: int = 1,
        threshold: float = 0.0,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a task using the configured retrieval strategy.

        Args:
            task_description: Natural language description of the task
            k: Number of memories to retrieve
            threshold: Minimum similarity threshold (for applicable strategies)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay (in seconds) between retries (exponential backoff)

        Returns:
            List of retrieved memory dictionaries

        Raises:
            RuntimeError: If memory retrieval fails after retries
        """
        retriever = get_retriever(
            self.strategy_config.retrieve,
            mos=self.mos,
            user_id=self.user_id,
            llm=self.llm_provider,
            keyer=self.avefact_keyer,
            max_keywords=self.max_keywords,
            embedder=self.embedding_provider,
        )

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return retriever.retrieve(task_description, k, threshold)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    logger.warning(
                        f"Retrieve attempt {attempt} failed: {e}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"Retrieve failed after {max_retries} attempts. Last error: {e}"
                    )
                    raise RuntimeError(
                        f"Failed to retrieve memories after {max_retries} attempts: {last_error}"
                    )

    def update_value(
        self,
        memory_id: Optional[str],
        reward: float,
        *,
        next_max_q: Optional[float] = None
    ) -> Optional[float]:
        """
        Update Q-value for the selected memory. If memory_id is None (null action),
        this is a no-op here (curation can be applied by caller if needed).

        Returns the new Q if updated, else None.
        """
        if not getattr(self, 'enable_value_driven', False) or getattr(self, '_q_updater', None) is None:
            return None
        if memory_id is None:
            return None
        try:
            return self._q_updater.update(memory_id, reward, next_max_q=next_max_q)
        except Exception as e:
            raise RuntimeError(f"Failed to update Q-value: {e}")


    def update_values(self, successes: list[float], retrieved_ids_list: list[list[str]]) -> dict[str, Optional[float]]:
        """
        Concurrently update Q-values for all retrieved memory_ids.

        Args:
            successes: list of rewards (one per trajectory)
            retrieved_ids_list: list of lists of memory_ids (aligned with successes)

        Returns:
            dict mapping memory_id -> new Q value (or None if failed).
        """
        if (
            not getattr(self, "enable_value_driven", False)
            or getattr(self, "_q_updater", None) is None
        ):
            return {}

        # Build update tasks: (memory_id, reward, next_max_q)
        updates = []
        for success, mem_ids in zip(successes, retrieved_ids_list):
            # Map success to 1 (True) or -1 (False/0)
            reward = (
                self.rl_config.success_reward
                if success
                else self.rl_config.failure_reward
            )
            for mem_id in mem_ids:
                updates.append((mem_id, reward, None))

        results = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_mem = {
                executor.submit(
                    self._q_updater.update, mem_id, reward, next_max_q
                ): mem_id
                for mem_id, reward, next_max_q in updates
            }
            for future in as_completed(future_to_mem):
                mem_id = future_to_mem[future]
                try:
                    new_q = future.result()
                    results[mem_id] = new_q

                    if new_q is not None:
                        # Check cache size limit (FIFO eviction)
                        if len(self._q_cache) >= self._q_cache_max_size:
                            num_to_remove = max(1, self._q_cache_max_size // 10)
                            for _ in range(num_to_remove):
                                self._q_cache.pop(next(iter(self._q_cache)), None)

                        self._q_cache[mem_id] = new_q

                except Exception as e:
                    results[mem_id] = None
                    logger.info(f"Failed to update Q-value for {mem_id}: {e}")
        return results

    def _add_to_mem_cache(self, mem_id: str, mem_obj: Any) -> None:
        """Add memory object to cache with FIFO eviction policy."""
        if mem_obj is None:
            return

        # Check capacity and evict if needed
        if len(self._mem_cache) >= self._mem_cache_max_size:
            # Remove oldest 10% entries (FIFO)
            num_to_remove = max(1, self._mem_cache_max_size // 10)
            for _ in range(num_to_remove):
                self._mem_cache.pop(next(iter(self._mem_cache)), None)
            logger.debug(
                f"[MemCache] FIFO evicted {num_to_remove} entries, size: {len(self._mem_cache)}/{self._mem_cache_max_size}"
            )

        self._mem_cache[mem_id] = mem_obj

    def update_memory(
        self,
        task_description: str,
        trajectory: str,
        success: bool,
        retrieved_memory_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Update memory based on new task experience using the configured update strategy.

        Args:
            task_description: Natural language description of the task
            trajectory: Detailed step-by-step trajectory of task execution
            success: Whether the task was completed successfully
            retrieved_memory_ids: IDs of memories that were retrieved for this task
            metadata: Optional additional metadata

        Returns:
            Memory ID if new memory was created or updated, None otherwise

        Raises:
            RuntimeError: If memory update fails
        """

        updater = get_updater(
            self.strategy_config.update,
            mos=self.mos,
            num_workers=self.num_workers,
            user_id=self.user_id,
            strategies=self.strategy_config,
            llm=self.llm_provider,
            default_cube_id=getattr(self, "default_cube_id", None),
            memory_confidence=self.memory_confidence,
            adjustment_mode=getattr(self, "adjustment_mode", "append"),
            adjustment_confidence_factor=0.8,
        )
        return updater.update(
            task_description=task_description,
            trajectory=trajectory,
            success=success,
            retrieved_memory_ids=retrieved_memory_ids,
            metadata=metadata,
        )

    # ----------------------- Timestamped cube helpers -----------------------
    def list_available_cube_timestamps(self) -> List[str]:
        """列出当前 user 下的所有时间戳 cube 目录（按时间排序）。"""
        import os

        user_dir = os.path.join(
            getattr(self, "_base_root", "./results/mem_cubes"), self.user_id
        )
        if not os.path.isdir(user_dir):
            return []
        ts_list = [
            d for d in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, d))
        ]
        return sorted(ts_list)

    def get_current_cube_id(self) -> Optional[str]:
        """返回当前默认使用的 cube_id。"""
        return getattr(self, "default_cube_id", None)

    def switch_to_cube_timestamp(self, timestamp: str) -> None:
        """
        切换到指定时间戳的历史 cube（只更改当前实例默认 cube，不清理/删除任何数据）。

        Args:
            timestamp: 形如 YYYYmmdd_HHMMSS 的时间戳目录名
        """
        import os

        cube_dir = os.path.join(
            getattr(self, "_base_root", "./results/mem_cubes"), self.user_id, timestamp
        )
        if not os.path.isdir(cube_dir):
            raise ValueError(f"Cube directory not found for timestamp: {timestamp}")

        try:
            cube = GeneralMemCube.init_from_dir(cube_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load cube from '{cube_dir}': {e}")

        mem_cube_id = f"cube_{self.user_id}_{timestamp}"
        self.mos.register_mem_cube(cube, mem_cube_id=mem_cube_id, user_id=self.user_id)
        self.default_cube_id = mem_cube_id
        self._cube_timestamp = timestamp
        self._cube_dir = cube_dir
        self._qdrant_dir = os.path.abspath(
            os.path.join(
                getattr(self, "_base_root", "./results/mem_cubes"),
                "..",
                "qdrant",
                self.user_id,
                timestamp,
            )
        )

    def _prepare_memory_item(
        self,
        task_description: str,
        trajectory: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TextualMemoryItem:
        """
        [NEW HELPER METHOD]
        This is the "thinking" part of build_memory. It performs all CPU/IO-bound
        operations (LLM calls, embedding) to create a TextualMemoryItem, but does NOT
        write it to the database. This method is safe to run in parallel.

        Args:
            task_description: Natural language description of the task.
            trajectory: Detailed step-by-step trajectory of task execution.
            metadata: Optional additional metadata.

        Returns:
            A fully prepared TextualMemoryItem ready for insertion.
        """
        # This code is a direct copy of the logic inside `build_memory`, but stops before `text_mem.add`

        # Extract source benchmark from metadata
        source_benchmark = (metadata or {}).get("source_benchmark", "unknown")
        raw_success = (metadata or {}).get("success", None)
        success = _coerce_success(raw_success)
        # Use Builder pattern to generate memory content
        builder = get_builder(self.strategy_config.build, self.llm_provider)
        memory_content = builder.build(task_description, trajectory)

        # Create procedural memory based on build strategy
        if self.strategy_config.build == BuildStrategy.TRAJECTORY:
            procedural_memory = ProceduralMemory.create_trajectory_memory(
                task_description=task_description,
                trajectory=memory_content,  # Use builder output
                build_strategy=self.strategy_config.build,
                retrieve_strategy=self.strategy_config.retrieve,
                update_strategy=self.strategy_config.update,
                source_benchmark=source_benchmark,
                confidence_score=self.memory_confidence,
            )

        elif self.strategy_config.build == BuildStrategy.SCRIPT:
            procedural_memory = ProceduralMemory.create_script_memory(
                task_description=task_description,
                script=memory_content,  # Use builder output
                trajectory=trajectory,  # Keep original trajectory for reference
                build_strategy=self.strategy_config.build,
                retrieve_strategy=self.strategy_config.retrieve,
                update_strategy=self.strategy_config.update,
                source_benchmark=source_benchmark,
                confidence_score=self.memory_confidence,
            )

        else:  # BuildStrategy.PROCEDURALIZATION
            # For proceduralization, the builder returns combined content
            # Extract script from the combined content for metadata
            lines = memory_content.split("\n")
            script_start = None
            trajectory_start = None

            for i, line in enumerate(lines):
                if line.strip() == "SCRIPT:":
                    script_start = i + 1
                elif line.strip() == "TRAJECTORY:":
                    trajectory_start = i + 1
                    break

            if script_start is not None and trajectory_start is not None:
                script = "\n".join(lines[script_start : trajectory_start - 2]).strip()
            else:
                # Fallback: generate script separately if parsing fails
                script = self.llm_provider.generate_script(trajectory)

            procedural_memory = ProceduralMemory.create_procedural_memory(
                task_description=task_description,
                script=script,
                trajectory=trajectory,
                build_strategy=self.strategy_config.build,
                retrieve_strategy=self.strategy_config.retrieve,
                update_strategy=self.strategy_config.update,
                source_benchmark=source_benchmark,
                confidence_score=self.memory_confidence,
            )

        # Generate retrieval keys if needed
        if self.strategy_config.retrieve == RetrieveStrategy.AVEFACT:
            keywords = self.llm_provider.extract_keywords(
                task_description, self.max_keywords
            )
            avefact_vector = self.avefact_keyer.generate_key(task_description)
            procedural_memory.memp_metadata.avefact_keywords = keywords
            procedural_memory.memp_metadata.avefact_vector = avefact_vector
        elif self.strategy_config.retrieve == RetrieveStrategy.QUERY:
            query_vector = self.simple_keyer.generate_key(task_description)
            procedural_memory.memp_metadata.query_vector = query_vector

        # Write using text_mem.add with retrieval key = task_description (embedding),
        # and full content stored in metadata.full_content
        full_content = f"Task: {task_description}\n\n{procedural_memory.memory_content}"

        base_meta = {
            "type": "procedure",
            "source": "conversation",
            "source_benchmark": source_benchmark,
            "success": success,
            "strategy_build": self.strategy_config.build.value,
            "strategy_retrieve": self.strategy_config.retrieve.value,
            "strategy_update": self.strategy_config.update.value,
            "confidence": self.memory_confidence,
            "full_content": full_content,
        }
        if (
            getattr(self, "enable_value_driven", False)
            and getattr(self, "rl_config", None) is not None
        ):
            # User-selected behavior: unknown success defaults to q_init_pos.
            is_success = True if success is None else bool(success)
            base_meta |= {
                "q_value": (
                    float(self.rl_config.q_init_pos)
                    if is_success
                    else float(self.rl_config.q_init_neg)
                ),
                "q_visits": 0,
                "q_updated_at": datetime.now().isoformat(),
                "last_used_at": datetime.now().isoformat(),
                "reward_ma": 0.0,
            }
        item = TextualMemoryItem(
            memory=task_description,
            metadata=TextualMemoryMetadata(**base_meta),
        )
        return item

    def build_memories(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Builds memories in a batch from lists of data.
        Returns a list of the new memory IDs.
        """
        logger.info(f"Starting parallel build for {len(task_descriptions)} memories...")

        items_to_add = []
        metadatas = metadatas or [{} for _ in task_descriptions]

        # --- Phase 1: Parallel "Thinking" ---
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Use zip to iterate through all lists in parallel
            tasks = [
                executor.submit(self._prepare_memory_item, td, traj, meta)
                for td, traj, meta in zip(task_descriptions, trajectories, metadatas)
            ]

            for future in tqdm(tasks, desc="Building memories (Parallel Processing)"):
                try:
                    items_to_add.append(future.result())
                except Exception as e:
                    logger.error(f"Failed to prepare a memory item in parallel: {e}")

        if not items_to_add:
            logger.warning(
                "No memory items were successfully prepared. Aborting build."
            )
            return {}

        # --- Phase 2: Serial "Writing" ---
        # Now, add all the prepared items to the database in a single, safe, serial operation.
        logger.info(
            f"Writing {len(items_to_add)} prepared memory items to the database..."
        )
        mem_cube_id = getattr(self, "default_cube_id", None)
        if mem_cube_id is None or mem_cube_id not in self.mos.mem_cubes:
            raise RuntimeError("MemCube is not registered for the user.")

        text_mem = self.mos.mem_cubes[mem_cube_id].text_mem
        if text_mem is None:
            raise RuntimeError("Textual memory is not initialized in the MemCube.")

        # The actual database write operation
        for item in tqdm(items_to_add):
            text_mem.add([item])

        # Create the results map
        results: Dict[str, str] = {
            str(item.memory): str(item.id) for item in items_to_add
        }
        logger.info("Batch memory build complete.")
        return results

    def update_memories(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        successes: List[bool],
        retrieved_ids_list: List[List[str]],
        metadatas: Optional[List[Dict]] = None,
    ) -> List[Optional[str]]:
        """
        Updates memories in a batch by delegating to the parallel-capable
        updater's `update_batch` method.
        """
        logger.info("Delegating batch update to the configured updater...")

        # 1. Get the appropriate updater instance
        updater = get_updater(
            self.strategy_config.update,
            mos=self.mos,
            num_workers=self.num_workers,
            user_id=self.user_id,
            strategies=self.strategy_config,
            llm=self.llm_provider,
            default_cube_id=getattr(self, "default_cube_id", None),
            memory_confidence=self.memory_confidence,
            adjustment_mode=getattr(self, "adjustment_mode", "append"),
            adjustment_confidence_factor=0.8,
        )

        # 2. Call its parallel-capable batch method
        return updater.update_batch(
            task_descriptions=task_descriptions,
            trajectories=trajectories,
            successes=successes,
            retrieved_ids_list=retrieved_ids_list,
            metadatas=metadatas,
        )

    def add_memory(
        self,
        task_description: str,
        trajectory: str,
        success: bool,
        retrieved_memory_query: Optional[List[Tuple[str, float]]] = None,
        retrieved_memory_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Add a single memory entry into the system.

        This is a thin wrapper around `add_memories()` to avoid maintaining two
        divergent implementations. It preserves the historical single-item API
        while reusing the batch-safe update/write path.
        """
        try:
            results = self.add_memories(
                task_descriptions=[task_description],
                trajectories=[trajectory],
                successes=[bool(success)],
                retrieved_memory_queries=[retrieved_memory_query],
                retrieved_memory_ids_list=[retrieved_memory_ids],
                metadatas=[metadata],
            )
            if not results:
                return None
            first = results[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                return first[1]
            return None
        except Exception as e:
            import traceback

            print(f"[add_memory] Error: {e}\n{traceback.format_exc()}")
            return None


    def _normalize_similarity(self, sim: float) -> float:
        """Z-norm similarity using precomputed mean/std."""
        if not getattr(self, "use_z_score_normalization", True):
            return float(sim)
        std = self.sim_norm_std if self.sim_norm_std and self.sim_norm_std > 1e-9 else 1.0
        return (sim - self.sim_norm_mean) / std
    
    def _normalize_q(self, q: float, mean: float, std: float) -> float:
        """Z-norm q using provided mean/std (per-call stats)."""
        if not getattr(self, "use_z_score_normalization", True):
            return float(q)
        std = std if std and std > 1e-9 else 1.0
        z = (q - mean) / std
        return max(min(z, 3.0), -3.0)

    def retrieve_query(
        self,
        task_description: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Unified retrieval using similarity-Q hybrid weighting.
        No two-stage retrieval; directly scores candidates by sim and Q mixture.

        Returns:
            {
                "actions": [...],
                "selected": [...],
                "candidates": [...],
                "simmax": float
            }
        """

        try:
            # -------- Basic checks --------
            if not hasattr(self, "dict_memory") or not self.dict_memory:
                return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}

            if not getattr(self, "embedding_provider", None):
                raise RuntimeError("embedding_provider is required for local retrieval")

            embed = getattr(self.embedding_provider, "embed", None)
            if not callable(embed):
                raise RuntimeError("embedding_provider.embed() not callable")

            # -------- Compute query embedding --------
            queries = list(self.dict_memory.keys())
            if not queries:
                return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}

            query_vec = get_embedding_with_retry(embed, [task_description])[0]
            query_norm = math.sqrt(sum(x * x for x in query_vec)) or 1e-8

            query_embeddings = getattr(self, "query_embeddings", {})
            missing_queries = [q for q in queries if q not in query_embeddings]

            if missing_queries:
                logger.info(
                    f"Missing embeddings for {len(missing_queries)} queries, fetching..."
                )
                new_vecs = get_embedding_with_retry(embed, missing_queries)
                for q, v in zip(missing_queries, new_vecs):
                    query_embeddings[q] = v
                self.query_embeddings.update(query_embeddings)

            # -------- Compute similarity for all queries --------
            sim_list = []
            for q in queries:
                qv = query_embeddings[q]
                q_norm = math.sqrt(sum(x * x for x in qv)) or 1e-8
                sim = sum(a * b for a, b in zip(query_vec, qv)) / (query_norm * q_norm)
                if sim >= threshold:
                    sim_list.append((q, sim))
            sim_list.sort(key=lambda x: x[1], reverse=True)

            if k is not None:
                sim_list = sim_list[:k]
            if not sim_list:
                return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}

            # -------- Fetch memory objects, build candidate list --------
            candidates = []
            for q, sim in sim_list:
                mem_ids = self.dict_memory.get(q, [])
                for mid in mem_ids:
                    try:
                        mem_obj = self._mem_cache.get(mid)
                        if mem_obj is None:
                            with self._db_gate:
                                mem_obj = self.mos.get(
                                    mem_cube_id=self.default_cube_id,
                                    memory_id=mid,
                                    user_id=self.user_id,
                                )
                            if mem_obj is not None:
                                self._add_to_mem_cache(mid, mem_obj)

                        if mem_obj is not None:
                            md = getattr(mem_obj, "metadata", {})
                            content = None
                            try:
                                if hasattr(md, "model_extra"):
                                    content = md.model_extra.get("full_content")
                                elif isinstance(md, dict):
                                    content = md.get("full_content")
                            except Exception:
                                content = None

                            candidates.append(
                                {
                                    "memory_id": mid,
                                    "content": content,
                                    "similarity": float(sim),
                                    "metadata": md,
                                    "memory_item": mem_obj,
                                }
                            )
                    except Exception:
                        logger.info(f"Failed to load memory {mid}", exc_info=True)

            if not candidates:
                return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}

            # -------- Compute Q value for each candidate --------
            enriched = []
            simmax = 0.0
            q_values = []
            for c in candidates:
                sim = c["similarity"]
                simmax = max(simmax, sim)

                # Check _q_cache first for latest Q-value
                mem_id = c.get("memory_id")
                md = _meta_to_dict(c.get("metadata"))
                # Stable "task" identifier for de-dup (LLB-only usage).
                # NOTE: task_id can legally be 0, so avoid truthiness-based fallbacks.
                task_id = extract_task_id(md)

                if mem_id and mem_id in self._q_cache:
                    q = self._q_cache[mem_id]
                else:
                    # Fall back to metadata
                    has_meta_q = isinstance(md, dict) and ("q_value" in md)
                    q = md.get("q_value", self.rl_config.q_init_pos)
                    logger.info(
                        f"[Q-Cache Miss] Using metadata Q-value for {mem_id}: {q}"
                    )
                    try:
                        q = float(q)
                    except Exception:
                        q = self.rl_config.q_init_pos

                    # Populate Q-cache from metadata when available
                    if mem_id and has_meta_q:
                        try:
                            self._q_cache[mem_id] = q
                            # Simple cap eviction to avoid unbounded growth
                            if len(self._q_cache) > self._q_cache_max_size:
                                num_to_remove = max(1, self._q_cache_max_size // 10)
                                for _ in range(num_to_remove):
                                    self._q_cache.pop(next(iter(self._q_cache)), None)
                        except Exception:
                            pass

                # recency boost
                if self.rl_config.recency_boost > 0 and md.get("last_used_at"):
                    ts = md["last_used_at"]
                    if isinstance(ts, str) and len(ts) >= 10:
                        q += self.rl_config.recency_boost

                # Optional Q floor (LLB may set this via experiment.llb_q_floor).
                q_floor = getattr(self.rl_config, "q_floor", None)
                if q_floor is not None:
                    try:
                        q = max(float(q_floor), float(q))
                    except Exception:
                        pass

                c_local = dict(c)
                c_local["q_estimate"] = q
                c_local["task_id"] = (str(task_id) if task_id is not None else None)
                q_values.append(q)
                enriched.append(c_local)

            # -------- Optional Q threshold --------
            q_min = getattr(self.rl_config, "q_min_threshold", None)
            if q_min is not None:
                enriched = [c for c in enriched if c["q_estimate"] >= q_min]

            if not enriched:
                return {
                    "actions": [],
                    "selected": [],
                    "candidates": [],
                    "simmax": simmax,
                }

            # -------- Hybrid scoring (similarity + Q) --------
            # You can adjust weights here
            w_sim = self.weight_sim
            w_q = self.weight_q

            # derive q normalization stats from current candidates
            if q_values:
                mean_q = float(statistics.fmean(q_values))
                std_q = float(statistics.pstdev(q_values)) if len(q_values) > 1 else 1.0
            else:
                mean_q, std_q = 0.0, 1.0

            for c in enriched:
                sim = c["similarity"]
                q = c["q_estimate"]

                sim_z = self._normalize_similarity(sim)
                q_z = self._normalize_q(q, mean_q, std_q)

                # two options:
                # 1. additive with normalized scores:
                c["similarity_z"] = sim_z
                c["q_z"] = q_z
                c["score"] = sim_z * w_sim + q_z * w_q
                # 2. multiplicative (if needed):
                # c["score"] = sim * q

            # -------- Sort by hybrid score --------
            enriched_sorted = sorted(enriched, key=lambda x: x["score"], reverse=True)

            # -------- epsilon-greedy sampling --------
            topk = min(self.rl_config.topk, len(enriched_sorted))
            if not getattr(self, "dedup_by_task_id", False):
                if random.random() < self.rl_config.epsilon:
                    selected = random.sample(enriched_sorted, topk)
                else:
                    selected = enriched_sorted[:topk]
            else:
                # LLB-only: de-dup by task_id while keeping epsilon-greedy behavior.
                # - greedy: iterate score-desc
                # - epsilon: shuffle before taking unique tasks
                pool = list(enriched_sorted)
                if random.random() < self.rl_config.epsilon:
                    random.shuffle(pool)

                selected = []
                seen_tasks: set[str] = set()
                for cand in pool:
                    tid = cand.get("task_id")
                    # If task_id missing, treat as unique by memory_id to avoid collapsing unrelated entries.
                    key = str(tid) if tid else f"__missing_task_id__:{cand.get('memory_id')}"
                    if key in seen_tasks:
                        continue
                    seen_tasks.add(key)
                    selected.append(cand)
                    if len(selected) >= topk:
                        break

            return {
                "actions": [s["memory_id"] for s in selected],
                "selected": selected,
                "candidates": enriched_sorted,
                "simmax": simmax,
            }, sim_list

        except Exception as e:
            logger.info(
                f"Local retrieve failed, task_desc: {task_description}", exc_info=True
            )
            raise RuntimeError(f"Local retrieve failed: {e}")

    def add_memories(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        successes: List[bool],
        retrieved_memory_queries: Optional[List[List[Tuple[str, float]]]] = None,
        retrieved_memory_ids_list: Optional[List[Optional[List[str]]]] = None,
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Batch version of `add_memory`.

        For each (task_description, trajectory, success), determine if a similar query already exists.
        If yes, attach the trajectory to that query; otherwise, create a new query entry.
        Stores query embeddings for new queries to avoid recomputation during retrieval.

        Args:
            task_descriptions: List of task descriptions.
            trajectories: List of trajectories corresponding to each task.
            successes: List of success flags.
            retrieved_memory_queries: Optional list of lists, each containing (query, score) pairs.
            retrieved_memory_ids_list: Optional list of retrieved memory ID lists for each task.
            metadatas: Optional list of metadata dicts.

        Returns:
            Dict mapping task_description → memory_id (or None if failed).
        """
        try:
            if not hasattr(self, "dict_memory"):
                self.dict_memory = {}
            if not hasattr(self, "query_embeddings"):
                self.query_embeddings = {}

            similarity_threshold = getattr(self, "add_similarity_threshold", 0.8)
            n = len(task_descriptions)

            if retrieved_memory_queries is None:
                retrieved_memory_queries = [None] * n
            if retrieved_memory_ids_list is None:
                retrieved_memory_ids_list = [None] * n
            if metadatas is None:
                metadatas = [None] * n
            td_list = []
            traj_list = []
            succ_list = []
            retrieved_ids_payload = []
            metadata_list = []

            for i in range(n):
                td = task_descriptions[i]
                traj = trajectories[i]
                succ = successes[i]
                retrieved_ids = retrieved_memory_ids_list[i]
                meta = dict(metadatas[i] or {})

                # 记录本次检索到的记忆，以便成功样本也能追溯引用链路
                if retrieved_ids:
                    meta["related_memory_ids"] = [
                        str(rid) for rid in retrieved_ids if rid
                    ]

                # Attach RL-style meta defaults.
                #
                # IMPORTANT: do NOT blindly override upstream metadata (runner may
                # provide q_value based on success/failure). Only fill missing
                # fields, and when q_value is missing/invalid, initialize from
                # q_init_pos/q_init_neg according to the success flag.
                meta.setdefault("success", bool(succ))

                rl_cfg = getattr(self, "rl_config", None)
                try:
                    q_init_pos = float(getattr(rl_cfg, "q_init_pos", 0.0))
                except Exception:
                    q_init_pos = 0.0
                try:
                    q_init_neg = float(getattr(rl_cfg, "q_init_neg", 0.0))
                except Exception:
                    q_init_neg = 0.0

                default_q = q_init_pos if bool(succ) else q_init_neg
                if "q_value" not in meta or meta.get("q_value") is None:
                    meta["q_value"] = default_q
                else:
                    # If provided but not castable, fall back to the default.
                    try:
                        meta["q_value"] = float(meta["q_value"])
                    except Exception:
                        meta["q_value"] = default_q

                meta.setdefault("q_visits", 0)
                meta.setdefault("q_updated_at", datetime.now().isoformat())
                meta.setdefault("last_used_at", datetime.now().isoformat())
                meta.setdefault("reward_ma", 0.0)

                # append to parallel lists
                td_list.append(td[:4096])  # Truncate to avoid excessive length)
                traj_list.append(traj)
                succ_list.append(succ)
                retrieved_ids_payload.append(retrieved_ids)
                metadata_list.append(meta)

            results = self.update_memories(
                task_descriptions=td_list,
                trajectories=traj_list,
                successes=succ_list,
                retrieved_ids_list=retrieved_ids_payload,
                metadatas=metadata_list,
            )

            for i, task_description in enumerate(task_descriptions):
                if i < len(results):
                    recorded_task, mem_id = results[i]
                    if recorded_task != task_description:
                        logger.warning(
                            f"Task description mismatch at index {i}: expected '{task_description}', got '{recorded_task}'"
                        )
                    mem_id = mem_id
                else:
                    logger.warning(f"No result found for task {task_description}")
                    mem_id = None

                retrieved_qs = retrieved_memory_queries[i]
                matched_query = None
                best_score = -1.0
                if retrieved_qs:
                    for query, score in retrieved_qs:
                        if score >= similarity_threshold and score > best_score:
                            matched_query, best_score = query, score

                if matched_query:
                    if matched_query not in self.dict_memory:
                        self.dict_memory[matched_query] = []
                    self.dict_memory[matched_query].append(mem_id)
                else:
                    self.dict_memory[task_description] = [mem_id]

                    if getattr(self, "embedding_provider", None):
                        embed = getattr(self.embedding_provider, "embed", None)
                        if callable(embed):
                            try:
                                vec = embed([task_description])[0]
                                self.query_embeddings[task_description] = vec
                                logger.info(
                                    f"Cached embedding for new query: {task_description[:40]}..."
                                )
                            except Exception:
                                logger.info(
                                    f"Failed to embed query '{task_description}'",
                                    exc_info=True,
                                )

            return results

        except Exception as e:
            import traceback

            print(f"[add_memories] Error: {e}\n{traceback.format_exc()}")
            return {}

    def save_checkpoint_snapshot(self, target_ck_dir: str, ckpt_id: str) -> dict:
        """
        保存当前 MemoryService 所指向的 MemCube 与向量库的独立快照到指定 ck 目录。
        目录结构：
        <ck_dir>/snapshot/
            - cube/                # GeneralMemCube.dump 导出的 config.json + textual_memory.json
            - qdrant/             # 直接拷贝当前 qdrant 本地目录（便于快速加载）
            - snapshot_meta.json  # 元信息（含校验/统计）
        返回：包含关键信息的字典，用于上层记录。
        """
        import os, json, shutil, hashlib

        os.makedirs(target_ck_dir, exist_ok=True)
        snapshot_root = os.path.join(target_ck_dir, f"snapshot/{ckpt_id}")
        cube_dst = os.path.join(snapshot_root, "cube")
        qdrant_dst = os.path.join(snapshot_root, "qdrant")
        os.makedirs(snapshot_root, exist_ok=True)
        # 清理旧目录以保证原子性
        if os.path.isdir(cube_dst):
            shutil.rmtree(cube_dst)
        if os.path.isdir(qdrant_dst):
            shutil.rmtree(qdrant_dst)
        # 1) dump cube 到全新目录（包含 textual_memory.json：完整向量与payload）
        cube_id = getattr(self, "default_cube_id", None)
        cube = self.mos.mem_cubes.get(cube_id) if cube_id else None
        if cube is None:
            raise RuntimeError("No active mem cube to snapshot.")
        cube.dump(cube_dst)
        # 2) 拷贝 qdrant 文件目录（便于直接加载；如失败不影响最小可复现）
        qdrant_src = getattr(self, "_qdrant_dir", None)
        qdrant_copied = False
        if qdrant_src and os.path.isdir(qdrant_src):
            try:
                shutil.copytree(qdrant_src, qdrant_dst)
                qdrant_copied = True
            except Exception:
                # Snapshot can still be loadable (qdrant can be rebuilt from cube dump),
                # but we must not write qdrant_dir=None to meta (it breaks loaders).
                logger.warning(
                    "Failed to copy qdrant dir from %s to %s; will keep empty qdrant dir and continue.",
                    qdrant_src,
                    qdrant_dst,
                    exc_info=True,
                )
                try:
                    os.makedirs(qdrant_dst, exist_ok=True)
                except Exception:
                    pass
        # 3) 统计与校验
        textual_path = os.path.join(cube_dst, "textual_memory.json")
        md5 = None
        if os.path.isfile(textual_path):
            h = hashlib.md5()
            with open(textual_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            md5 = h.hexdigest()
        total_count = 0
        try:
            res_all = self.mos.get_all(user_id=self.user_id)
            text_sections = (
                res_all.get("text_mem", []) if isinstance(res_all, dict) else []
            )
            for sec in text_sections:
                mems = sec.get("memories", [])
                total_count += len(mems) if isinstance(mems, list) else 0
        except Exception:
            total_count = 0
        meta = {
            "user_id": self.user_id,
            "mem_cube_id": cube_id,
            "cube_timestamp": getattr(self, "_cube_timestamp", None),
            "checkpoint_id": ckpt_id,
            "cube_dir": cube_dst,
            # Always write a string path so load_checkpoint_snapshot can safely use it.
            # If qdrant files weren't copied, loader may rebuild the local DB.
            "qdrant_dir": qdrant_dst,
            "textual_memory_md5": md5,
            "visible_memories": total_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(
            os.path.join(snapshot_root, "snapshot_meta.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        try:
            self._persist_local_caches(snapshot_root)
        except Exception:
            logger.warning(
                "Failed to persist local caches during snapshot save", exc_info=True
            )
        return meta

    def load_checkpoint_snapshot(
        self, snapshot_root: str, *, mem_cube_id: str | None = None
    ) -> int:
        """
        从 <ck_dir>/snapshot/ 加载快照并切换当前默认 cube，用于独立评测。
        - 如果 snapshot_root 不包含 epoch number，自动查找最大的 epoch 并加载
        - 优先读取 snapshot_meta.json；若缺失则按约定目录推断 cube/qdrant 路径。
        - 使用 GeneralMemCube.init_from_dir + default_config 覆盖 vector_db.path 指向快照内 qdrant。

        Returns:
            int: The checkpoint_id (section/epoch number) from the loaded snapshot
        """
        import os, json, re, shutil, sqlite3
        from memos.configs.mem_cube import GeneralMemCubeConfig

        # 如果 snapshot_root 不是具体的 epoch 目录，则自动查找最大的 epoch
        if os.path.isdir(snapshot_root) and not os.path.isfile(
            os.path.join(snapshot_root, "snapshot_meta.json")
        ):
            # 检查是否有子目录是数字（epoch number）
            try:
                epoch_dirs = []
                for item in os.listdir(snapshot_root):
                    item_path = os.path.join(snapshot_root, item)
                    if os.path.isdir(item_path) and item.isdigit():
                        epoch_dirs.append(int(item))

                if epoch_dirs:
                    max_epoch = max(epoch_dirs)
                    snapshot_root = os.path.join(snapshot_root, str(max_epoch))
                    logger.info(
                        f"Auto-selected latest checkpoint: epoch {max_epoch} from {snapshot_root}"
                    )
            except Exception as e:
                logger.warning(f"Failed to auto-detect epoch directory: {e}")

        # 解析路径
        meta_path = os.path.join(snapshot_root, "snapshot_meta.json")
        cube_dir = os.path.join(snapshot_root, "cube")
        qdrant_dir = os.path.join(snapshot_root, "qdrant")
        checkpoint_id = 0  # Default to 0 if not found

        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                cube_dir, qdrant_dir, checkpoint_id = _resolve_snapshot_dirs(
                    snapshot_root, meta
                )
            except Exception:
                pass
        if not os.path.isdir(cube_dir):
            raise ValueError(f"Snapshot cube directory not found: {cube_dir}")
        # 构造 default_config 以覆盖向量库路径与基础 provider 配置
        chat = self.mos_config.chat_model
        openai_cfg = chat.config.model_dump()
        embedder = self.mos_config.mem_reader.config.embedder.config
        default_cfg = GeneralMemCubeConfig(
            user_id=self.user_id,
            text_mem={
                "backend": "general_text",
                "config": {
                    "extractor_llm": {"backend": chat.backend, "config": openai_cfg},
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": getattr(embedder, "provider", None),
                            "model_name_or_path": getattr(
                                embedder, "model_name_or_path", None
                            ),
                            "api_key": getattr(embedder, "api_key", None)
                            or openai_cfg.get("api_key"),
                            "base_url": getattr(embedder, "base_url", None)
                            or openai_cfg.get("api_base"),
                        },
                    },
                    "vector_db": {
                        "backend": "qdrant",
                        "config": {
                            "collection_name": f"memp_{self.user_id}_snapshot",
                            "vector_dimension": 3072,
                            "distance_metric": "cosine",
                            "path": qdrant_dir,
                        },
                    },
                },
            },
            act_mem={"backend": "uninitialized", "config": {}},
            para_mem={"backend": "uninitialized", "config": {}},
        )
        # 载入并注册
        # Ensure qdrant directory exists; QdrantLocal will create internal sqlite files.
        if not isinstance(qdrant_dir, str) or not qdrant_dir:
            qdrant_dir = os.path.join(snapshot_root, "qdrant")
        try:
            os.makedirs(qdrant_dir, exist_ok=True)
        except Exception:
            logger.warning("Failed to ensure qdrant_dir exists: %r", qdrant_dir, exc_info=True)

        def _is_sqlite_malformed(err: Exception) -> bool:
            msg = str(err).lower()
            return (
                isinstance(err, sqlite3.DatabaseError)
                or "database disk image is malformed" in msg
                or "sqlite" in msg and "malformed" in msg
            )

        try:
            cube = GeneralMemCube.init_from_dir(cube_dir, default_config=default_cfg)
        except Exception as e:
            # Common failure mode: qdrant_client local persistence uses SQLite.
            # If the sqlite file inside qdrant_dir is corrupted (often due to
            # interrupted writes or non-atomic snapshot copy), we can rebuild it
            # by recreating qdrant_dir and reloading from cube dump.
            if _is_sqlite_malformed(e):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = f"{qdrant_dir}.corrupt_{ts}"
                logger.warning(
                    "Detected corrupted Qdrant local DB at %s; backing up to %s and rebuilding.",
                    qdrant_dir,
                    backup_dir,
                    exc_info=True,
                )
                try:
                    if os.path.isdir(qdrant_dir):
                        shutil.move(qdrant_dir, backup_dir)
                except Exception:
                    logger.warning(
                        "Failed to move corrupted qdrant dir; will try deleting in-place.",
                        exc_info=True,
                    )
                    try:
                        if os.path.isdir(qdrant_dir):
                            shutil.rmtree(qdrant_dir)
                    except Exception:
                        pass
                os.makedirs(qdrant_dir, exist_ok=True)
                cube = GeneralMemCube.init_from_dir(cube_dir, default_config=default_cfg)
            else:
                raise
        target_id = mem_cube_id or f"cube_{self.user_id}_snapshot"
        # register_mem_cube 如果目标 ID 已存在会直接跳过，因此需要在加载快照时显式替换掉旧的 cube
        existing_cubes = getattr(self.mos, "mem_cubes", {})
        if target_id in existing_cubes:
            try:
                self.mos.unregister_mem_cube(target_id, user_id=self.user_id)
            except Exception:
                logger.warning(
                    "Failed to unregister existing cube %s, force overriding in memory.",
                    target_id,
                    exc_info=True,
                )
                existing_cubes.pop(target_id, None)
        self.mos.register_mem_cube(cube, mem_cube_id=target_id, user_id=self.user_id)
        self.default_cube_id = target_id
        self._cube_dir = cube_dir
        self._qdrant_dir = qdrant_dir
        cache_dir = os.path.join(snapshot_root, "local_cache")
        restored_cache = False
        if os.path.isdir(cache_dir):
            try:
                restored_cache = self._restore_local_caches(cache_dir)
            except Exception:
                logger.warning(
                    "Failed to restore local caches from %s", cache_dir, exc_info=True
                )
        if not restored_cache or not getattr(self, "dict_memory", None):
            res_all = self.mos.get_all(mem_cube_id=target_id, user_id=self.user_id)
            text_sections = (
                res_all.get("text_mem", []) if isinstance(res_all, dict) else []
            )
            # cutoff_dt = datetime.fromisoformat("2025-11-28 05:39:34")
            logger.info("Rebuilding local memory")
            rebuilt = self._rebuild_local_memory_index(text_sections)
            logger.info("Local memory index rebuilt with %s entries", rebuilt)
        else:
            logger.info(
                "Local caches restored from snapshot cache directory %s", cache_dir
            )

        logger.info(f"Checkpoint loaded from section/epoch {checkpoint_id}")
        return checkpoint_id

    def _rebuild_local_memory_index(
        self,
        text_sections: List[Dict[str, Any]],
        *,
        cutoff_before: datetime | None = None,
    ) -> int:
        """Reconstruct dict_memory/cache from MOS get_all() output."""
        if not hasattr(self, "dict_memory") or not isinstance(self.dict_memory, dict):
            self.dict_memory = {}
        else:
            self.dict_memory.clear()

        if not hasattr(self, "_mem_cache") or not isinstance(self._mem_cache, dict):
            self._mem_cache = {}
        else:
            self._mem_cache.clear()

        # Clear Q-cache as well
        if not hasattr(self, "_q_cache") or not isinstance(self._q_cache, dict):
            self._q_cache = {}
        else:
            self._q_cache.clear()

        if not hasattr(self, "query_embeddings") or not isinstance(
            self.query_embeddings, dict
        ):
            self.query_embeddings = {}
        else:
            self.query_embeddings.clear()

        added = 0
        for sec in text_sections or []:
            mems = sec.get("memories", [])
            if not isinstance(mems, list):
                continue
            for mem in mems:
                mem_id = getattr(mem, "id", None)
                query = getattr(mem, "memory", None)
                if mem_id is None and isinstance(mem, dict):
                    mem_id = mem.get("id")
                if query is None and isinstance(mem, dict):
                    query = mem.get("memory")
                if not mem_id or not query:
                    continue

                md = getattr(mem, "metadata", None)
                if md is None and isinstance(mem, dict):
                    md = mem.get("metadata")
                md_dict = _meta_to_dict(md)

                mem_time_value = md_dict.get("memory_time")
                mem_dt = _parse_datetime(mem_time_value)
                if cutoff_before:
                    if mem_dt is None:
                        # If we cannot determine the timestamp, skip to keep filtering explicit.
                        continue
                    if mem_dt >= cutoff_before:
                        continue

                bucket = self.dict_memory.setdefault(query, [])
                bucket.append(mem_id)
                self._add_to_mem_cache(mem_id, mem)
                added += 1
        query_embeddings = getattr(self, "query_embeddings", {})
        queries = list(self.dict_memory.keys())
        missing_queries = [q for q in queries if q not in query_embeddings]
        embed = getattr(self.embedding_provider, "embed", None)
        if missing_queries and callable(embed):
            batch_size = getattr(self, "embedding_batch_size", 256)
            for start in tqdm(
                range(0, len(missing_queries), batch_size),
                desc="Rebuilding query embeddings",
            ):
                batch_queries = missing_queries[start : start + batch_size]
                batch_vecs = get_embedding_with_retry(embed, batch_queries)
                for q, v in zip(batch_queries, batch_vecs):
                    query_embeddings[q] = v
            self.query_embeddings.update(query_embeddings)

        return added

    def _persist_local_caches(self, snapshot_root: str) -> None:
        """Persist local cache dictionaries alongside the cube snapshot."""
        import json
        import os

        cache_dir = os.path.join(snapshot_root, "local_cache")
        os.makedirs(cache_dir, exist_ok=True)

        def _write_json(path: str, payload: dict) -> None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

        dict_mem = getattr(self, "dict_memory", {}) or {}
        dict_payload: Dict[str, List[str]] = {}
        if isinstance(dict_mem, dict):
            for query, mem_ids in dict_mem.items():
                if not isinstance(mem_ids, list):
                    continue
                cleaned_ids = [str(mid) for mid in mem_ids if mid]
                if cleaned_ids:
                    dict_payload[str(query)] = cleaned_ids
        _write_json(os.path.join(cache_dir, "dict_memory.json"), dict_payload)

        query_embeddings = getattr(self, "query_embeddings", {}) or {}
        emb_payload: Dict[str, List[float]] = {}
        if isinstance(query_embeddings, dict):
            for query, vec in query_embeddings.items():
                if vec is None:
                    continue
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                try:
                    emb_payload[str(query)] = [float(x) for x in vec]
                except Exception:
                    continue
        _write_json(os.path.join(cache_dir, "query_embeddings.json"), emb_payload)

        mem_cache = getattr(self, "_mem_cache", {}) or {}
        mem_payload: Dict[str, Any] = {}
        if isinstance(mem_cache, dict):
            for mem_id, item in mem_cache.items():
                serialized = None
                if hasattr(item, "model_dump"):
                    try:
                        serialized = item.model_dump(mode="json")
                    except Exception:
                        serialized = item.model_dump()
                elif isinstance(item, dict):
                    serialized = item
                else:
                    serialized = getattr(item, "__dict__", None)
                if serialized:
                    mem_payload[str(mem_id)] = serialized
        _write_json(os.path.join(cache_dir, "mem_cache.json"), mem_payload)

        # Persist _q_cache (lightweight Q-value cache)
        q_cache = getattr(self, "_q_cache", {}) or {}
        q_cache_payload: Dict[str, float] = {}
        if isinstance(q_cache, dict):
            for mem_id, q_value in q_cache.items():
                try:
                    q_cache_payload[str(mem_id)] = float(q_value)
                except Exception:
                    continue
        _write_json(os.path.join(cache_dir, "q_cache.json"), q_cache_payload)

        logger.info(
            "Persisted local caches to %s (dict=%d, embeddings=%d, mem_cache=%d, q_cache=%d)",
            cache_dir,
            len(dict_payload),
            len(emb_payload),
            len(mem_payload),
            len(q_cache_payload),
        )

    def _restore_local_caches(self, cache_dir: str) -> bool:
        """Restore dict_memory/_mem_cache/_q_cache/query_embeddings if cache files exist."""
        import json
        import os

        restored_any = False
        dict_path = os.path.join(cache_dir, "dict_memory.json")
        if os.path.isfile(dict_path):
            try:
                with open(dict_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.dict_memory = {
                        str(k): [str(x) for x in (v or []) if x] for k, v in raw.items()
                    }
                    restored_any = True
            except Exception:
                logger.warning(
                    "Failed to restore dict_memory cache from %s",
                    dict_path,
                    exc_info=True,
                )

        emb_path = os.path.join(cache_dir, "query_embeddings.json")
        if os.path.isfile(emb_path):
            try:
                with open(emb_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.query_embeddings = {}
                    for k, vec in raw.items():
                        if isinstance(vec, list):
                            try:
                                self.query_embeddings[str(k)] = [float(x) for x in vec]
                            except Exception:
                                continue
                    restored_any = True
            except Exception:
                logger.warning(
                    "Failed to restore query embeddings cache from %s",
                    emb_path,
                    exc_info=True,
                )

        mem_path = os.path.join(cache_dir, "mem_cache.json")
        if os.path.isfile(mem_path):
            try:
                with open(mem_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                new_cache: Dict[str, TextualMemoryItem] = {}
                if isinstance(raw, dict):
                    for mid, payload in raw.items():
                        if not isinstance(payload, dict):
                            continue
                        try:
                            new_cache[str(mid)] = TextualMemoryItem(**payload)
                        except Exception:
                            try:
                                new_cache[str(mid)] = TextualMemoryItem.model_validate(
                                    payload
                                )
                            except Exception:
                                logger.debug(
                                    "Failed to rehydrate memory %s from cache",
                                    mid,
                                    exc_info=True,
                                )
                    self._mem_cache = new_cache
                    restored_any = True
            except Exception:
                logger.warning(
                    "Failed to restore _mem_cache from %s", mem_path, exc_info=True
                )

        # Restore _q_cache (lightweight Q-value cache)
        q_cache_path = os.path.join(cache_dir, "q_cache.json")
        if os.path.isfile(q_cache_path):
            try:
                with open(q_cache_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self._q_cache = {}
                    for mem_id, q_value in raw.items():
                        try:
                            self._q_cache[str(mem_id)] = float(q_value)
                        except Exception:
                            continue
                    restored_any = True
                    logger.info(
                        f"Restored _q_cache ({len(self._q_cache)} entries) from {q_cache_path}"
                    )
            except Exception:
                logger.warning(
                    "Failed to restore _q_cache from %s", q_cache_path, exc_info=True
                )

        return restored_any
