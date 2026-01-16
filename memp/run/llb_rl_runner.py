# memp/run/llb_rl_runner.py
import logging
import os
import sys
import yaml
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence, Tuple
from datetime import datetime
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import pandas as pd
import psutil
from torch.utils.tensorboard import SummaryWriter

from .base_runner import BaseRunner
from memp.service.memory_service import MemoryService
from memp.service.value_driven import RLConfig
from memp.providers.llm import OpenAILLM
from memp.providers.embedding import OpenAIEmbedder

from memp.lifelongbench_eval.prompts import (
    DEFAULT_SYSTEM_PROMPT as LLB_DEFAULT_SYSTEM_PROMPT,
    build_llb_prompt_with_memory,
    build_llb_system_prompt,
)
from memp.lifelongbench_eval.memory_context import format_llb_memory_context

# --- Setup LLB Path ---
# 动态查找项目根目录和 LLB 路径
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent  # memp/run/llb_rl_runner.py -> memp/
LLB_ROOT = _project_root / "3rdparty" / "LifelongAgentBench"

if not LLB_ROOT.exists():
    raise RuntimeError(f"LLB directory not found: {LLB_ROOT}")

# Python 3.10 兼容：为 enum.StrEnum 提供兜底实现
try:
    import enum as _enum

    if not hasattr(_enum, "StrEnum"):

        class _StrEnum(str, _enum.Enum):
            pass

        _enum.StrEnum = _StrEnum  # type: ignore[attr-defined]
    import typing as _typing

    if not hasattr(_typing, "reveal_type"):

        def _noop_reveal_type(x):
            return x

        _typing.reveal_type = _noop_reveal_type  # type: ignore[attr-defined]
    if not hasattr(_typing, "Self"):
        _typing.Self = object  # type: ignore[attr-defined]
except Exception:
    pass

if str(LLB_ROOT) not in sys.path:
    sys.path.insert(0, str(LLB_ROOT))

# 导入 LLB 组件
from src.agents.instance.language_model_agent import LanguageModelAgent  # type: ignore
from src.typings import (  # type: ignore
    Session,
    SampleStatus,
    SessionMetricCalculationPartial,
    TaskName,
    SessionEvaluationOutcome,
)
from src.tasks.instance.db_bench.task import DBBench  # type: ignore
from src.tasks.instance.os_interaction.task import OSInteraction  # type: ignore
from src.factories.chat_history_item import ChatHistoryItemFactory  # type: ignore

MAX_RETRIES = 4
RETRY_DELAY = 2

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = LLB_DEFAULT_SYSTEM_PROMPT


class LLBRunner(BaseRunner):
    """
    Runner for LifelongAgentBench tasks (DB, OS, KG).
    Handles memory-driven agent evaluation and training.
    """

    def __init__(
        self,
        root: Path,
        memory_service: MemoryService,
        llm_provider: OpenAILLM,
        embedding_provider: OpenAIEmbedder,
        exp_name: str,
        task: str,
        split_file: str,
        num_section: int,
        batch_size: int,
        max_steps: int,
        rl_config: Optional[RLConfig],
        retrieve_k: int = 1,
        mode: str = "train",
        bon: int = 0,
        random_seed: int = 42,
        valid_interval: int = 2,
        test_interval: int = 2,
        train_set_ratio: float = 1.0,
        start_section: int = 0,
        algorithm: str = "rl",
        val_before_train: bool = True,
        system_prompt: str = "",
        os_timeout: int = 20,
        sparql_url: Optional[str] = None,
        ontology_dir: Optional[str] = None,
        kg_offline_fallback: bool = False,
        limit: Optional[int] = None,
        valid_file: Optional[str] = None,
    ):
        self.root = root
        self.memory_service = memory_service
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.exp_name = exp_name
        self.task = task
        self.split_file = split_file
        self.random_seed = random_seed
        self.num_section = num_section
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.retrieve_k = retrieve_k
        self.mode = mode
        self.valid_interval = valid_interval
        self.test_interval = test_interval
        self.train_set_ratio = train_set_ratio
        self.start_section = start_section
        self.bon = bon
        self.algorithm = algorithm
        self.val_before_train = val_before_train
        self._base_system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        self.system_prompt = build_llb_system_prompt(
            task=self.task,
            base_prompt=self._base_system_prompt,
        )
        self.os_timeout = os_timeout
        self.sparql_url = sparql_url
        self.ontology_dir = ontology_dir
        self.kg_offline_fallback = kg_offline_fallback
        self.limit = limit
        self.results_log = []
        self.valid_file = valid_file

        self.rl_config: Optional[RLConfig] = rl_config

        # Create LLM adapter for LLB LanguageModelAgent
        from memp.lifelongbench_eval.lm_adapter import MempOpenAIAdapter

        self.adapter = MempOpenAIAdapter(self.llm_provider)

        # --- [TENSORBOARD] Initialize SummaryWriter ---
        tb_log_dir = (
            self.root
            / "logs"
            / "tensorboard"
            / f"exp_{self.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        self.writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
        self.ck_dir = (
            self.root
            / "results"
            / "llb"
            / f"exp_{self.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        )

        # Build LLB task and load datasets
        self._build_llb_task()
        self._load_eval_datasets()

    def _log_token_usage(self, section_num: int, mini_batch: Optional[int] = None):
        """Log current token usage for LLM and Embedding providers."""
        try:
            llm_usage = self.llm_provider.get_token_usage()
            emb_usage = self.embedding_provider.get_token_usage()

            context = f"Section {section_num}"
            if mini_batch is not None:
                context += f" Mini-batch {mini_batch}"

            logger.info(f"\n=== Token Usage after {context} ===")
            logger.info(f"LLM Prompt Tokens:     {llm_usage.get('prompt_tokens', 0)}")
            logger.info(
                f"LLM Completion Tokens: {llm_usage.get('completion_tokens', 0)}"
            )
            logger.info(f"LLM Total Tokens:      {llm_usage.get('total_tokens', 0)}")
            logger.info(f"Embedding Total Tokens: {emb_usage.get('total_tokens', 0)}")
            logger.info(
                f"GRAND TOTAL:           {llm_usage.get('total_tokens', 0) + emb_usage.get('total_tokens', 0)}"
            )
            logger.info("==========================================\n")

            # Log to TensorBoard (only for section level to avoid clutter)
            if hasattr(self, "writer") and self.writer and mini_batch is None:
                self.writer.add_scalar(
                    "Token_Usage/LLM_Total",
                    llm_usage.get("total_tokens", 0),
                    section_num,
                )
                self.writer.add_scalar(
                    "Token_Usage/Embedding_Total",
                    emb_usage.get("total_tokens", 0),
                    section_num,
                )
                self.writer.add_scalar(
                    "Token_Usage/Grand_Total",
                    llm_usage.get("total_tokens", 0) + emb_usage.get("total_tokens", 0),
                    section_num,
                )

        except Exception as e:
            logger.warning(f"Failed to log token usage: {e}")

    def _check_memory_usage(self, context: str = ""):
        """Monitor memory usage to detect memory leaks."""
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            logger.info(f"[Memory Monitor] {context}: {mem_mb:.2f} MB")
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")

    def _build_llb_task(self):
        """Build LLB task object and load dataset."""
        from memp.lifelongbench_eval.task_wrappers import (
            build_task,
            ensure_standard_prompts,
        )

        # Ensure standard prompts are generated
        ensure_standard_prompts()

        # Build task object
        self.task_obj, self.task_name = build_task(
            task=self.task,
            data_file_path=self.split_file,
            max_round=self.max_steps,
            os_timeout=self.os_timeout,
            kg_sparql_url=self.sparql_url,
            kg_ontology_dir=self.ontology_dir,
            kg_offline_fallback=self.kg_offline_fallback,
        )

        # Load dataset
        with open(self.split_file, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        logger.info(f"Loaded {len(self.dataset)} samples from {self.split_file}")

        # Split dataset into sections
        self._split_dataset()

    def _split_dataset(self):
        """Split dataset into sections based on num_section and train_set_ratio."""
        # Get all sample keys
        all_keys = sorted(list(self.dataset.keys()), key=lambda x: str(x))

        # Apply train_set_ratio
        if self.train_set_ratio < 1.0:
            num_total = len(all_keys)
            num_to_sample = int(num_total * self.train_set_ratio)
            logger.info(
                f"Sampling {num_to_sample} from {num_total} total samples ({self.train_set_ratio:.2%})"
            )
            random.seed(self.random_seed)
            all_keys = random.sample(all_keys, k=num_to_sample)

        # Apply limit if specified
        if self.limit is not None:
            all_keys = all_keys[: self.limit]
            logger.info(f"Limited to {len(all_keys)} samples")

        # Split into sections
        if self.num_section == 1:
            self.section_splits = [all_keys]
        else:
            # Copy all keys for each section instead of splitting
            self.section_splits = [list(all_keys) for _ in range(self.num_section)]

        logger.info(
            f"Split {len(all_keys)} samples into {len(self.section_splits)} sections"
        )
        for i, section_keys in enumerate(self.section_splits):
            logger.info(f"  Section {i}: {len(section_keys)} samples")

    def _load_eval_datasets(self):
        """Load validation dataset."""
        self.valid_dataset = {}

        if self.valid_file and os.path.exists(self.valid_file):
            with open(self.valid_file, "r", encoding="utf-8") as f:
                self.valid_dataset = json.load(f)
            logger.info(
                f"Loaded {len(self.valid_dataset)} validation samples from {self.valid_file}"
            )
        else:
            logger.info("No validation dataset specified or file not found")

    def _create_llb_agent(
        self, memory_context: Optional[str] = None
    ) -> LanguageModelAgent:
        """Create a LanguageModelAgent instance for LLB task execution.

        Args:
            memory_context: Optional memory context to prepend to system prompt

        Returns:
            LanguageModelAgent instance configured with system prompt
        """
        # Align prompt assembly ordering with memory_rl/dev/feat-mdp-llb:
        # system prompt -> (optional) memory context -> strict output constraints at the very end.
        if memory_context:
            full_prompt = build_llb_prompt_with_memory(
                task=self.task,
                base_prompt=self._base_system_prompt,
                memory_context=memory_context,
            )
        else:
            full_prompt = self.system_prompt

        # Create and return agent
        return LanguageModelAgent(
            language_model=self.adapter, system_prompt=full_prompt
        )

    def process_retrieve_mems(
        self, retrieved_mems: List[dict]
    ) -> Dict[str, List[dict]]:
        """Process retrieved memories into success/failed categories.

        Args:
            retrieved_mems: List of retrieved memory dictionaries

        Returns:
            Dictionary with 'successed' and/or 'failed' keys containing categorized memories
        """
        success_mems = []
        failed_mems = []

        for mem in retrieved_mems:
            metadata = mem["metadata"]
            # Access Pydantic model attribute directly
            is_success = getattr(metadata, "success", False)

            if is_success:
                success_mems.append(mem)
            else:
                failed_mems.append(mem)

        final_mems = {}
        if success_mems:
            final_mems["successed"] = success_mems
        if failed_mems:
            final_mems["failed"] = failed_mems

        return final_mems

    def _sample_from_indices(
        self,
        sample_indices: List[str],
        phase: str = "train",
        custom_dataset: Optional[Dict[str, Any]] = None,
        custom_data_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Sample trajectories for given sample indices.

        Args:
            sample_indices: List of sample keys to process
            phase: Phase name ('train', 'eval', etc.)
            custom_dataset: Optional custom dataset to use for evaluation
            custom_data_file: Optional custom data file path for building task_obj

        Returns:
            List of trajectory dictionaries with keys:
                - sample_index: str
                - task_description: str
                - retrieved_memories: List[dict]
                - retrieved_ids: List[str]
                - session: Session object
                - success: bool
                - steps: int (number of rounds)
        """
        completed_trajectories = []

        logger.info(f"Sampling {len(sample_indices)} trajectories in parallel...")

        # Process samples in parallel
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_idx = {}

            for idx in sample_indices:
                future = executor.submit(
                    self._sample_single_trajectory,
                    idx,
                    phase,
                    custom_dataset,
                    custom_data_file,
                )
                future_to_idx[future] = idx

            # Collect results
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(sample_indices),
                desc=f"Sampling {phase}",
            ):
                idx = future_to_idx[future]
                try:
                    traj = future.result()
                    if traj is not None:
                        completed_trajectories.append(traj)
                except Exception as e:
                    logger.error(
                        f"Error sampling trajectory for {idx}: {e}", exc_info=True
                    )

        logger.info(
            f"Completed {len(completed_trajectories)}/{len(sample_indices)} trajectories"
        )
        return completed_trajectories

    def _sample_single_trajectory(
        self,
        sample_index: str,
        phase: str = "train",
        custom_dataset: Optional[Dict[str, Any]] = None,
        custom_data_file: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Sample a single trajectory.

        Args:
            sample_index: Sample key
            phase: Phase name
            custom_dataset: Optional custom dataset to use instead of self.dataset
            custom_data_file: Optional custom data file path for building task_obj

        Returns:
            Trajectory dictionary or None if failed
        """
        try:
            # Use custom dataset if provided, otherwise use self.dataset
            dataset = custom_dataset if custom_dataset is not None else self.dataset
            data_file = (
                custom_data_file if custom_data_file is not None else self.split_file
            )

            # Get task entry
            entry = dataset[sample_index]
            task_description = self._task_description_from_entry(entry)

            # Retrieve memories
            retrieved_mems = []
            topk_queries = []
            processed_mems = {}
            memory_context = ""

            if self.memory_service is not None:
                try:
                    results = self.memory_service.retrieve_query(
                        task_description=task_description,
                        k=self.retrieve_k,
                        threshold=(
                            (
                                getattr(
                                    self.rl_config,
                                    "sim_threshold",
                                    getattr(self.rl_config, "tau", 0.0),
                                )
                                if self.rl_config
                                else 0.0
                            )
                        ),
                    )
                    # retrieve_query returns tuple: (dict with 'selected' key, topk_queries)
                    if isinstance(results, tuple):
                        retrieved_mems = results[0]["selected"]
                        topk_queries = results[1]
                    else:
                        retrieved_mems = []
                        topk_queries = []

                    # Process and categorize memories
                    processed_mems = self.process_retrieve_mems(retrieved_mems)

                    # Format memory context from categorized memories
                    if processed_mems:
                        memory_context = self._format_memory_context(processed_mems)
                except Exception as e:
                    logger.warning(f"Memory retrieval failed for {sample_index}: {e}")

            # Create agent with memory context
            agent = self._create_llb_agent(memory_context=memory_context)

            # Create new task instance for this sample (avoid state pollution)
            from memp.lifelongbench_eval.task_wrappers import build_task

            task_obj, _ = build_task(
                task=self.task,
                data_file_path=data_file,
                max_round=self.max_steps,
                os_timeout=self.os_timeout,
                kg_sparql_url=self.sparql_url,
                kg_ontology_dir=self.ontology_dir,
                kg_offline_fallback=self.kg_offline_fallback,
            )

            # Create session
            session = Session(task_name=self.task_name, sample_index=sample_index)

            # Reset task
            task_obj.reset(session)

            # Run inference loop
            step_count = 0
            while session.sample_status == SampleStatus.RUNNING:
                agent.inference(session)
                task_obj.interact(session)
                step_count += 1

                # Safety check
                if step_count > self.max_steps * 2:
                    logger.warning(
                        f"Sample {sample_index} exceeded max steps, terminating"
                    )
                    break

            # Complete the session
            task_obj.complete(session)

            # Check success
            success = self._session_success(session)

            # Convert session to trajectory string
            trajectory = self._session_to_trajectory(session)
            if not trajectory:
                trajectory = ""  # Fallback to empty string

            return {
                "sample_index": sample_index,
                "task_description": task_description,
                "trajectory": trajectory,  # String for add_memories
                "retrieved_mems": processed_mems,  # Use processed memories from process_retrieve_mems
                "retrieved_queries": (
                    topk_queries if topk_queries else [(task_description, 1.0)]
                ),  # Use actual topk_queries
                "session": session,
                "success": success,
                "steps": step_count,
            }

        except Exception as e:
            logger.error(
                f"Failed to sample trajectory for {sample_index}: {e}", exc_info=True
            )
            return None

    def _format_memory_context(
        self, processed_mems: Dict[str, List[dict]], budget_tokens: Optional[int] = None
    ) -> str:
        return format_llb_memory_context(
            processed_mems, task=self.task, budget_tokens=budget_tokens
        )

    def _session_to_trajectory(self, session: Any) -> Optional[str]:
        """
        将 LLB Session 的 chat_history 序列化为可用于记忆构建的 trajectory 文本。

        返回:
            多行字符串，每行形如 "<role>: <content>"；若无法提取则返回 None。
        """
        if session is None:
            return None

        # 兼容属性形式与 dict 形式
        ch = getattr(session, "chat_history", None)
        if ch is None and isinstance(session, dict):
            ch = session.get("chat_history")

        # 优先处理 LLB 自带的 ChatHistory 类型（带有 get_value_str / get_value_length）。
        # 该类型禁止直接访问 .value 属性，因此不能用 hasattr(x, "value") / getattr(x, "value")。
        try:
            from src.typings import ChatHistory as _LLBChatHistory, Role as _LLBRole  # type: ignore
        except Exception:
            _LLBChatHistory = None  # type: ignore
            _LLBRole = None  # type: ignore

        if _LLBChatHistory is not None and isinstance(ch, _LLBChatHistory):
            try:
                role_dict = {}
                if _LLBRole is not None:
                    try:
                        role_dict = {
                            _LLBRole.USER: "user",
                            _LLBRole.AGENT: "assistant",
                        }
                    except Exception:
                        role_dict = {}
                # 当 role_dict 为空时，LLB 的实现仍会正常工作，只是 role 文本会保持原样。
                traj = ch.get_value_str(
                    role_dict=role_dict, start_index=None, end_index=None
                )
            except Exception:
                traj = None
            return traj or None

        # 兜底：兼容老版本或非 LLB 的结构（list / dict）。
        msgs: Optional[list[Any]] = None
        if isinstance(ch, dict):
            v = ch.get("value") or ch.get("messages")
            if isinstance(v, list):
                msgs = v
        elif isinstance(ch, list):
            msgs = ch

        if not msgs:
            return None

        lines: list[str] = []
        for m in msgs:
            if isinstance(m, dict):
                role = m.get("role") or m.get("speaker") or "unknown"
                content = m.get("content") or m.get("text") or ""
            else:
                role, content = "unknown", str(m)
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _session_to_message_list(self, session: Session) -> List[str]:
        """Convert LLB Session's chat_history to list of message strings.

        Returns:
            List of formatted messages like ['role: content', ...]
        """
        if session is None:
            return []

        ch = getattr(session, "chat_history", None)
        if ch is None and isinstance(session, dict):
            ch = session.get("chat_history")

        if not ch:
            return []

        messages = []
        for msg in ch:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            messages.append(f"{role}: {content}")

        return messages

    def _task_description_from_entry(self, entry: Dict[str, Any]) -> str:
        if self.task in ("db_bench", "os_interaction", "db", "os"):
            return entry.get("instruction", "")
        return entry.get("question", "")

    def _session_success(self, session: Session) -> bool:
        """Check if session was successful."""
        if session is None:
            return False
        outcome = getattr(session, "evaluation_record", None)
        if outcome:
            outcome = getattr(outcome, "outcome", None)
            return outcome == SessionEvaluationOutcome.CORRECT
        return False

    def _add_to_memid_pair_fifo(
        self,
        memid_pair: OrderedDict,
        key: str,
        values: List[str],
        max_capacity: int = 10000,
    ):
        """
        Add memory reference to memid_pair with FIFO eviction policy.

        Args:
            memid_pair: OrderedDict storing memory references
            key: New memory ID
            values: List of referenced memory IDs
            max_capacity: Maximum number of keys to keep (default 10000)
        """
        if key not in memid_pair:
            # Check capacity before adding
            if len(memid_pair) >= max_capacity:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(memid_pair))
                removed_value = memid_pair.pop(oldest_key)
                logger.debug(
                    f"[FIFO] Evicted oldest memid_pair entry: {oldest_key} (had {len(removed_value)} refs)"
                )

            memid_pair[key] = []

        # Extend with new values
        memid_pair[key].extend(values)

        # Move to end (mark as recently used)
        memid_pair.move_to_end(key)

        return memid_pair

    def _analyze_and_report_results(self):
        """
        Analyzes and reports the final results for both training and evaluation,
        including success rates and average steps for all phases.
        """
        if not self.results_log:
            logger.warning("No results were logged. Cannot perform analysis.")
            return

        logger.info(
            "\n" + "#" * 20 + " FULL EXPERIMENT FINISHED - FINAL RESULTS " + "#" * 20
        )
        results_df = pd.DataFrame(self.results_log)

        # --- Training Performance ---
        train_df = results_df[results_df["mode"].isin(["build", "update"])]
        if not train_df.empty:
            overall_success_rate = train_df["success"].mean()
            logger.info("\n--- Training Performance (on Train Set) ---")
            logger.info(f"Total Training Trajectories: {len(train_df)}")
            logger.info(f"Overall Success Rate: {overall_success_rate:.2%}")

            section_performance = (
                train_df.groupby("section")
                .agg(success_rate=("success", "mean"), avg_steps=("steps", "mean"))
                .reset_index()
            )
            logger.info("\n>>> Training Performance by Section <<<")
            print(
                section_performance.to_string(
                    index=False, formatters={"success_rate": "{:.2%}".format}
                )
            )

        # --- Evaluation Performance ---
        eval_df = results_df[~results_df["mode"].isin(["build", "update"])]
        if not eval_df.empty:
            logger.info("\n--- Evaluation Performance Summary ---")

            # Pivot table for Success Rate on Eval Sets
            logger.info("\n>>> Success Rate (%) by Evaluation Set <<<")
            # In eval logs, the 'success' column already holds the rate
            eval_success_summary = eval_df.pivot_table(
                index="after_section", columns="mode", values="success"
            )
            with pd.option_context("display.float_format", "{:.2%}".format):
                print(eval_success_summary)

            # Pivot table for Average Steps on Success on Eval Sets
            logger.info("\n>>> Average Steps on Success by Evaluation Set <<<")
            # In eval logs, the 'steps' column holds the average steps on success
            eval_steps_summary = eval_df.pivot_table(
                index="after_section", columns="mode", values="steps"
            )
            with pd.option_context("display.float_format", "{:.2f}".format):
                print(eval_steps_summary)

        # --- Save results to a CSV file ---
        log_dir = self.root / "logs"
        log_dir.mkdir(exist_ok=True)
        results_csv_path = (
            log_dir
            / f"experiment_results_{self.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        )
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"\nDetailed results saved to: {results_csv_path}")

    def _evaluate(
        self, eval_dataset: Dict[str, Any], eval_type: str, after_section: int
    ) -> None:
        """Run evaluation on validation or test set.

        Args:
            eval_dataset: Dictionary of evaluation samples
            eval_type: String identifier ('Validation' or 'Test')
            after_section: Current section number for logging
        """
        if not eval_dataset:
            logger.warning(f"No {eval_type} dataset available for evaluation.")
            return

        logger.info(
            f"\n--- Starting {eval_type} Evaluation (after Section {after_section}) ---"
        )

        # Get all sample keys
        eval_keys = sorted(list(eval_dataset.keys()), key=lambda x: str(x))
        logger.info(f"Evaluating on {len(eval_keys)} {eval_type.lower()} samples...")

        # Split into mini-batches
        num_mini_batches = int(np.ceil(len(eval_keys) / self.batch_size))
        eval_mini_batches = [
            eval_keys[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(num_mini_batches)
        ]

        # Sample trajectories from all mini-batches using custom dataset
        # This creates separate task_obj instances for evaluation
        eval_trajectories = []
        for mini_batch_idx, mini_batch_keys in enumerate(
            tqdm(eval_mini_batches, desc=f"{eval_type} Evaluation")
        ):
            collected_trajs = self._sample_from_indices(
                sample_indices=mini_batch_keys,
                phase="eval",
                custom_dataset=eval_dataset,
                custom_data_file=self.valid_file,
            )
            eval_trajectories.extend(collected_trajs)

        if not eval_trajectories:
            logger.warning(f"No trajectories collected during {eval_type} evaluation.")
            self.writer.add_scalar(
                f"Evaluation/Success_Rate/{eval_type}", 0.0, after_section
            )
            self.writer.add_scalar(
                f"Evaluation/Avg_Steps/{eval_type}", 0.0, after_section
            )
            return

        # Calculate metrics
        successes = sum(1 for traj in eval_trajectories if traj["success"])
        success_rate = successes / len(eval_trajectories) if eval_trajectories else 0.0
        avg_steps = np.mean([traj["steps"] for traj in eval_trajectories])

        logger.info(
            f"--- {eval_type} Evaluation Complete (after Section {after_section}) ---"
        )
        logger.info(
            f"Success Rate: {success_rate:.2%} ({successes}/{len(eval_trajectories)})"
        )
        logger.info(f"Average Steps: {avg_steps:.2f}")

        # Log to TensorBoard
        self.writer.add_scalar(
            f"Evaluation/Success_Rate/{eval_type}", success_rate, after_section
        )
        self.writer.add_scalar(
            f"Evaluation/Avg_Steps/{eval_type}", avg_steps, after_section
        )

        # Log to results
        self.results_log.append(
            {
                "section": f"eval_s{after_section}",
                "after_section": after_section,
                "mode": eval_type,
                "success": success_rate,
                "steps": avg_steps,
            }
        )

        # Log token usage
        self._log_token_usage(after_section)

    def run(self):
        """Main entry point for running LLB evaluation with RL training.

        Supports multiple algorithms through a single unified run flow:
        - 'rl': Mini-batch level update_values() only (fast feedback)
        - 'mdp': Section level update_values_chain_mdp() only (slow propagation)
        - 'rl_mdp' or 'memp': Both mini-batch and section updates (combined)

        Flow:
        1. Section loop
        2. Mini-batch loop within each section
        3. Sample mini-batch trajectories
        4. [RL] Update Q-values for retrieved memories (if 'rl' in algorithm)
        5. Add memories using add_memories()
        6. [MDP] After section: call update_values_chain_mdp() (if 'mdp' in algorithm)
        """
        logger.info("Starting LLB RL evaluation...")
        logger.info(f"Task: {self.task}")
        logger.info(f"Dataset: {self.split_file}")
        logger.info(f"Num sections: {self.num_section}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Algorithm: {self.algorithm}")
        logger.info(f"Max steps per task: {self.max_steps}")

        # Initial evaluation before training (if not resuming from checkpoint)
        if self.start_section == 0 and self.val_before_train:
            if self.valid_dataset:
                logger.info("\n" + "=" * 50)
                logger.info("Running initial validation evaluation before training...")
                logger.info("=" * 50)
                self._evaluate(self.valid_dataset, "Validation", 0)
            else:
                logger.info("Skipping initial validation (no validation dataset)")
        else:
            logger.info(
                f"Skipping initial validation (start_section={self.start_section}, val_before_train={self.val_before_train})"
            )

        # Track memory references for chain MDP
        memid_pair = OrderedDict()  # new_id -> [referenced_ids]

        # Main training loop: iterate through sections
        for section_idx in range(self.start_section, len(self.section_splits)):
            section_num = section_idx + 1
            section_keys = self.section_splits[section_idx]

            logger.info(
                "\n"
                + "#" * 20
                + f" STARTING SECTION {section_num}/{self.num_section}"
                + "#" * 20
            )
            logger.info(f"Total samples in section {section_num}: {len(section_keys)}")

            # Split section into mini-batches
            num_mini_batches = int(np.ceil(len(section_keys) / self.batch_size))
            section_mini_batches = [
                section_keys[i * self.batch_size : (i + 1) * self.batch_size]
                for i in range(num_mini_batches)
            ]

            logger.info(
                f"Split into {len(section_mini_batches)} mini-batches of size <= {self.batch_size}"
            )

            section_trajectories = []
            des_id_list = []  # For chain MDP: [(task_desc, mem_id), ...]

            # Inner loop: iterate through mini-batches
            for mini_batch_idx, mini_batch_keys in enumerate(
                tqdm(section_mini_batches, desc=f"Section {section_num}")
            ):
                logger.info(
                    f"Processing mini-batch {mini_batch_idx+1}/{len(section_mini_batches)} in section {section_num}..."
                )

                # 1. Sample trajectories for this mini-batch
                collected_trajs = self._sample_from_indices(
                    sample_indices=mini_batch_keys,
                    phase="train",
                )

                if not collected_trajs:
                    logger.warning(
                        f"No trajectories collected for mini-batch {mini_batch_idx+1}"
                    )
                    continue

                logger.info(
                    f"Mini-batch {mini_batch_idx+1} collected {len(collected_trajs)} trajectories."
                )
                section_trajectories.extend(collected_trajs)

                # 2. Extract data for memory processing (matching alfworld structure)
                task_descriptions = [
                    traj["task_description"] for traj in collected_trajs
                ]
                trajectories = [
                    traj["trajectory"] for traj in collected_trajs
                ]  # Trajectory strings
                successes = [traj["success"] for traj in collected_trajs]

                # Extract retrieved memory IDs
                retrieved_ids_list = [
                    [
                        mem["memory_id"]
                        for mem_list in traj["retrieved_mems"].values()
                        for mem in mem_list
                        if "memory_id" in mem
                    ]
                    for traj in collected_trajs
                ]

                retrieved_queries = [
                    traj["retrieved_queries"] for traj in collected_trajs
                ]

                # 3. Update Q-values for retrieved memories (immediate feedback) - only if algorithm includes 'rl'
                if "rl" in self.algorithm.lower():
                    updated_q_list = self.memory_service.update_values(
                        successes, retrieved_ids_list
                    )
                    logger.info(
                        f"[RL] Updated Q-values for mini-batch {mini_batch_idx+1}: {len(updated_q_list)} memories"
                    )
                else:
                    logger.debug(
                        f"Skipping mini-batch Q-value update (algorithm={self.algorithm})"
                    )

                # 4. Prepare metadata for new memories
                metadatas_update = [
                    {
                        "source_benchmark": f"llb_{self.task}",
                        "success": traj["success"],
                        "q_value": (
                            float(self.rl_config.q_init_pos)
                            if traj["success"]
                            else float(self.rl_config.q_init_neg)
                        ),
                        "q_visits": 0,
                        "q_updated_at": datetime.now().isoformat(),
                        "last_used_at": datetime.now().isoformat(),
                        "reward_ma": 0.0,
                    }
                    for traj in collected_trajs
                ]

                # 5. Add memories using add_memories (batch update)
                result_vis = self.memory_service.add_memories(
                    task_descriptions=task_descriptions,
                    trajectories=trajectories,
                    successes=successes,
                    retrieved_memory_queries=retrieved_queries,
                    retrieved_memory_ids_list=retrieved_ids_list,
                    metadatas=metadatas_update,
                )

                # Track memory references for chain MDP
                for i, (task_desc, mem_id) in enumerate(result_vis):
                    if mem_id:
                        des_id_list.append((task_desc, mem_id))
                        # Update memid_pair with references
                        if retrieved_ids_list[i]:
                            self._add_to_memid_pair_fifo(
                                memid_pair,
                                key=mem_id,
                                values=retrieved_ids_list[i],
                                max_capacity=10000,
                            )

                logger.info(f"Mini-batch {mini_batch_idx+1} memory update complete.")
                self._log_token_usage(section_num, mini_batch=mini_batch_idx + 1)

            # Section complete - log section-level metrics
            logger.info(
                f"Section {section_num} complete. Total {len(section_trajectories)} trajectories collected."
            )

            # Calculate and log section metrics
            if section_trajectories:
                section_success = sum(
                    1 for traj in section_trajectories if traj["success"]
                )
                section_success_rate = section_success / len(section_trajectories)
                section_avg_steps = np.mean(
                    [traj["steps"] for traj in section_trajectories]
                )

                logger.info(
                    f"Section {section_num} Training Stats: Success Rate={section_success_rate:.2%}, Avg Steps={section_avg_steps:.2f}"
                )

                # TensorBoard logging
                self.writer.add_scalar(
                    "Train/Section_Success_Rate", section_success_rate, section_num
                )
                self.writer.add_scalar(
                    "Train/Section_Avg_Steps", section_avg_steps, section_num
                )

                # Log individual results
                for traj_data in section_trajectories:
                    self.results_log.append(
                        {
                            "section": section_num,
                            "success": traj_data["success"],
                            "steps": traj_data["steps"],
                        }
                    )

            # 6. After section: update values using chain MDP - only if algorithm includes 'mdp'
            if "mdp" in self.algorithm.lower() and self.rl_config and des_id_list:
                logger.info(
                    f"[MDP] Running update_values_chain_mdp for section {section_num}..."
                )
                successes_for_chain = [
                    (
                        1.0
                        if any(
                            t["task_description"] == desc and t["success"]
                            for t in section_trajectories
                        )
                        else 0.0
                    )
                    for desc, _ in des_id_list
                ]

                self.memory_service.update_values_chain_mdp(
                    des_id_list=des_id_list,
                    memid_pair=memid_pair,
                    successes=successes_for_chain,
                )
                logger.info(
                    f"[MDP] Chain MDP update complete for section {section_num}"
                )
            else:
                logger.debug(
                    f"Skipping chain MDP update (algorithm={self.algorithm}, rl_config={'present' if self.rl_config else 'missing'}, des_id_list={'present' if des_id_list else 'empty'})"
                )

            # Save checkpoint
            ckpt_meta = self.memory_service.save_checkpoint_snapshot(
                self.ck_dir, ckpt_id=section_num
            )
            logger.info(f"Saved checkpoint: {ckpt_meta}")

            # Log token usage for section
            self._log_token_usage(section_num)
            self._check_memory_usage(f"After section {section_num}")

            # Periodic evaluation (matching alfworld pattern)
            if self.mode != "test":
                if self.valid_interval > 0 and section_num % self.valid_interval == 0:
                    if self.valid_dataset:
                        self._evaluate(self.valid_dataset, "Validation", section_num)
                    else:
                        logger.info(
                            f"Validation evaluation skipped (no validation dataset)"
                        )

        # Final analysis
        self._analyze_and_report_results()

        # Close TensorBoard writer
        self.writer.close()
        logger.info("\nTraining completed!")

    # Removed _update_memory_from_trajectories - now using add_memories() directly in run()

    def _save_checkpoint(self, section_idx: int):
        """Save memory checkpoint.

        Args:
            section_idx: Current section index
        """
        try:
            ckpt_path = self.ck_dir / f"section_{section_idx}.pkl"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_service.save_checkpoint_snapshot(str(ckpt_path))
            logger.info(f"Saved checkpoint to {ckpt_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
