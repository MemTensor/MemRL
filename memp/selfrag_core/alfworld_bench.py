#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memp.configs.config import MempConfig
from memp.providers.llm import OpenAILLM
from memp.providers.embedding import OpenAIEmbedder
from memp.agent.memp_agent import MempAgent
from memp.run.alfworld_rl_runner import AlfworldRunner

from memp.selfrag_core import SelfRAGConfig, SelfRAGStore, SelfRAGClient, SelfRAGExperience

logger = logging.getLogger(__name__)


class _MetaWrapper:
    def __init__(self, meta: Dict[str, Any]) -> None:
        self.model_extra = meta


def _build_log_callback(log_path: Optional[Path]) -> Callable[[str, Dict[str, object]], None]:
    def _log(event: str, payload: Dict[str, object]) -> None:
        if not log_path:
            return
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **payload}
        try:
            text = json.dumps(entry, ensure_ascii=False, default=str)
        except Exception:
            text = json.dumps({"event": event, "payload": str(payload)}, ensure_ascii=False)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    return _log


class SelfRAGMemoryAdapter:
    """Adapt SelfRAGStore/SelfRAGClient to the MemoryService interface AlfworldRunner expects."""

    def __init__(
        self,
        store: SelfRAGStore,
        client: SelfRAGClient,
        *,
        log_callback: Optional[Callable[[str, Dict[str, object]], None]] = None,
    ) -> None:
        self.store = store
        self.client = client
        self._log_callback = log_callback

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        if not self._log_callback:
            return
        try:
            self._log_callback(event, payload)
        except Exception:
            logger.exception("[selfrag-alfworld] log_callback failed")

    def retrieve_query(self, task_description: str, k: int = 5, threshold: float = 0.0):
        try:
            candidates = self.store.search(task_description, top_k=k)
            if threshold is not None:
                candidates = [c for c in candidates if c.score >= float(threshold)]

            if not candidates:
                result = {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}
                return result, []

            decision = self.client.decide(task_description, candidates)
            selected_ids = {d.id for d in decision.docs if d.selected}
            if not decision.should_retrieve:
                selected_ids = set()

            wrapped_all = []
            for c in candidates:
                md = c.metadata or {}
                wrapped_all.append(
                    {
                        "memory_id": c.id,
                        "content": c.text,
                        "metadata": _MetaWrapper(md),
                        "score": c.score,
                    }
                )

            wrapped_selected = [m for m in wrapped_all if m["memory_id"] in selected_ids]
            result = {
                "actions": [m["memory_id"] for m in wrapped_selected],
                "selected": wrapped_selected,
                "candidates": wrapped_all,
                "simmax": max([m["score"] for m in wrapped_all], default=0.0),
            }
            topk_queries = [(m["memory_id"], m["score"]) for m in wrapped_all]

            self._log(
                "selfrag.retrieve_query",
                {
                    "query_preview": task_description[:80],
                    "returned": len(wrapped_selected),
                    "candidates": len(wrapped_all),
                    "should_retrieve": decision.should_retrieve,
                },
            )
            return result, topk_queries
        except Exception:
            result = {
                "actions": [],
                "selected": [],
                "candidates": [],
                "simmax": 0.0,
            }            
            return result, []

    def add_memories(
        self,
        task_descriptions: List[str],
        trajectories: List[Any],
        successes: List[bool],
        retrieved_memory_queries: Optional[List[Any]] = None,
        retrieved_memory_ids_list: Optional[List[Any]] = None,
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
    ):
        metadatas = metadatas or [{} for _ in task_descriptions]
        for td, traj, succ, meta in zip(task_descriptions, trajectories, successes, metadatas):
            try:
                traj_text = json.dumps(traj, ensure_ascii=False, default=str)
                exp = SelfRAGExperience(
                    benchmark="alfworld",
                    task_id=str(meta.get("task_id") or meta.get("gamefile") or ""),
                    phase=str(meta.get("phase") or "train"),
                    success=bool(succ),
                    task_text=f"TASK:\n{td}",
                    trajectory=f"TRAJECTORY:\n{traj_text}",
                    metadata=meta or {},
                )
                self.store.add_experience(exp)
            except Exception:
                self._log("selfrag.add_experience", {"error": Exception})
        self._log("selfrag.add_experience", {"count": len(task_descriptions)})
        return {}

    def update_values(self, *args, **kwargs):
        return []

    def save_checkpoint_snapshot(self, *args, **kwargs):
        return self.store.save_checkpoint_snapshot(*args, **kwargs)

    def load_checkpoint_snapshot(self, *args, **kwargs):
        return self.store.load_checkpoint_snapshot(*args, **kwargs)


def setup_logging(name: str) -> Path:
    log_dir = PROJECT_ROOT / "logs" / name
    log_dir.mkdir(exist_ok=True)
    log_filename = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info("Logging configured. Log file: %s", log_filepath)
    return log_dir



def main() -> None:
    parser = argparse.ArgumentParser(description="Run Alfworld with Self-RAG baseline")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "selfrag_alf_config.yaml"))
    args = parser.parse_args()

    cfg = MempConfig.from_yaml(args.config)
    log_dir = setup_logging(cfg.experiment.experiment_name)
    token_log_dir = log_dir / "local_cache"
    token_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"selfrag_events_{cfg.experiment.experiment_name}.jsonl"
    log_callback = _build_log_callback(log_path)

    llm_provider = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_temperature=cfg.llm.temperature,
        default_max_tokens=cfg.llm.max_tokens,
        token_log_dir=token_log_dir,
    )
    with open(PROJECT_ROOT / cfg.experiment.few_shot_path, "r", encoding="utf-8") as f:
        few_shot_examples = json.load(f)
    agent = MempAgent(llm_provider=llm_provider, few_shot_examples=few_shot_examples)

    embedder = OpenAIEmbedder(
        api_key=cfg.embedding.api_key,
        base_url=cfg.embedding.base_url,
        model=cfg.embedding.model,
        token_log_dir=token_log_dir,
    )

    resume_user_id = None
    resume_path = getattr(cfg.experiment, "ckpt_resume_path", None)
    resume_epoch = getattr(cfg.experiment, "ckpt_resume_epoch", None)
    resume_enabled = bool(getattr(cfg.experiment, "ckpt_resume_enabled", False))
    if resume_enabled and resume_path:
        resume_root = Path(resume_path)
        if resume_epoch is not None:
            if resume_root.name == "snapshot":
                resume_root = resume_root / str(resume_epoch)
            elif (resume_root / "snapshot").exists():
                resume_root = resume_root / "snapshot" / str(resume_epoch)
        meta_path = resume_root / "snapshot_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                resume_user_id = meta.get("user_id")
            except Exception:
                logger.debug("Failed to read snapshot_meta.json for resume user_id", exc_info=True)

    selfrag_user_id = resume_user_id or f"alfworld_selfrag_{cfg.experiment.experiment_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    selfrag_cfg = SelfRAGConfig(
        user_id=selfrag_user_id,
        top_k=cfg.memory.k_retrieve,
        retrieval_mode="adaptive",
        embedding_cfg={
            "model": cfg.embedding.model,
            "base_url": cfg.embedding.base_url or "",
        },
    )
    selfrag_store = SelfRAGStore(selfrag_cfg, embed_fn=embedder.embed, log_callback=log_callback)

    def _generate(messages: List[Dict[str, str]]) -> str:
        return llm_provider.generate(messages)

    selfrag_client = SelfRAGClient(selfrag_cfg, generate_fn=_generate, log_callback=log_callback)
    memory_adapter = SelfRAGMemoryAdapter(selfrag_store, selfrag_client, log_callback=log_callback)

    alfworld_config_path = PROJECT_ROOT / "configs" / "envs" / "alfworld.yaml"
    runner = AlfworldRunner(
        agent=agent,
        root=PROJECT_ROOT,
        env_config=alfworld_config_path,
        memory_service=memory_adapter,
        exp_name=cfg.experiment.experiment_name,
        random_seed=cfg.experiment.random_seed,
        num_section=cfg.experiment.num_sections,
        batch_size=cfg.experiment.batch_size,
        max_steps=cfg.experiment.max_steps,
        rl_config=cfg.rl_config,
        bon=cfg.experiment.bon,
        retrieve_k=cfg.memory.k_retrieve,
        mode=cfg.experiment.mode,
        valid_interval=cfg.experiment.valid_interval,
        test_interval=cfg.experiment.test_interval,
        train_set_ratio=getattr(cfg.experiment, "train_set_ratio", 1.0),
        ckpt_resume_enabled=getattr(cfg.experiment, "ckpt_resume_enabled", False),
        ckpt_resume_path=getattr(cfg.experiment, "ckpt_resume_path", None),
        ckpt_resume_epoch=getattr(cfg.experiment, "ckpt_resume_epoch", None),
        baseline_mode=getattr(cfg.experiment, "baseline_mode", None),
        baseline_k=getattr(cfg.experiment, "baseline_k", 10),
    )
    runner.run()


if __name__ == "__main__":
    main()
