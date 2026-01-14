from __future__ import annotations

import logging
import os
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import Mem0Config
from .client import Mem0Client
from .types import Experience, RetrievedMemory

logger = logging.getLogger(__name__)


class Mem0Store:
    """
    基于 Experience 抽象的高层封装，是 benchmark 使用 mem0 的主要入口。

    能力：
    - `add_experience`：将一次任务经验写入 mem0；
    - `search`：基于 query 从 mem0 中检索记忆；
    - 通过 log_callback 钩子发出结构化事件，方便上层写 JSONL 日志。
    """

    def __init__(
        self,
        cfg: Mem0Config,
        *,
        log_callback: Optional[
            callable
        ] = None,  # 回调签名：(event_name: str, payload: dict) -> None
    ) -> None:
        self.cfg = cfg
        self.client = Mem0Client(cfg)
        self._log_callback = log_callback

    def _snapshot_local_cache(self, local_cache_dir: Optional[str], snapshot_root: Path) -> None:
        if not local_cache_dir:
            return
        src = Path(local_cache_dir)
        if not src.is_dir():
            return
        dst = snapshot_root / "local_cache"
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        try:
            shutil.copytree(src, dst)
        except Exception:
            logger.exception("[mem0-store] Failed to snapshot local_cache from %s", src)

    def _restore_local_cache(self, local_cache_dir: Optional[str], snapshot_root: Path) -> None:
        if not local_cache_dir:
            return
        src = snapshot_root / "local_cache"
        if not src.is_dir():
            return
        dst = Path(local_cache_dir)
        dst.mkdir(parents=True, exist_ok=True)
        try:
            for item in src.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst / item.name)
        except Exception:
            logger.exception("[mem0-store] Failed to restore local_cache to %s", dst)

    def save_checkpoint_snapshot(
        self,
        target_ck_dir: str,
        *,
        ckpt_id: int | str,
        local_cache_dir: Optional[str] = None,
    ) -> Dict[str, object]:
        snapshot_root = Path(target_ck_dir) / "snapshot" / str(ckpt_id)
        qdrant_dst = snapshot_root / "qdrant"
        history_dst = snapshot_root / "history.db"
        snapshot_root.mkdir(parents=True, exist_ok=True)

        qdrant_src = self.client.qdrant_path
        history_src = self.client.history_path
        if qdrant_src and qdrant_src.exists():
            if qdrant_dst.exists():
                shutil.rmtree(qdrant_dst, ignore_errors=True)
            shutil.copytree(qdrant_src, qdrant_dst)
        else:
            logger.warning("[mem0-store] qdrant path not found; skip snapshot")

        if history_src and history_src.exists():
            try:
                shutil.copy2(history_src, history_dst)
            except Exception:
                logger.exception("[mem0-store] Failed to copy history db from %s", history_src)

        self._snapshot_local_cache(local_cache_dir, snapshot_root)

        meta = {
            "checkpoint_id": str(ckpt_id),
            "user_id": self.cfg.user_id,
            "qdrant_src": str(qdrant_src) if qdrant_src else None,
            "history_src": str(history_src) if history_src else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            with (snapshot_root / "snapshot_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
        except Exception:
            logger.exception("[mem0-store] Failed to write snapshot_meta.json")
        return meta

    def load_checkpoint_snapshot(
        self,
        snapshot_root: str,
        *,
        local_cache_dir: Optional[str] = None,
    ) -> None:
        try:
            self.client.close()
        except Exception:
            logger.debug("[mem0-store] Failed to close existing client before restore", exc_info=True)
        try:
            time.sleep(0.5)
        except Exception:
            pass

        try:
            base_dir = Path(local_cache_dir) if local_cache_dir else Path(snapshot_root)
            runtime_dir = base_dir / f"mem0_runtime_{os.getpid()}_{int(time.time())}"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MEM0_DIR"] = str(runtime_dir)
        except Exception:
            logger.debug("[mem0-store] Failed to reset MEM0_DIR before restore", exc_info=True)

        root = Path(snapshot_root)
        qdrant_src = root / "qdrant"
        history_src = root / "history.db"
        qdrant_dst = self.client.qdrant_path
        history_dst = self.client.history_path

        if qdrant_src.is_dir() and qdrant_dst:
            if qdrant_dst.exists():
                shutil.rmtree(qdrant_dst, ignore_errors=True)
            shutil.copytree(qdrant_src, qdrant_dst)
        else:
            logger.warning("[mem0-store] No qdrant snapshot found at %s", qdrant_src)

        if history_src.is_file() and history_dst:
            try:
                history_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(history_src, history_dst)
            except Exception:
                logger.exception("[mem0-store] Failed to restore history db from %s", history_src)

        self._restore_local_cache(local_cache_dir, root)

        # Reinitialize client to pick up restored storage
        self.client = Mem0Client(self.cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_experience(
        self,
        exp: Experience,
        *,
        infer: bool = False,
    ) -> List[RetrievedMemory]:
        """
        将 Experience 写入 mem0，作为一条完整的记忆条目。

        日志约定：
        - DEBUG 日志：记录 benchmark / task_id / phase / success 等关键字段；
        - 若提供 log_callback，则触发事件 `mem0.add_experience`：
          payload 只包含精简的元信息（不写入完整文本），便于上层写 JSONL。
        """
        # 为了避免 mem0 在 infer=False 时按 message 维度拆分成多条记忆，这裡将
        # task_text 与 trajectory 合并为一条完整的文本，只写入一条 memory。
        # 这样无论是 BigCodeBench 还是 LifelongBench，每个 Experience 对应 mem0 中
        # 都是一条“task+trajectory”合在一起的记忆，便于后续检索与分析。
        full_content = (str(exp.task_text or "") + "\n\n" + str(exp.trajectory or "")).strip()
        messages = [
            {
                "role": "assistant",  # 视为“经验/解法”的输出
                "content": full_content,
            }
        ]

        metadata: Dict[str, object] = {
            "benchmark": exp.benchmark,
            "task_id": exp.task_id,
            "phase": exp.phase,
            "success": bool(exp.success),
        }
        # Avoid accidental override of core keys from extra metadata.
        for k, v in (exp.metadata or {}).items():
            if k in metadata:
                continue
            metadata[k] = v

        logger.debug(
            "[mem0-store] add_experience benchmark=%s task_id=%s phase=%s success=%s meta_keys=%s",
            exp.benchmark,
            exp.task_id,
            exp.phase,
            exp.success,
            list((exp.metadata or {}).keys()),
        )
        if self._log_callback is not None:
            payload = {
                "event": "mem0.add_experience",
                "benchmark": exp.benchmark,
                "task_id": exp.task_id,
                "phase": exp.phase,
                "success": bool(exp.success),
                "meta": list((exp.metadata or {}).keys()),
            }
            try:
                self._log_callback("mem0.add_experience", payload)
            except Exception:
                logger.exception("[mem0-store] log_callback failed on add_experience")

        return self.client.add(messages, metadata=metadata, infer=infer)

    def search(
        self,
        query: str,
        *,
        limit: int,
        threshold: Optional[float] = None,
        extra_filters: Optional[Dict[str, object]] = None,
    ) -> List[RetrievedMemory]:
        """
        使用自然语言 query 在 mem0 中检索记忆。

        benchmark 可以通过 extra_filters 进一步限定范围（例如按 benchmark / task_id 等过滤）。
        """
        filters: Dict[str, object] = {}
        if extra_filters:
            filters.update(extra_filters)

        memories = self.client.search(
            query=query,
            limit=limit,
            threshold=threshold,
            filters=filters or None,
        )

        if self._log_callback is not None:
            truncated: List[Tuple[str, float, str]] = []
            for m in memories[: min(len(memories), 8)]:
                preview = m.memory[:120]
                truncated.append((m.id, m.score, preview))
            payload = {
                "event": "mem0.search",
                "query_preview": query[:120],
                "limit": limit,
                "threshold": threshold,
                "filters_keys": list(filters.keys()),
                "returned": len(memories),
                "top_samples": [
                    {"id": mid, "score": sc, "preview": pv} for (mid, sc, pv) in truncated
                ],
            }
            try:
                self._log_callback("mem0.search", payload)
            except Exception:
                logger.exception("[mem0-store] log_callback failed on search")

        return memories
