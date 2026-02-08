from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional
import threading

from .config import SelfRAGConfig
from .types import RetrievedCandidate, SelfRAGExperience

logger = logging.getLogger(__name__)


class SelfRAGStore:
    """
    Self-RAG 基线的候选经验存储与向量检索封装。

    设计目标：
    - 提供简单的 add / search 接口，屏蔽具体向量库实现；
    - 默认使用 OpenAI embedding + 本地简单索引（先实现 in-memory + JSON 持久化），
      后续如有需要可升级为 FAISS / Chroma 等；
    - 配合 SelfRAGClient，一起被上层 BCB/LLB runner 使用。
    """

    def __init__(
        self,
        cfg: SelfRAGConfig,
        *,
        embed_fn: Callable[[List[str]], List[List[float]]],
        log_callback: Optional[Callable[[str, Dict[str, object]], None]] = None,
    ) -> None:
        """
        :param cfg: SelfRAGConfig 配置对象
        :param embed_fn: 将一批文本编码为向量的函数
        :param log_callback: 可选事件回调，用于写入 selfrag_events.jsonl
        """
        self.cfg = cfg
        self._embed_fn = embed_fn
        self._log_callback = log_callback
        # 用于保护内存索引和磁盘写入的互斥锁，避免多线程并发读写时索引损坏
        self._lock = threading.Lock()

        # 简单实现：内存索引 + JSON 持久化
        self._repo_root = Path(os.getcwd())
        self._index_dir = (
            self._repo_root / (cfg.index_root or ".selfrag") / "index" / cfg.user_id
        )
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._index_dir / "index.jsonl"

        # 内存结构：List[Dict]，每条记录包含 id/text/embedding/metadata
        self._records: List[Dict[str, object]] = []
        # 仅内存缓存：query -> {"gen": int, "scores": List[float], "computed_upto": int}
        # 只存相似度分数，避免重复存储文本；gen 用于重新加载索引时整体失效。
        self._sim_cache: Dict[str, Dict[str, object]] = {}
        self._cache_generation: int = 0
        self._load_index()

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
            logger.exception("[selfrag-store] Failed to snapshot local_cache from %s", src)

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
            logger.exception("[selfrag-store] Failed to restore local_cache to %s", dst)

    def save_checkpoint_snapshot(
        self,
        target_ck_dir: str,
        *,
        ckpt_id: int | str,
        local_cache_dir: Optional[str] = None,
    ) -> Dict[str, object]:
        snapshot_root = Path(target_ck_dir) / "snapshot" / str(ckpt_id)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        index_dst = snapshot_root / "selfrag_index.jsonl"

        if self._index_path.exists():
            try:
                shutil.copy2(self._index_path, index_dst)
            except Exception:
                logger.exception("[selfrag-store] Failed to copy index file to %s", index_dst)
        else:
            logger.warning("[selfrag-store] index.jsonl missing; skip snapshot")

        self._snapshot_local_cache(local_cache_dir, snapshot_root)

        meta = {
            "checkpoint_id": str(ckpt_id),
            "user_id": self.cfg.user_id,
            "index_src": str(self._index_path),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            with (snapshot_root / "snapshot_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
        except Exception:
            logger.exception("[selfrag-store] Failed to write snapshot_meta.json")
        return meta

    def load_checkpoint_snapshot(
        self,
        snapshot_root: str,
        *,
        local_cache_dir: Optional[str] = None,
    ) -> None:
        root = Path(snapshot_root)
        index_src = root / "selfrag_index.jsonl"
        if index_src.exists():
            try:
                self._index_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(index_src, self._index_path)
            except Exception:
                logger.exception("[selfrag-store] Failed to restore index from %s", index_src)
        else:
            logger.warning("[selfrag-store] No index snapshot found at %s", index_src)

        self._restore_local_cache(local_cache_dir, root)

        with self._lock:
            self._records.clear()
            self._load_index()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _load_index(self) -> None:
        """从磁盘加载已有索引（如有），失败时忽略。"""
        if not self._index_path.exists():
            return
        try:
            with self._index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    # 兼容性：忽略缺少 embedding 的记录
                    if "embedding" not in obj or "text" not in obj:
                        continue
                    self._records.append(obj)
            logger.info(
                "[selfrag-store] Loaded %d records from %s",
                len(self._records),
                self._index_path,
            )
            # 索引重载：提升缓存代次，清空缓存
            self._cache_generation += 1
            self._sim_cache.clear()
        except Exception:
            logger.exception("[selfrag-store] Failed to load index from %s", self._index_path)

    def _append_record(self, rec: Dict[str, object]) -> None:
        """将单条记录追加到内存与磁盘（带锁，保证多线程安全）。"""
        with self._lock:
            self._records.append(rec)
            try:
                with self._index_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("[selfrag-store] Failed to append record to %s", self._index_path)
            # 新增记录：不清空缓存，只保留缓存的 computed_upto，让后续查询补算增量
            # 若需要彻底失效（例如删除记录），可提升 _cache_generation 并清空 _sim_cache

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def add_experience(self, exp: SelfRAGExperience) -> str:
        """
        将一条 Experience 写入 SelfRAG 索引。

        返回内部生成的经验 id。
        """
        # 统一使用 "task+trajectory" 的文本内容，避免拆分多条 message
        full_text = (exp.task_text or "") + "\n\n" + (exp.trajectory or "")
        full_text = full_text.strip()
        if not full_text:
            # 空内容没意义，直接跳过
            return ""

        embedding = self._embed_fn([full_text])[0]
        rec_id = f"selfrag-{len(self._records)}"

        rec: Dict[str, object] = {
            "id": rec_id,
            "text": full_text,
            "embedding": embedding,
            "metadata": {
                "benchmark": exp.benchmark,
                "task_id": exp.task_id,
                "phase": exp.phase,
                "success": bool(exp.success),
                **{k: v for k, v in (exp.metadata or {}).items() if k not in {"benchmark", "task_id", "phase", "success"}},
            },
        }

        if self._log_callback:
            payload = {
                "event": "selfrag.add_experience",
                "benchmark": exp.benchmark,
                "task_id": exp.task_id,
                "phase": exp.phase,
                "success": bool(exp.success),
            }
            try:
                self._log_callback("selfrag.add_experience", payload)
            except Exception:
                logger.exception("[selfrag-store] log_callback failed on add_experience")

        self._append_record(rec)
        return rec_id

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[RetrievedCandidate]:
        """
        使用简单向量相似度在 SelfRAG 索引中检索候选经验。

        当前实现：
        - 使用内存中保存的 embedding 计算余弦相似度；
        - 支持基于 metadata 的简单等值过滤（filters）。
        """
        # 拷贝一份当前索引快照，避免在遍历列表时被其它线程 append 触发
        # "list changed size during iteration" 异常。
        with self._lock:
            records = list(self._records)
            cache_hit = False
            cache_entry = self._sim_cache.get(query)
            if cache_entry and cache_entry.get("gen") == self._cache_generation:
                scores = list(cache_entry.get("scores") or [])
                computed_upto = int(cache_entry.get("computed_upto", 0))
            else:
                scores = []
                computed_upto = 0

        if not records:
            return []

        k = top_k or self.cfg.top_k or 5

        # 生成/复用相似度结果：仅对新增记录做增量计算
        if computed_upto < len(records):
            vec = self._embed_fn([query])[0]

            def cosine(a: List[float], b: List[float]) -> float:
                import math

                if not a or not b or len(a) != len(b):
                    return 0.0
                num = sum(x * y for x, y in zip(a, b))
                da = math.sqrt(sum(x * x for x in a))
                db = math.sqrt(sum(y * y for y in b))
                if da <= 0.0 or db <= 0.0:
                    return 0.0
                return num / (da * db)

            # 保持已有 candidates，追加新计算的记录
            for rec in records[computed_upto:]:
                meta = rec.get("metadata") or {}
                emb = rec.get("embedding") or []
                score = cosine(vec, emb)
                scores.append(float(score))
            with self._lock:
                self._sim_cache[query] = {
                    "gen": self._cache_generation,
                    "computed_upto": len(records),
                    "scores": scores,
                }
        else:
            cache_hit = True

        # 根据 scores 构建候选并排序、过滤
        pair_list = []
        for rec, score in zip(records, scores):
            meta = rec.get("metadata") or {}
            pair_list.append((score, rec, meta))

        pair_list.sort(key=lambda x: x[0], reverse=True)

        out: List[RetrievedCandidate] = []
        for score, rec, meta in pair_list:
            if filters:
                ok = True
                for kf, vf in filters.items():
                    if meta.get(kf) != vf:
                        ok = False
                        break
                if not ok:
                    continue
            out.append(
                RetrievedCandidate(
                    id=str(rec.get("id")),
                    text=str(rec.get("text") or ""),
                    score=float(score),
                    metadata=dict(meta),
                )
            )
            if len(out) >= k:
                break

        if self._log_callback:
            truncated = [
                {"id": c.id, "score": c.score, "task_id": c.metadata.get("task_id")}
                for c in out[: min(len(out), 8)]
            ]
            payload = {
                "event": "selfrag.search",
                "query_preview": query[:120],
                "top_k": k,
                "filters_keys": list((filters or {}).keys()),
                "returned": len(out),
                "top_samples": truncated,
                "cache_hit": cache_hit,
            }
            try:
                self._log_callback("selfrag.search", payload)
            except Exception:
                logger.exception("[selfrag-store] log_callback failed on search")

        return out
