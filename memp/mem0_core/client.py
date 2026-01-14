from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Mem0Config
from .types import RetrievedMemory

logger = logging.getLogger(__name__)


class Mem0Client:
    """
    对 mem0 Python SDK 的简单封装。

    设计目标：
    - 屏蔽 mem0 v0.x / v1.x 在返回格式上的差异；
    - 统一 search / add 的输出结构，方便上层 benchmark 使用；
    - 提供结构化的 DEBUG 日志，便于分析 mem0 行为。
    """

    def __init__(self, cfg: Mem0Config) -> None:
        self.cfg = cfg
        self._mode = cfg.normalized_mode()
        self._mem_root: Optional[Path] = None
        self._qdrant_path: Optional[Path] = None
        self._history_path: Optional[Path] = None
        self._memory = self._init_memory()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_memory(self) -> Any:
        """
        初始化底层 mem0 Memory 对象。

        当前只支持本地开源版 Memory；未来如需托管/HTTP 客户端，可在此扩展。
        """
        if self._mode != "oss":
            logger.warning(
                "[mem0] mode=%s is not fully supported yet; falling back to OSS Memory.",
                self._mode,
            )

        # 为 mem0 内部使用的 OpenAI 客户端与本地存储准备环境变量：
        # - 若 cfg.api_key 存在且环境中尚未设置 OPENAI_API_KEY，则在此设置；
        # - 若 cfg.base_url 存在且环境中尚未设置 OPENAI_BASE_URL，则在此设置；
        # - 为避免多个进程同时使用同一个 /root/.mem0/migrations_qdrant 路径，
        #   在此为当前进程设置一个基于项目目录与 pid 的 MEM0_DIR，用于 mem0
        #   内部的迁移/遥测存储（不影响我们手动指定的主向量库路径）。
        try:
            import os
            from pathlib import Path

            if self.cfg.api_key and not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = self.cfg.api_key
            if self.cfg.base_url and not os.environ.get("OPENAI_BASE_URL"):
                os.environ["OPENAI_BASE_URL"] = self.cfg.base_url

            # 若外部尚未设置 MEM0_DIR，则为当前进程设置一个独立目录：
            # <repo_root>/.mem0/runtime_<pid>
            if not os.environ.get("MEM0_DIR"):
                try:
                    repo_root = Path(__file__).resolve().parents[2]
                    mem_root = repo_root / ".mem0"
                    runtime_dir = mem_root / f"runtime_{os.getpid()}"
                    os.makedirs(runtime_dir, exist_ok=True)
                    os.environ["MEM0_DIR"] = str(runtime_dir)
                except Exception:
                    # 无法设置 MEM0_DIR 时退回 mem0 默认行为
                    logger.debug("[mem0] 设置 MEM0_DIR 环境变量失败", exc_info=True)
        except Exception:
            logger.debug(
                "[mem0] 设置 OPENAI_API_KEY/OPENAI_BASE_URL/MEM0_DIR 环境变量失败",
                exc_info=True,
            )

        def _sync_mem0_dir() -> None:
            try:
                mem0_dir = os.environ.get("MEM0_DIR")
                if not mem0_dir:
                    return
                import sys
                import mem0  # type: ignore

                if hasattr(mem0, "mem0_dir"):
                    setattr(mem0, "mem0_dir", mem0_dir)
                for name, module in list(sys.modules.items()):
                    if name.startswith("mem0") and hasattr(module, "mem0_dir"):
                        setattr(module, "mem0_dir", mem0_dir)
            except Exception:
                logger.debug("[mem0] 同步 mem0_dir 失败", exc_info=True)

        _sync_mem0_dir()

        try:
            # 基本的 Memory 对象；MemoryConfig 等用于自定义本地存储位置
            from mem0 import Memory  # type: ignore
            try:
                # v1.x 版本中提供了标准的 MemoryConfig / VectorStoreConfig，
                # 用于精细控制向量库等后端组件；老版本没有这些定义时自动降级。
                from mem0.configs.base import MemoryConfig  # type: ignore
                from mem0.vector_stores.configs import (  # type: ignore
                    VectorStoreConfig,
                )
            except Exception:  # pragma: no cover - 兼容老版本 mem0
                MemoryConfig = None  # type: ignore[assignment]
                VectorStoreConfig = None  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover - environment specific
            raise RuntimeError(
                "mem0 Python package is not available. Please install `mem0ai` "
                "or ensure it is importable as `mem0`."
            ) from exc

        _sync_mem0_dir()

        # 调用方可以通过 extra_init_kwargs 显式覆盖所有参数；否则我们会构造一个
        # 带有“项目内稳定本地向量库路径”的默认配置，避免 mem0 默认使用 /tmp/qdrant
        # 或 $HOME/.mem0 在不同项目/用户之间互相干扰。
        init_kwargs: Dict[str, Any] = dict(self.cfg.extra_init_kwargs or {})

        if "config" not in init_kwargs:
            try:
                if MemoryConfig is not None:  # type: ignore[name-defined]
                    # 1）先构造 mem0 自带的默认 MemoryConfig
                    base_cfg = MemoryConfig()  # type: ignore[call-arg]
                    vs_cfg = base_cfg.vector_store.config

                    # 2）计算“当前项目根目录”，将向量库和 history.db 都放在项目下：
                    #    <repo_root>/.mem0/...
                    #
                    #    这样：
                    #    - 不会写入 root 用户的 HOME 目录；
                    #    - 不同项目的向量库天然隔离；
                    #    - 更方便整体打包 / 迁移实验。
                    from pathlib import Path  # 局部导入，避免全局污染
                    from os import makedirs as _makedirs

                    repo_root = Path(__file__).resolve().parents[2]
                    mem_root = repo_root / ".mem0"

                    # 针对不同 user_id 使用互相独立的本地向量库目录，避免多个进程
                    # 同时连接同一个 qdrant 路径时出现“already accessed by another instance”
                    # 的冲突；train/val 由于共用同一个 user_id，仍然会共享同一目录。
                    from hashlib import md5 as _md5

                    user_hash = _md5(self.cfg.user_id.encode("utf-8")).hexdigest()[:8]
                    qdrant_path = mem_root / "qdrant" / f"user_{user_hash}"
                    history_path = mem_root / "history.db"
                    self._mem_root = mem_root
                    self._qdrant_path = qdrant_path
                    self._history_path = history_path

                    try:
                        _makedirs(qdrant_path, exist_ok=True)
                    except Exception:
                        logger.debug(
                            "[mem0] 创建项目内向量库目录失败 path=%s", qdrant_path, exc_info=True
                        )

                    # 如果底层向量库支持本地 path，则切换到项目内的 qdrant 目录
                    if getattr(vs_cfg, "path", None):
                        try:
                            setattr(vs_cfg, "path", str(qdrant_path))
                        except Exception:
                            logger.debug(
                                "[mem0] 设置 vector_store.path 失败", exc_info=True
                            )

                    # 3）若向量库支持 on_disk 标志，则显式打开持久化模式，避免纯内存模式
                    #    在进程结束后丢失所有记忆。
                    if hasattr(vs_cfg, "on_disk"):
                        try:
                            setattr(vs_cfg, "on_disk", True)
                        except Exception:
                            logger.debug(
                                "[mem0] 设置 vector_store.on_disk 失败", exc_info=True
                            )

                    # 4）将历史数据库也放在项目内，避免写入家目录
                    try:
                        _makedirs(mem_root, exist_ok=True)
                        base_cfg.history_db_path = str(history_path)
                    except Exception:
                        logger.debug(
                            "[mem0] 设置 history_db_path 失败", exc_info=True
                        )

                    # 5）将自定义的 MemoryConfig 注入到构造参数中
                    init_kwargs["config"] = base_cfg
            except Exception:
                # 无法构造自定义配置时，打印 DEBUG 日志并退回到 mem0 默认行为
                logger.debug(
                    "[mem0] 构造自定义 MemoryConfig 失败，将使用 mem0 默认配置",
                    exc_info=True,
                )

        logger.info(
            "[mem0] Initializing Memory (mode=%s, user_id=%s, extra_init_keys=%s)",
            self._mode,
            self.cfg.user_id,
            list(init_kwargs.keys()),
        )
        memory = Memory(**init_kwargs)

        # 记录一份底层向量库配置快照到日志，便于排查“跨进程是否共享到同一个存储”的问题。
        try:
            cfg_obj = getattr(memory, "config", None)
            if cfg_obj is not None:
                vs = getattr(cfg_obj, "vector_store", None)
                vs_conf = getattr(vs, "config", None)
                vs_path = getattr(vs_conf, "path", None)
                if vs_path:
                    try:
                        self._qdrant_path = Path(vs_path)
                    except Exception:
                        pass
                snapshot = {
                    "provider": getattr(vs, "provider", None),
                    "path": getattr(vs_conf, "path", None),
                    "host": getattr(vs_conf, "host", None),
                    "port": getattr(vs_conf, "port", None),
                    "collection_name": getattr(vs_conf, "collection_name", None),
                }
                logger.info("[mem0] Memory vector_store config snapshot: %s", snapshot)
        except Exception:
            logger.debug("[mem0] 记录 Memory.config 快照失败", exc_info=True)

        return memory

    @property
    def qdrant_path(self) -> Optional[Path]:
        return self._qdrant_path

    @property
    def history_path(self) -> Optional[Path]:
        return self._history_path

    @property
    def mem_root(self) -> Optional[Path]:
        return self._mem_root

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    @property
    def user_id(self) -> str:
        return self.cfg.user_id

    def close(self) -> None:
        memory = getattr(self, "_memory", None)
        if memory is None:
            return
        for attr in ("close", "shutdown", "cleanup"):
            fn = getattr(memory, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    logger.debug("[mem0] memory.%s() failed", attr, exc_info=True)
        for attr in ("client", "_client", "vector_store", "_vector_store"):
            obj = getattr(memory, attr, None)
            if obj is None:
                continue
            fn = getattr(obj, "close", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    logger.debug("[mem0] %s.close() failed", attr, exc_info=True)
        try:
            self._memory = None
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    def add(
        self,
        messages: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> List[RetrievedMemory]:
        """
        将一段对话消息写入 mem0，返回归一化后的记忆列表。

        由于 mem0 在不同版本中 add() 的返回格式不同，这里统一转换为
        List[RetrievedMemory]，保证上层调用稳定。
        """
        md = dict(metadata or {})
        try:
            logger.debug(
                "[mem0] add() start user_id=%s infer=%s meta_keys=%s",
                self.cfg.user_id,
                infer,
                list(md.keys()),
            )
            result = self._memory.add(  # type: ignore[attr-defined]
                messages,
                user_id=self.cfg.user_id,
                metadata=md or None,
                infer=infer,
            )
        except Exception:
            logger.exception("[mem0] add() failed")
            raise

        items: List[Dict[str, Any]] = []
        if isinstance(result, dict) and "results" in result:
            items = list(result.get("results") or [])
        elif isinstance(result, list):
            items = list(result)
        else:
            logger.warning(
                "[mem0] Unexpected add() response type: %r", type(result)
            )

        memories = [self._to_retrieved_memory(it) for it in items if isinstance(it, dict)]
        logger.debug(
            "[mem0] add() done user_id=%s count=%d", self.cfg.user_id, len(memories)
        )
        return memories

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedMemory]:
        """
        对 mem0 中的记忆做语义检索。

        具体检索策略由 mem0 自身的配置决定（是否 rerank、keyword_search 等），
        此处只负责统一结果格式为 List[RetrievedMemory]。
        """
        filters = dict(filters or {})
        try:
            logger.debug(
                "[mem0] search() start user_id=%s query_preview=%r limit=%d threshold=%r filters_keys=%s",
                self.cfg.user_id,
                query[:80],
                limit,
                threshold,
                list(filters.keys()),
            )
            # OSS Memory.search signature (v1.x) is:
            #   search(query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None, threshold=None)
            raw = self._memory.search(  # type: ignore[attr-defined]
                query=query,
                user_id=self.cfg.user_id,
                limit=limit,
                filters=filters or None,
                threshold=threshold,
            )
        except Exception:
            logger.exception("[mem0] search() failed")
            raise

        items: List[Dict[str, Any]] = []
        if isinstance(raw, dict) and "results" in raw:
            items = list(raw.get("results") or [])
        elif isinstance(raw, list):
            items = list(raw)
        else:
            logger.warning(
                "[mem0] Unexpected search() response type: %r", type(raw)
            )

        memories = [self._to_retrieved_memory(it) for it in items if isinstance(it, dict)]
        logger.debug(
            "[mem0] search() done user_id=%s query_preview=%r returned=%d",
            self.cfg.user_id,
            query[:80],
            len(memories),
        )
        return memories

    # ------------------------------------------------------------------
    # 帮助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _to_retrieved_memory(item: Dict[str, Any]) -> RetrievedMemory:
        mem_id = str(item.get("id") or "")
        memory_text = str(item.get("memory") or "")
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0

        metadata = {}
        if isinstance(item.get("metadata"), dict):
            metadata = dict(item["metadata"])

        return RetrievedMemory(
            id=mem_id,
            memory=memory_text,
            score=score,
            metadata=metadata,
            user_id=item.get("user_id"),
            agent_id=item.get("agent_id"),
            run_id=item.get("run_id"),
        )
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Mem0Config
from .types import RetrievedMemory

logger = logging.getLogger(__name__)


class Mem0Client:
    """
    对 mem0 Python SDK 的简单封装。

    设计目标：
    - 屏蔽 mem0 v0.x / v1.x 在返回格式上的差异；
    - 统一 search / add 的输出结构，方便上层 benchmark 使用；
    - 提供结构化的 DEBUG 日志，便于分析 mem0 行为。
    """

    def __init__(self, cfg: Mem0Config) -> None:
        self.cfg = cfg
        self._mode = cfg.normalized_mode()
        self._mem_root: Optional[Path] = None
        self._qdrant_path: Optional[Path] = None
        self._history_path: Optional[Path] = None
        self._memory = self._init_memory()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_memory(self) -> Any:
        """
        初始化底层 mem0 Memory 对象。

        当前只支持本地开源版 Memory；未来如需托管/HTTP 客户端，可在此扩展。
        """
        if self._mode != "oss":
            logger.warning(
                "[mem0] mode=%s is not fully supported yet; falling back to OSS Memory.",
                self._mode,
            )

        # 为 mem0 内部使用的 OpenAI 客户端与本地存储准备环境变量：
        # - 若 cfg.api_key 存在且环境中尚未设置 OPENAI_API_KEY，则在此设置；
        # - 若 cfg.base_url 存在且环境中尚未设置 OPENAI_BASE_URL，则在此设置；
        # - 为避免多个进程同时使用同一个 /root/.mem0/migrations_qdrant 路径，
        #   在此为当前进程设置一个基于项目目录与 pid 的 MEM0_DIR，用于 mem0
        #   内部的迁移/遥测存储（不影响我们手动指定的主向量库路径）。
        try:
            import os
            from pathlib import Path

            if self.cfg.api_key and not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = self.cfg.api_key
            if self.cfg.base_url and not os.environ.get("OPENAI_BASE_URL"):
                os.environ["OPENAI_BASE_URL"] = self.cfg.base_url

            # 若外部尚未设置 MEM0_DIR，则为当前进程设置一个独立目录：
            # <repo_root>/.mem0/runtime_<pid>
            if not os.environ.get("MEM0_DIR"):
                try:
                    repo_root = Path(__file__).resolve().parents[2]
                    mem_root = repo_root / ".mem0"
                    runtime_dir = mem_root / f"runtime_{os.getpid()}"
                    os.makedirs(runtime_dir, exist_ok=True)
                    os.environ["MEM0_DIR"] = str(runtime_dir)
                except Exception:
                    # 无法设置 MEM0_DIR 时退回 mem0 默认行为
                    logger.debug("[mem0] 设置 MEM0_DIR 环境变量失败", exc_info=True)
        except Exception:
            logger.debug(
                "[mem0] 设置 OPENAI_API_KEY/OPENAI_BASE_URL/MEM0_DIR 环境变量失败",
                exc_info=True,
            )

        try:
            # 基本的 Memory 对象；MemoryConfig 等用于自定义本地存储位置
            from mem0 import Memory  # type: ignore
            try:
                # v1.x 版本中提供了标准的 MemoryConfig / VectorStoreConfig，
                # 用于精细控制向量库等后端组件；老版本没有这些定义时自动降级。
                from mem0.configs.base import MemoryConfig  # type: ignore
                from mem0.vector_stores.configs import (  # type: ignore
                    VectorStoreConfig,
                )
            except Exception:  # pragma: no cover - 兼容老版本 mem0
                MemoryConfig = None  # type: ignore[assignment]
                VectorStoreConfig = None  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover - environment specific
            raise RuntimeError(
                "mem0 Python package is not available. Please install `mem0ai` "
                "or ensure it is importable as `mem0`."
            ) from exc

        # 调用方可以通过 extra_init_kwargs 显式覆盖所有参数；否则我们会构造一个
        # 带有“项目内稳定本地向量库路径”的默认配置，避免 mem0 默认使用 /tmp/qdrant
        # 或 $HOME/.mem0 在不同项目/用户之间互相干扰。
        init_kwargs: Dict[str, Any] = dict(self.cfg.extra_init_kwargs or {})

        if "config" not in init_kwargs:
            try:
                if MemoryConfig is not None:  # type: ignore[name-defined]
                    # 1）先构造 mem0 自带的默认 MemoryConfig
                    base_cfg = MemoryConfig()  # type: ignore[call-arg]
                    vs_cfg = base_cfg.vector_store.config

                    # 2）计算“当前项目根目录”，将向量库和 history.db 都放在项目下：
                    #    <repo_root>/.mem0/...
                    #
                    #    这样：
                    #    - 不会写入 root 用户的 HOME 目录；
                    #    - 不同项目的向量库天然隔离；
                    #    - 更方便整体打包 / 迁移实验。
                    from pathlib import Path  # 局部导入，避免全局污染
                    from os import makedirs as _makedirs

                    repo_root = Path(__file__).resolve().parents[2]
                    mem_root = repo_root / ".mem0"

                    # 针对不同 user_id 使用互相独立的本地向量库目录，避免多个进程
                    # 同时连接同一个 qdrant 路径时出现“already accessed by another instance”
                    # 的冲突；train/val 由于共用同一个 user_id，仍然会共享同一目录。
                    from hashlib import md5 as _md5

                    user_hash = _md5(self.cfg.user_id.encode("utf-8")).hexdigest()[:8]
                    qdrant_path = mem_root / "qdrant" / f"user_{user_hash}"
                    self._mem_root = mem_root
                    self._qdrant_path = qdrant_path

                    try:
                        _makedirs(qdrant_path, exist_ok=True)
                    except Exception:
                        logger.debug(
                            "[mem0] 创建项目内向量库目录失败 path=%s", qdrant_path, exc_info=True
                        )

                    # 如果底层向量库支持本地 path，则切换到项目内的 qdrant 目录
                    if getattr(vs_cfg, "path", None):
                        try:
                            setattr(vs_cfg, "path", str(qdrant_path))
                        except Exception:
                            logger.debug(
                                "[mem0] 设置 vector_store.path 失败", exc_info=True
                            )

                    # 3）若向量库支持 on_disk 标志，则显式打开持久化模式，避免纯内存模式
                    #    在进程结束后丢失所有记忆。
                    if hasattr(vs_cfg, "on_disk"):
                        try:
                            setattr(vs_cfg, "on_disk", True)
                        except Exception:
                            logger.debug(
                                "[mem0] 设置 vector_store.on_disk 失败", exc_info=True
                            )

                    # 4）将历史数据库也放在项目内，避免写入家目录
                    try:
                        history_path = mem_root / "history.db"
                        self._history_path = history_path
                        _makedirs(mem_root, exist_ok=True)
                        base_cfg.history_db_path = str(history_path)
                    except Exception:
                        logger.debug(
                            "[mem0] 设置 history_db_path 失败", exc_info=True
                        )

                    # 5）将自定义的 MemoryConfig 注入到构造参数中
                    init_kwargs["config"] = base_cfg
            except Exception:
                # 无法构造自定义配置时，打印 DEBUG 日志并退回到 mem0 默认行为
                logger.debug(
                    "[mem0] 构造自定义 MemoryConfig 失败，将使用 mem0 默认配置",
                    exc_info=True,
                )

        logger.info(
            "[mem0] Initializing Memory (mode=%s, user_id=%s, extra_init_keys=%s)",
            self._mode,
            self.cfg.user_id,
            list(init_kwargs.keys()),
        )
        memory = Memory(**init_kwargs)

        # 记录一份底层向量库配置快照到日志，便于排查“跨进程是否共享到同一个存储”的问题。
        try:
            cfg_obj = getattr(memory, "config", None)
            if cfg_obj is not None:
                vs = getattr(cfg_obj, "vector_store", None)
                vs_conf = getattr(vs, "config", None)
                snapshot = {
                    "provider": getattr(vs, "provider", None),
                    "path": getattr(vs_conf, "path", None),
                    "host": getattr(vs_conf, "host", None),
                    "port": getattr(vs_conf, "port", None),
                    "collection_name": getattr(vs_conf, "collection_name", None),
                }
                if snapshot.get("path"):
                    try:
                        self._qdrant_path = Path(snapshot["path"])
                    except Exception:
                        pass
                history_path = getattr(cfg_obj, "history_db_path", None)
                if history_path:
                    try:
                        self._history_path = Path(history_path)
                    except Exception:
                        pass
                logger.info("[mem0] Memory vector_store config snapshot: %s", snapshot)
        except Exception:
            logger.debug("[mem0] 记录 Memory.config 快照失败", exc_info=True)

        return memory

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    @property
    def user_id(self) -> str:
        return self.cfg.user_id

    @property
    def mem_root(self) -> Optional[Path]:
        return self._mem_root

    @property
    def qdrant_path(self) -> Optional[Path]:
        return self._qdrant_path

    @property
    def history_path(self) -> Optional[Path]:
        return self._history_path

    def add(
        self,
        messages: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> List[RetrievedMemory]:
        """
        将一段对话消息写入 mem0，返回归一化后的记忆列表。

        由于 mem0 在不同版本中 add() 的返回格式不同，这里统一转换为
        List[RetrievedMemory]，保证上层调用稳定。
        """
        md = dict(metadata or {})
        try:
            logger.debug(
                "[mem0] add() start user_id=%s infer=%s meta_keys=%s",
                self.cfg.user_id,
                infer,
                list(md.keys()),
            )
            result = self._memory.add(  # type: ignore[attr-defined]
                messages,
                user_id=self.cfg.user_id,
                metadata=md or None,
                infer=infer,
            )
        except Exception:
            logger.exception("[mem0] add() failed")
            raise

        items: List[Dict[str, Any]] = []
        if isinstance(result, dict) and "results" in result:
            items = list(result.get("results") or [])
        elif isinstance(result, list):
            items = list(result)
        else:
            logger.warning(
                "[mem0] Unexpected add() response type: %r", type(result)
            )

        memories = [self._to_retrieved_memory(it) for it in items if isinstance(it, dict)]
        logger.debug(
            "[mem0] add() done user_id=%s count=%d", self.cfg.user_id, len(memories)
        )
        return memories

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedMemory]:
        """
        对 mem0 中的记忆做语义检索。

        具体检索策略由 mem0 自身的配置决定（是否 rerank、keyword_search 等），
        此处只负责统一结果格式为 List[RetrievedMemory]。
        """
        filters = dict(filters or {})
        try:
            logger.debug(
                "[mem0] search() start user_id=%s query_preview=%r limit=%d threshold=%r filters_keys=%s",
                self.cfg.user_id,
                query[:80],
                limit,
                threshold,
                list(filters.keys()),
            )
            # OSS Memory.search signature (v1.x) is:
            #   search(query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None, threshold=None)
            raw = self._memory.search(  # type: ignore[attr-defined]
                query=query,
                user_id=self.cfg.user_id,
                limit=limit,
                filters=filters or None,
                threshold=threshold,
            )
        except Exception:
            logger.exception("[mem0] search() failed")
            raise

        items: List[Dict[str, Any]] = []
        if isinstance(raw, dict) and "results" in raw:
            items = list(raw.get("results") or [])
        elif isinstance(raw, list):
            items = list(raw)
        else:
            logger.warning(
                "[mem0] Unexpected search() response type: %r", type(raw)
            )

        memories = [self._to_retrieved_memory(it) for it in items if isinstance(it, dict)]
        logger.debug(
            "[mem0] search() done user_id=%s query_preview=%r returned=%d",
            self.cfg.user_id,
            query[:80],
            len(memories),
        )
        return memories

    # ------------------------------------------------------------------
    # 帮助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _to_retrieved_memory(item: Dict[str, Any]) -> RetrievedMemory:
        mem_id = str(item.get("id") or "")
        memory_text = str(item.get("memory") or "")
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0

        metadata = {}
        if isinstance(item.get("metadata"), dict):
            metadata = dict(item["metadata"])

        return RetrievedMemory(
            id=mem_id,
            memory=memory_text,
            score=score,
            metadata=metadata,
            user_id=item.get("user_id"),
            agent_id=item.get("agent_id"),
            run_id=item.get("run_id"),
        )
