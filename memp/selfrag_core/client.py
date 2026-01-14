from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Optional

from .config import SelfRAGConfig
from .types import RetrievedCandidate, SelfRAGDecision, SelfRAGDocDecision

logger = logging.getLogger(__name__)


class SelfRAGClient:
    """
    Self-RAG 决策层封装。

    使用当前主 LLM（通过一个统一的 generate_fn 回调）完成：
    - 是否需要检索（should_retrieve）；
    - 对候选经验逐条打标签（relevance / support / utility / selected）。

    这里不直接依赖具体的 LLM Provider，只依赖一个“给出 messages，返回文本”的函数，
    由上层 BCB/LLB runner 提供。
    """

    def __init__(
        self,
        cfg: SelfRAGConfig,
        *,
        generate_fn: Callable[[List[Dict[str, str]]], str],
        log_callback: Optional[Callable[[str, Dict[str, object]], None]] = None,
    ) -> None:
        """
        :param cfg: SelfRAGConfig 配置对象
        :param generate_fn: 接受 OpenAI 风格 messages，返回单轮非流式文本的函数
        :param log_callback: 可选事件回调，用于写入 selfrag_events.jsonl
        """
        self.cfg = cfg
        self._generate = generate_fn
        self._log_callback = log_callback

    # ------------------------------------------------------------------
    # 决策主入口
    # ------------------------------------------------------------------
    def gate_retrieval(self, query: str) -> SelfRAGDecision:
        """
        第一阶段：仅基于当前任务 query 判断是否“值得检索”历史经验。

        注意：
        - 该阶段不会访问向量索引，不会看到任何候选 memory；
        - 返回的 SelfRAGDecision 中 docs 为空，should_retrieve 表示是否推荐检索。
        """
        messages: List[Dict[str, str]] = self._build_gate_messages(query)
        raw = ""
        error: Optional[str] = None
        decision: Optional[SelfRAGDecision] = None

        # 记录 gate 阶段的 Prompt
        if self._log_callback:
            try:
                payload: Dict[str, object] = {
                    "event": "selfrag.gate_prompt",
                    "query_preview": query[:120],
                    "messages": messages,
                }
                self._log_callback("selfrag.gate_prompt", payload)
            except Exception:
                logger.exception("[selfrag-client] log_callback failed on gate_prompt")

        for attempt in range(max(1, self.cfg.max_decision_retries or 1)):
            try:
                raw = self._generate(messages)
            except Exception as e:
                logger.exception("[selfrag-client] LLM generate failed in gate on attempt=%d", attempt + 1)
                error = str(e)
                continue

            # 记录 gate 阶段的原始输出
            if self._log_callback:
                try:
                    payload_out: Dict[str, object] = {
                        "event": "selfrag.gate_raw_output",
                        "query_preview": query[:120],
                        "raw_output": raw,
                        "attempt": attempt + 1,
                    }
                    self._log_callback("selfrag.gate_raw_output", payload_out)
                except Exception:
                    logger.exception("[selfrag-client] log_callback failed on gate_raw_output")

            try:
                obj = json.loads(self._extract_json(raw))
                should = bool(obj.get("should_retrieve", False))
                decision = SelfRAGDecision(
                    should_retrieve=should,
                    docs=[],
                    raw_output=raw,
                    error=None,
                )
                error = None
                break
            except Exception as e:
                logger.debug(
                    "[selfrag-client] Failed to parse gate decision on attempt=%d: %s",
                    attempt + 1,
                    e,
                )
                error = str(e)

        if decision is None:
            decision = SelfRAGDecision(
                should_retrieve=False,
                docs=[],
                raw_output=raw,
                error=error,
            )

        if self._log_callback:
            try:
                payload_decision: Dict[str, object] = {
                    "event": "selfrag.gate_decision",
                    "query_preview": query[:120],
                    "should_retrieve": decision.should_retrieve,
                    "error": decision.error,
                }
                self._log_callback("selfrag.gate_decision", payload_decision)
            except Exception:
                logger.exception("[selfrag-client] log_callback failed on gate_decision")

        return decision

    def decide(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
    ) -> SelfRAGDecision:
        """
        基于当前任务 query 与候选经验列表，调用主 LLM 做 self-rag 风格决策。
        这里会额外通过 log_callback 记录 selfrag 自身的「决策 Prompt」与「原始输出」，便于调试。
        """
        if not candidates:
            return SelfRAGDecision(should_retrieve=False, docs=[], raw_output="")

        mode = (self.cfg.retrieval_mode or "adaptive").strip().lower()
        if mode == "never":
            return SelfRAGDecision(should_retrieve=False, docs=[], raw_output="")
        if mode == "always":
            docs = [
                SelfRAGDocDecision(
                    id=c.id,
                    task_id=c.metadata.get("task_id"),
                    selected=True,
                    relevance=c.score,
                    raw_labels={"mode": "always"},
                )
                for c in candidates
            ]
            decision = SelfRAGDecision(should_retrieve=True, docs=docs, raw_output="")
            if self._log_callback:
                docs_summary = [
                    {
                        "id": d.id,
                        "task_id": d.task_id,
                        "selected": d.selected,
                        "relevance": d.relevance,
                        "support": d.support,
                        "utility": d.utility,
                        "labels": {
                            "isrel": d.raw_labels.get("isrel"),
                            "issup": d.raw_labels.get("issup"),
                            "isuse": d.raw_labels.get("isuse"),
                        },
                    }
                    for d in docs
                ]
                payload = {
                    "event": "selfrag.decision",
                    "query_preview": query[:120],
                    "num_candidates": len(candidates),
                    "should_retrieve": True,
                    "selected_ids": [c.id for c in candidates],
                    "docs": docs_summary,
                    "error": None,
                }
                try:
                    self._log_callback("selfrag.decision", payload)
                except Exception:
                    logger.exception("[selfrag-client] log_callback failed on decision(always)")
            return decision

        # adaptive 模式：交给 LLM 决策（一次调用处理所有候选）
        messages = self._build_messages(query, candidates)
        raw = ""
        error: Optional[str] = None
        decision: Optional[SelfRAGDecision] = None

        # 额外日志：记录 self-rag 决策阶段的完整 Prompt（system + user），方便排查问题
        if self._log_callback:
            try:
                prompt_payload: Dict[str, object] = {
                    "event": "selfrag.decision_prompt",
                    "query_preview": query[:120],
                    "num_candidates": len(candidates),
                    # 为了方便分析，这里直接记录 messages（已在 _build_messages 中做了长度截断）
                    "messages": messages,
                }
                self._log_callback("selfrag.decision_prompt", prompt_payload)
            except Exception:
                logger.exception("[selfrag-client] log_callback failed on decision_prompt")

        for attempt in range(max(1, self.cfg.max_decision_retries or 1)):
            try:
                raw = self._generate(messages)
            except Exception as e:
                logger.exception("[selfrag-client] LLM generate failed on attempt=%d", attempt + 1)
                error = str(e)
                continue

            # 额外日志：记录 self-rag 决策阶段 LLM 的原始输出，便于你重放和人工检查
            if self._log_callback:
                try:
                    out_payload: Dict[str, object] = {
                        "event": "selfrag.decision_raw_output",
                        "query_preview": query[:120],
                        # 原始输出一般为一段 JSON，这里完整保留；如后续过长可再加截断。
                        "raw_output": raw,
                        "attempt": attempt + 1,
                    }
                    self._log_callback("selfrag.decision_raw_output", out_payload)
                except Exception:
                    logger.exception("[selfrag-client] log_callback failed on decision_raw_output")

            try:
                decision = self._parse_decision(raw, candidates)
                error = None
                break
            except Exception as e:
                logger.debug(
                    "[selfrag-client] Failed to parse decision on attempt=%d: %s",
                    attempt + 1,
                    e,
                )
                error = str(e)

        if decision is None:
            # 解析多次失败：可选回退为简单 Top-K 全选
            if self.cfg.fallback_on_error:
                logger.warning(
                    "[selfrag-client] Fallback to simple top-k selection due to parse errors: %s",
                    error,
                )
                docs = [
                    SelfRAGDocDecision(
                        id=c.id,
                        task_id=c.metadata.get("task_id"),
                        selected=True,
                        relevance=c.score,
                        raw_labels={"mode": "fallback_topk"},
                    )
                    for c in candidates
                ]
                decision = SelfRAGDecision(
                    should_retrieve=True,
                    docs=docs,
                    raw_output=raw,
                    error=error,
                )
            else:
                decision = SelfRAGDecision(
                    should_retrieve=False,
                    docs=[],
                    raw_output=raw,
                    error=error,
                )

        if self._log_callback:
            docs_summary = [
                {
                    "id": d.id,
                    "task_id": d.task_id,
                    "selected": d.selected,
                    "relevance": d.relevance,
                    "support": d.support,
                    "utility": d.utility,
                    "labels": {
                        "isrel": d.raw_labels.get("isrel"),
                        "issup": d.raw_labels.get("issup"),
                        "isuse": d.raw_labels.get("isuse"),
                    },
                }
                for d in decision.docs
            ]
            payload = {
                "event": "selfrag.decision",
                "query_preview": query[:120],
                "num_candidates": len(candidates),
                "should_retrieve": decision.should_retrieve,
                "selected_ids": [d.id for d in decision.docs if d.selected],
                "docs": docs_summary,
                "error": decision.error,
            }
            try:
                self._log_callback("selfrag.decision", payload)
            except Exception:
                logger.exception("[selfrag-client] log_callback failed on decision")

        # 将原始输出记入 decision，便于上层写日志
        decision.raw_output = decision.raw_output or raw
        return decision

    # ------------------------------------------------------------------
    # Prompt 构造与解析
    # ------------------------------------------------------------------
    def _build_gate_messages(self, query: str) -> List[Dict[str, str]]:
        """
        第一阶段 gate：仅根据当前任务，判断是否需要检索记忆。
        """
        sys_prompt = (
            "You are a retrieval gate for a code/SQL/agent task. Your job is to decide whether "
            "retrieving past experiences from a memory index is likely to help solve the current task.\n\n"
            "Output a JSON object with the following fields:\n"
            "{\n"
            "  \"should_retrieve\": true/false,\n"
            "  \"reason\": \"short English explanation\"\n"
            "}\n"
            "- should_retrieve=true: retrieving and injecting past experiences is likely to help.\n"
            "- should_retrieve=false: the task is simple enough, or past experiences are unlikely to help.\n"
            "- Do not output anything except this one JSON object.\n"
        )

        user_prompt = "Current task:\n" + query.strip()

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_json(self, raw: str) -> str:
        """
        从 LLM 输出中提取 JSON 子串（用于 gate 和 candidate 决策）。
        """
        raw = raw.strip()
        if raw.startswith("{"):
            return raw
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return raw[start:end]
        except ValueError:
            raise ValueError("LLM output does not contain a JSON object")

    def _build_messages(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
    ) -> List[Dict[str, str]]:
        """
        构造用于 self-rag 决策的 messages。

        约定：
        - system 提示中说明标签语义与 JSON 输出格式；
        - user 部分给出当前任务 + N 条候选经验摘要。
        """
        sys_prompt = (
            "You are an assistant that decides which past experiences should be injected as extra "
            "context for solving the current task.\n"
            "You will receive the current task description and a list of candidate experiences. "
            "Each candidate contains a task description and its execution trajectory (e.g., code, "
            "SQL, or action steps).\n\n"
            "Your goals (Self-RAG style):\n"
            "1. Decide whether retrieval is needed for this task "
            "(analogous to [Retrieval] vs [No Retrieval]).\n"
            "2. If retrieval is needed, rate every candidate and decide which ones to select. "
            "For each candidate, you must output:\n"
            "   - relevance: 0.0–1.0, how relevant this experience is to the current task "
            "(maps to ISREL strength);\n"
            "   - support: 0.0–1.0, how much this experience can factually/structurally support "
            "a good solution to the current task (maps to ISSUP strength);\n"
            "   - utility: integer 1–5, overall usefulness of this experience for solving the "
            "current task (maps to ISUSE, 5 = very useful, 1 = barely useful);\n"
            "   - selected: true/false, whether you recommend injecting this experience into "
            "the model's memory context.\n\n"
            "Output requirements:\n"
            "- Return exactly one JSON object, with no extra explanation or commentary.\n"
            "- JSON schema:\n"
            "{\n"
            "  \"should_retrieve\": true/false,\n"
            "  \"docs\": [\n"
            "    {\n"
            "      \"id\": \"candidate id string\",\n"
            "      \"task_id\": \"optional original task id\",\n"
            "      \"selected\": true/false,\n"
            "      \"relevance\": 0.0–1.0,\n"
            "      \"support\": 0.0–1.0,\n"
            "      \"utility\": 1–5\n"
            "    }, ...\n"
            "  ]\n"
            "}\n"
            "- The docs list must contain one entry for EVERY candidate (no dropping).\n"
        )

        # 将候选经验压缩为相对短的摘要，只保留前若干字符，避免 prompt 过长。
        # 上层可以通过 cfg.top_k 控制候选数量。
        lines: List[str] = []
        lines.append("Current task:")
        lines.append(query.strip())
        lines.append("\nCandidate experiences:")
        for idx, cand in enumerate(candidates, start=1):
            preview = cand.text.strip().replace("\n", "\\n")
            if len(preview) > 400:
                preview = preview[:400] + "..."
            lines.append(
                f"[{idx}] id={cand.id}, task_id={cand.metadata.get('task_id')}, score={cand.score:.3f}\n"
                f"{preview}"
            )

        user_prompt = "\n\n".join(lines)

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_decision(
        self,
        raw: str,
        candidates: List[RetrievedCandidate],
    ) -> SelfRAGDecision:
        """
        解析 LLM 返回的 JSON 文本，构造 SelfRAGDecision。
        """
        raw = self._extract_json(raw)
        obj = json.loads(raw)
        should_retrieve = bool(obj.get("should_retrieve", False))

        docs: List[SelfRAGDocDecision] = []
        docs_raw = obj.get("docs") or []

        # 构建一个 map，方便按 id 合并候选与决策信息
        cand_by_id: Dict[str, RetrievedCandidate] = {c.id: c for c in candidates}

        for item in docs_raw:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("id", "") or "")
            if not doc_id:
                continue
            cand = cand_by_id.get(doc_id)
            task_id = item.get("task_id")
            if cand and not task_id:
                task_id = cand.metadata.get("task_id")
            selected = bool(item.get("selected", False))
            relevance = item.get("relevance")
            support = item.get("support")
            utility = item.get("utility")
            try:
                if utility is not None:
                    utility = int(utility)
            except Exception:
                utility = None

            docs.append(
                SelfRAGDocDecision(
                    id=doc_id,
                    task_id=str(task_id) if task_id is not None else None,
                    selected=selected,
                    relevance=float(relevance) if isinstance(relevance, (int, float)) else None,
                    support=float(support) if isinstance(support, (int, float)) else None,
                    utility=utility,
                    raw_labels=dict(item),
                )
            )

        return SelfRAGDecision(
            should_retrieve=should_retrieve,
            docs=docs,
            raw_output=raw,
        )
