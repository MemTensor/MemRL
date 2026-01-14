"""
Retrievers for different retrieve strategies in the Memp system.

This module provides a strategy-pattern implementation for retrieval logic,
mirroring builders.py for build strategies. It centralizes:
- Flattening MemOS search/get_all results
- Formatting each memory item into a consistent dict
- Concrete retrievers for RANDOM, QUERY, AVEFACT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import random

from memos.mem_os.main import MOS
from memos.memories.textual.item import TextualMemoryItem

from .strategies import RetrieveStrategy
from ..providers.embedding import AverageEmbedder



# ---------- Utilities ----------

def _flatten_text_mem_results(result: Dict[str, Any]) -> List[TextualMemoryItem]:
    """Flatten MemOS MOSSearchResult text_mem section into a plain list of items.

    Expected input format (from MOS.search/get_all):
    {
        "text_mem": [
            {"cube_id": "...", "memories": [TextualMemoryItem, ...]},
            ...
        ],
        "act_mem": [...],
        "para_mem": [...],
    }
    """
    items: List[TextualMemoryItem] = []
    for cube in result.get("text_mem", []):
        items.extend(cube.get("memories", []))
    return items


def _format_memory_result(item: Any) -> Dict[str, Any]:
    """Format a TextualMemoryItem (or compatible dict) to a consistent dict.

    Returns keys:
    - memory_id: str
    - content: str (prefer metadata.full_content; fallback to item.memory)
    - metadata: Any
    - similarity: float (uses metadata.relativity if present)
    - memory_item: original object
    """
    if hasattr(item['item'], "memory"):
        metadata = getattr(item['item'], "metadata", None)
        similarity = item['score']
        full_content = None
        try:
            if metadata is not None and hasattr(metadata, "model_dump"):
                md = metadata.model_dump()
                full_content = md.get("full_content")
            else:
                full_content = getattr(metadata, "full_content", None)
        except Exception:
            full_content = None
        content = full_content or getattr(item['item'], "memory", "")
        return {
            "memory_id": getattr(item['item'], "id", "unknown"),
            "content": content,
            "metadata": metadata,
            "similarity": similarity,
            "memory_item": item['item'],
        }
    # Fallback for dict-like items
    m = item if isinstance(item, dict) else {}
    md = m.get("metadata", {})
    if isinstance(md, dict):
        full_content = md.get("full_content")
    else:
        full_content = None
    return {
        "memory_id": m.get("id", "unknown"),
        "content": full_content or m.get("memory", str(item)),
        "metadata": md,
        "similarity": m['score'],
        "memory_item": item,
    }


# ---------- Strategy Base Class ----------

class BaseRetriever(ABC):
    def __init__(self, mos: MOS, user_id: str):
        self.mos = mos
        self.user_id = user_id

    @abstractmethod
    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        ...


# ---------- Concrete Retrievers ----------

class RandomRetriever(BaseRetriever):
    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        all_res = self.mos.get_all(user_id=self.user_id)
        items = _flatten_text_mem_results(all_res)
        if not items:
            return []
        sel = random.sample(items, min(k, len(items)))
        return [_format_memory_result(x) for x in sel]


class QueryRetriever(BaseRetriever):
    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        res = self.mos.search_score(query=task_description, user_id=self.user_id, top_k=k)
        items = _flatten_text_mem_results(res)
        if threshold > 0:
            items = [x for x in items if x['score'] >= threshold]
        return [_format_memory_result(x) for x in items[:k]]


class AveFactRetriever(BaseRetriever):
    def __init__(
        self,
        mos: MOS,
        user_id: str,
        llm: Any,
        keyer: Any,
        max_keywords: int = 8,
        embedder: Any | None = None,
    ):
        super().__init__(mos, user_id)
        self.llm = llm
        self.keyer = keyer
        self.max_keywords = max_keywords
        self.embedder = embedder

    # Build query vector from keywords (or fallback to full text)
    def _build_query_vector(self, text: str) -> List[float] | None:
        if self.embedder is None:
            return None
        try:
            keywords = self.llm.extract_keywords(text, self.max_keywords)
        except Exception:
            keywords = None
        try:
            if keywords:
                vecs = self.embedder.embed(keywords)
                if not vecs:
                    return None
                return (
                    AverageEmbedder.average_embeddings(vecs)
                    if len(vecs) > 1
                    else vecs[0]
                )
            # fallback to embedding the full text
            single = getattr(self.embedder, "embed_single", None)
            if callable(single):
                return single(text)
            vecs = self.embedder.embed([text])
            return vecs[0] if vecs else None
        except Exception:
            # any embedding failure -> no vector path
            return None

    # Directly query MemOS vector DBs (general_text backend) with a query vector
    def _search_by_vector(self, qv: List[float], k: int) -> List[Any]:
        results: List[Any] = []
        try:
            # Get accessible cubes for the user
            cubes = self.mos.user_manager.get_user_cubes(self.user_id)
            for cube in cubes:
                cube_id = getattr(cube, "cube_id", None)
                if cube_id is None or cube_id not in getattr(self.mos, "mem_cubes", {}):
                    continue
                mem_cube = self.mos.mem_cubes[cube_id]
                text_mem = getattr(mem_cube, "text_mem", None)
                if text_mem is None:
                    continue
                backend = getattr(getattr(text_mem, "config", None), "backend", None)
                # only vector-search for general_text backend which exposes vector_db
                if backend == "general_text" and hasattr(text_mem, "vector_db"):
                    try:
                        vec_hits = text_mem.vector_db.search(qv, k)
                        results.extend(vec_hits)
                    except Exception:
                        # continue other cubes even if one fails
                        continue
        except Exception:
            return []
        # higher score = more similar (qdrant cosine/dot)
        try:
            results.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
        except Exception:
            pass
        return results[:k]

    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        # Try vector path first
        qv = self._build_query_vector(task_description)
        if qv is not None:
            vec_hits = self._search_by_vector(qv, max(k * 2, k))  # broaden then cut
            if vec_hits:
                # Convert payload to TextualMemoryItem and attach similarity from score
                items: List[Dict[str, Any]] = []
                for hit in vec_hits:
                    try:
                        payload = getattr(hit, "payload", {})
                        # payload is expected to match TextualMemoryItem schema
                        itm = TextualMemoryItem(**payload)
                        if threshold > 0 and getattr(hit, "score", 0.0) < threshold:
                            continue
                        fm = _format_memory_result(itm)
                        fm["similarity"] = float(getattr(hit, "score", 0.0))
                        items.append(fm)
                    except Exception:
                        continue
                if items:
                    return items[:k]
        # Fallback: keyword/text search via MOS.search
        try:
            _ = self.keyer.generate_key(task_description)
        except Exception:
            pass
        try:
            keywords = self.llm.extract_keywords(task_description, self.max_keywords)
        except Exception:
            keywords = None
        keyword_query = " ".join(keywords) if keywords else task_description
        res = self.mos.search(query=keyword_query, user_id=self.user_id, top_k=k)
        items2 = _flatten_text_mem_results(res)
        if threshold > 0:
            def _sim(x: Any) -> float:
                meta = getattr(x, "metadata", None)
                return getattr(meta, "relativity", 0.0) if meta is not None else 0.0
            items2 = [x for x in items2 if _sim(x) >= threshold]
        return [_format_memory_result(x) for x in items2[:k]]


# ---------- Factory ----------

def get_retriever(
    strategy: RetrieveStrategy,
    *,
    mos: MOS,
    user_id: str,
    llm: Optional[Any] = None,
    keyer: Optional[Any] = None,
    max_keywords: int = 8,
    embedder: Optional[Any] = None,
) -> BaseRetriever:
    if strategy == RetrieveStrategy.RANDOM:
        return RandomRetriever(mos, user_id)
    if strategy == RetrieveStrategy.QUERY:
        return QueryRetriever(mos, user_id)
    # default to AVEFACT
    return AveFactRetriever(
        mos,
        user_id,
        llm=llm,
        keyer=keyer,
        max_keywords=max_keywords,
        embedder=embedder,
    )
