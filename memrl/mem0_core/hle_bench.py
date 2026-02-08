#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import sys
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
os.environ["MEM0_TELEMETRY"] = "False"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from memrl.configs.config import MempConfig
from memrl.providers.llm import OpenAILLM

from memrl.mem0_core.config import Mem0Config
from memrl.mem0_core.store import Mem0Store
from memrl.mem0_core.types import Experience, RetrievedMemory

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class HLESelection:
    dataset_path: str
    num_valid: Optional[int] = None
    num_train: Optional[int] = None
    categories: Optional[List[str]] = None
    category_ratio: Optional[float] = None


class HLEMem0Bench:
    def __init__(
        self,
        name: str,
        llm: OpenAILLM,
        llm_judge: Optional[OpenAILLM],
        selection: HLESelection,
        output_dir: Path,
        mem0_store: Mem0Store,
        run_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        retrieve_k: int = 1,
        num_sections: int = 1,
        batch_size: int = 8,
        dataset_ratio: float = 1.0,
        random_seed: int = 42,
        train_valid_split: float = 0.8,
        ckpt_eval_enabled: bool = False,
        ckpt_eval_path: Optional[str] = None,
        ckpt_resume_enabled: bool = False,
        ckpt_resume_path: Optional[str] = None,
        ckpt_resume_epoch: Optional[int] = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.llm_judge = llm_judge
        self.sel = selection
        self.output_dir = Path(output_dir)
        self.mem0_store = mem0_store
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieve_k = max(0, int(retrieve_k))
        self.num_sections = max(1, int(num_sections))
        self.batch_size = max(1, int(batch_size))
        self.dataset_ratio = float(dataset_ratio)
        self.random_seed = random_seed
        self.train_valid_split = float(train_valid_split)
        self.ckpt_eval_enabled = bool(ckpt_eval_enabled)
        self.ckpt_eval_path = str(ckpt_eval_path) if ckpt_eval_path else None
        self.ckpt_resume_enabled = bool(ckpt_resume_enabled)
        self.ckpt_resume_path = str(ckpt_resume_path) if ckpt_resume_path else None
        self.ckpt_resume_epoch = ckpt_resume_epoch

        self.run_id = run_id or time.strftime('%Y%m%d-%H%M%S')
        self.ck_dir = self.output_dir / "hle_mem0" / f"exp_{self.name}_{self.run_id}"
        self.log_dir = self.ck_dir / "local_cache"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.llm_log_path = self.log_dir / "llm_calls.jsonl"
        self.mem0_log_path = self.log_dir / "mem0_calls.jsonl"
        self.timing_log_path = self.log_dir / "timing.jsonl"
        self._log_lock = threading.Lock()
        self._timing_lock = threading.Lock()
        self._image_lock = threading.Lock()
        self._image_store: Dict[str, str] = {}
        self._image_hash_to_id: Dict[str, str] = {}
        self._image_id_counter = 0
        self._image_store_path = self.log_dir / "image_store.json"
        self._image_index_path = self.log_dir / "image_hash_index.json"
        self._cum_state_path = self.log_dir / "cum_state.json"
        self._load_image_cache()
        self._cum_correct: Dict[str, Set[str]] = {}
        self._cum_totals: Dict[str, int] = {}

        self.EXACT_ANSWER_SYSTEM_PROMPT = (
            "Your response should be in the following format:\n"
            "Explanation: {your explanation for your final answer}\n"
            "Exact Answer: {your succinct, final answer}\n"
            "Confidence: {your confidence score between 0% and 100% for your answer}"
        )

        self.MULTIPLE_CHOICE_SYSTEM_PROMPT = (
            "Your response should be in the following format:\n"
            "Explanation: {your explanation for your answer choice}\n"
            "Answer: {your chosen answer}\n"
            "Confidence: {your confidence score between 0% and 100% for your answer}"
        )

        self.JUDGE_PROMPT = (
            "Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.\n\n"
            "[question]: {question}\n\n"
            "[response]: {response}\n\n"
            "Your judgement must be in the format and criteria specified below:\n\n"
            "extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.\n\n"
            "[correct_answer]: {correct_answer}\n\n"
            "reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.\n\n"
            "correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.\n\n"
            "confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."
        )

    def _resolve_ckpt_dirs(self, ckpt_root: Path) -> List[Path]:
        if (ckpt_root / "snapshot").is_dir():
            snapshot_root = ckpt_root / "snapshot"
        else:
            snapshot_root = ckpt_root
        if not snapshot_root.is_dir():
            raise ValueError(f"ckpt root does not exist: {snapshot_root}")
        ckpts = [p for p in snapshot_root.iterdir() if p.is_dir() and p.name.isdigit()]
        ckpts.sort(key=lambda p: int(p.name))
        return ckpts

    def _resolve_resume_dir(self) -> Optional[Path]:
        if not self.ckpt_resume_enabled:
            return None
        if not self.ckpt_resume_path:
            raise ValueError("ckpt_resume_path is not set")
        if not self.ckpt_resume_epoch:
            raise ValueError("ckpt_resume_epoch is not set")
        epoch = int(self.ckpt_resume_epoch)
        if epoch < 1:
            raise ValueError("ckpt_resume_epoch must be >= 1")
        root = Path(self.ckpt_resume_path)
        if (root / "snapshot").is_dir():
            candidate = root / "snapshot" / str(epoch)
        elif (root / str(epoch)).is_dir():
            candidate = root / str(epoch)
        else:
            candidate = root
        if not candidate.exists():
            raise ValueError(f"Resume snapshot not found: {candidate}")
        return candidate

    def _resume_from_ckpt(self) -> int:
        resume_dir = self._resolve_resume_dir()
        if not resume_dir:
            return 1
        logger.info("Resuming from ckpt: %s", resume_dir)
        self.mem0_store.load_checkpoint_snapshot(str(resume_dir), local_cache_dir=str(self.log_dir))
        self._load_image_cache()
        self._load_cum_state()
        return int(self.ckpt_resume_epoch) + 1

    def _eval_ckpt_sequence(self, valid_df: pd.DataFrame) -> None:
        if not self.mem0_store:
            raise RuntimeError("mem0_store is required for ckpt evaluation")
        if not self.ckpt_eval_path:
            raise ValueError("ckpt_eval_path is not set")
        ckpt_root = Path(self.ckpt_eval_path)
        ckpt_dirs = self._resolve_ckpt_dirs(ckpt_root)
        if not ckpt_dirs:
            raise ValueError(f"No checkpoint folders found under {ckpt_root}")
        for idx, ckpt_dir in enumerate(ckpt_dirs, start=1):
            logger.info("Loading ckpt %s (%d/%d) for eval", ckpt_dir, idx, len(ckpt_dirs))
            self.mem0_store.load_checkpoint_snapshot(str(ckpt_dir), local_cache_dir=str(self.log_dir))
            self._load_image_cache()
            self._eval_split(valid_df, tag=f"valid_ckpt_{idx}", step=idx, cum_key="valid", total=len(valid_df))

    def _save_ckpt(self, sec_idx: int) -> None:
        try:
            self._persist_cum_state()
            meta = self.mem0_store.save_checkpoint_snapshot(
                str(self.ck_dir),
                ckpt_id=sec_idx,
                local_cache_dir=str(self.log_dir),
            )
            logger.info("Saved ckpt: %s", meta)
        except Exception:
            logger.warning("Failed to save ckpt for section %d", sec_idx, exc_info=True)

    def _persist_cum_state(self) -> None:
        payload = {
            "cum_correct": {k: sorted(list(v)) for k, v in self._cum_correct.items()},
            "cum_totals": dict(self._cum_totals),
        }
        try:
            with open(self._cum_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            logger.debug("Failed to persist cumulative state", exc_info=True)

    def _load_cum_state(self) -> None:
        if not self._cum_state_path.exists():
            return
        try:
            with open(self._cum_state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            raw_correct = payload.get("cum_correct", {})
            self._cum_correct = {k: set(v or []) for k, v in raw_correct.items()}
            raw_totals = payload.get("cum_totals", {})
            if isinstance(raw_totals, dict):
                self._cum_totals.update({k: int(v) for k, v in raw_totals.items()})
        except Exception:
            logger.debug("Failed to load cumulative state", exc_info=True)

    def _acc_key(self, rec: Dict[str, Any]) -> str:
        rid = rec.get("id")
        if rid is not None and str(rid).strip():
            return str(rid)
        return str(rec.get("question", "")).strip()

    def _update_cumulative(self, split_key: str, results: List[Dict[str, Any]], total: int) -> Tuple[float, int]:
        seen = self._cum_correct.setdefault(split_key, set())
        for r in results:
            if r.get("correct"):
                key = self._acc_key(r)
                if key:
                    seen.add(key)
        cum_acc = len(seen) / max(1, total)
        return cum_acc, len(seen)

    def _log_mem0_event(self, event: str, payload: Dict[str, Any]) -> None:
        entry = {"ts": time.strftime('%Y-%m-%dT%H:%M:%S'), "event": event, **payload}
        try:
            text = json.dumps(entry, ensure_ascii=False, default=str)
        except Exception:
            text = json.dumps({"ts": entry.get("ts"), "event": event, "payload": str(payload)}, ensure_ascii=False)
        with self._log_lock:
            with open(self.mem0_log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    def _log_llm_call(self, call_type: str, messages: Any, response: Any, meta: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "ts": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "type": call_type,
            "meta": meta or {},
            "messages": messages,
            "response": response,
        }
        try:
            payload = json.dumps(entry, ensure_ascii=False, default=str)
        except Exception:
            entry["messages"] = str(messages)
            payload = json.dumps(entry, ensure_ascii=False, default=str)
        with self._log_lock:
            with open(self.llm_log_path, "a", encoding="utf-8") as f:
                f.write(payload + "\n")

    def _log_timing(self, event: str, elapsed: float, meta: Optional[Dict[str, Any]] = None) -> None:
        """Append lightweight timing stats to a JSONL file for profiling."""
        rec = {
            "ts": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "event": event,
            "elapsed_sec": round(float(elapsed), 3),
        }
        if meta:
            rec.update(meta)
        try:
            text = json.dumps(rec, ensure_ascii=False, default=str)
        except Exception:
            text = json.dumps({"event": event, "elapsed_sec": float(elapsed), "meta": str(meta)}, ensure_ascii=False)
        try:
            with self._timing_lock:
                with open(self.timing_log_path, "a", encoding="utf-8") as f:
                    f.write(text + "\n")
        except Exception:
            logger.debug("Failed to log timing event", exc_info=True)

    def _apply_dataset_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        ratio = getattr(self, "dataset_ratio", 1.0)
        if df is None or df.empty or not (0 < ratio < 1):
            return df
        n_keep = max(1, int(len(df) * ratio))
        if n_keep >= len(df):
            return df
        sampled = df.sample(n=n_keep, random_state=self.random_seed).reset_index(drop=True)
        logger.info("HLE reduced via dataset_ratio %.2f: %d -> %d rows", ratio, len(df), len(sampled))
        return sampled

    def _filter_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        cats = self.sel.categories
        ratio = self.sel.category_ratio
        if cats:
            if "category" not in df.columns:
                raise ValueError("HLE dataset missing 'category' column for category filtering")
            df = df[df["category"].isin(cats)].reset_index(drop=True)
            logger.info("HLE filtered categories %s -> %d rows", cats, len(df))
        if ratio is not None and 0 < ratio < 1:
            if "category" not in df.columns:
                raise ValueError("HLE dataset missing 'category' column for category ratio sampling")

            def _sample_group(g: pd.DataFrame) -> pd.DataFrame:
                n_keep = max(1, int(len(g) * ratio))
                return g.sample(n=n_keep, random_state=self.random_seed)

            df = df.groupby("category", group_keys=False).apply(_sample_group).reset_index(drop=True)
            logger.info("HLE applied category_ratio %.2f -> %d rows", ratio, len(df))
        return df

    def _load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not Path(self.sel.dataset_path).exists():
            raise ValueError(f"HLE dataset path does not exist: {self.sel.dataset_path}")
        df = pd.read_parquet(self.sel.dataset_path)
        df = self._filter_by_category(df)
        df = self._apply_dataset_ratio(df)

        if len(df) == 0:
            raise ValueError("HLE dataset is empty after category filtering/sampling")

        for c in ["id", "question", "answer"]:
            if c not in df.columns:
                raise ValueError(f"HLE dataset missing required column: {c}")

        df = df.reset_index(drop=True)
        n_total = len(df)
        split_ratio = getattr(self, "train_valid_split", 0.8)
        if "category" not in df.columns:
            raise ValueError("HLE dataset missing 'category' column for category-wise split")

        train_parts = []
        valid_parts = []
        for _, group in df.groupby("category", sort=False):
            shuffled = group.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
            n_train = int(len(shuffled) * split_ratio)
            train_parts.append(shuffled.iloc[:n_train].copy())
            valid_parts.append(shuffled.iloc[n_train:].copy())

        train = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[:0].copy()
        valid = pd.concat(valid_parts, ignore_index=True) if valid_parts else df.iloc[:0].copy()

        if self.sel.num_train:
            train = train.head(int(self.sel.num_train))
        if self.sel.num_valid:
            valid = valid.head(int(self.sel.num_valid))

        logger.info("HLE loaded: total=%d train=%d valid=%d", n_total, len(train), len(valid))
        return train.reset_index(drop=True), valid.reset_index(drop=True)

    def _collect_question_images(self, row: pd.Series) -> List[Any]:
        images: List[Any] = []
        if "image" in row.index and pd.notna(row["image"]) and str(row["image"]).strip():
            images.append(row["image"])
        return images

    def _register_image(self, image: Any) -> Optional[Tuple[str, str]]:
        if image is None:
            return None
        data_url = None
        if isinstance(image, str) and image.strip():
            data_url = image.strip()
        elif isinstance(image, dict) and "bytes" in image:
            raw = image.get("bytes")
            if isinstance(raw, bytes):
                b64 = base64.b64encode(raw).decode("utf-8")
                data_url = f"data:image/jpeg;base64,{b64}"
        if not data_url:
            return None
        key = hashlib.md5(data_url.encode("utf-8")).hexdigest()
        with self._image_lock:
            if key in self._image_hash_to_id:
                img_id = self._image_hash_to_id[key]
            else:
                self._image_id_counter += 1
                img_id = f"img_{self._image_id_counter:06d}"
                self._image_hash_to_id[key] = img_id
                self._image_store[img_id] = data_url
                self._persist_image_cache_unlocked()
        return img_id, data_url

    def _fetch_images_by_ids(self, image_ids: List[str]) -> List[Tuple[str, str]]:
        imgs: List[Tuple[str, str]] = []
        for iid in image_ids or []:
            url = self._image_store.get(str(iid))
            if url:
                imgs.append((str(iid), url))
        return imgs

    def _persist_image_cache_unlocked(self) -> None:
        try:
            with open(self._image_store_path, "w", encoding="utf-8") as f:
                json.dump(self._image_store, f, ensure_ascii=False)
            with open(self._image_index_path, "w", encoding="utf-8") as f:
                payload = {"hash_index": self._image_hash_to_id, "counter": self._image_id_counter}
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            logger.debug("Failed to persist image cache", exc_info=True)

    def _load_image_cache(self) -> None:
        try:
            if self._image_store_path.exists():
                with open(self._image_store_path, "r", encoding="utf-8") as f:
                    self._image_store = json.load(f)
            if self._image_index_path.exists():
                with open(self._image_index_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    self._image_hash_to_id = payload.get("hash_index", {})
                    self._image_id_counter = int(payload.get("counter", 0))
            if self._image_store:
                logger.info("Loaded %d images from cache", len(self._image_store))
        except Exception:
            logger.debug("Failed to load image cache", exc_info=True)

    def _extract_mem_image_ids(self, mem: RetrievedMemory) -> List[str]:
        ids = []
        md = mem.metadata or {}
        if isinstance(md, dict):
            ids = md.get("image_ids") or []
        try:
            return [str(x) for x in ids if x]
        except Exception:
            return []

    def _mem_success_flag(self, mem: RetrievedMemory) -> bool:
        md = mem.metadata or {}
        if isinstance(md, dict):
            return bool(md.get("success"))
        return False

    def _build_memory_context(
        self, selected_mems: List[RetrievedMemory], limit: int
    ) -> Tuple[str, List[str], Set[str]]:
        if not selected_mems:
            return "", [], set()
        retrieved_ids: List[str] = []
        memory_image_ids: Set[str] = set()
        succ_blocks, fail_blocks = [], []
        for m in selected_mems[: max(0, limit) or len(selected_mems)]:
            retrieved_ids.append(str(m.id))
            content = m.memory or ""
            img_ids = self._extract_mem_image_ids(m)
            if img_ids:
                memory_image_ids.update(img_ids)
                content = f"[Image IDs: {', '.join(img_ids)}]\n{content}"
            (succ_blocks if self._mem_success_flag(m) else fail_blocks).append(content)
        sections = []
        if succ_blocks:
            sections.append("=== Successful Memories ===\n" + "\n\n".join(succ_blocks))
        if fail_blocks:
            sections.append("=== Failed Memories (for caution) ===\n" + "\n\n".join(fail_blocks))
        return "\n\n".join(sections), retrieved_ids, memory_image_ids

    def _build_messages(
        self,
        question: str,
        memory_ctx: Optional[str] = None,
        answer_type: Optional[Any] = None,
        question_image_ids: Optional[List[str]] = None,
        images_info: Optional[List[Tuple[str, str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        answer_type_norm = ""
        if answer_type is not None:
            answer_type_norm = str(answer_type).strip().lower()
        system_prompt = (
            self.EXACT_ANSWER_SYSTEM_PROMPT
            if answer_type_norm == "exactmatch"
            else self.MULTIPLE_CHOICE_SYSTEM_PROMPT
        )
        legend = ""
        if images_info:
            lines = [f"{i+1}. [{img_id}] ({source})" for i, (img_id, _, source) in enumerate(images_info)]
            legend = "Attached images:\n" + "\n".join(lines)
        text_block = question if not legend else f"Now solve the following question: \n\n[Image IDs: {question_image_ids}]\n{question}\n\n{legend}"
        content: List[Dict[str, Any]] = [{"type": "text", "text": text_block}]
        if images_info:
            for img_id, url, source in images_info:
                content.append({"type": "text", "text": f"Image [{img_id}] ({source})"})
                content.append({"type": "image_url", "image_url": {"url": url}})

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if memory_ctx:
            msgs.append({"role": "system", "content": memory_ctx})
        msgs.append({"role": "user", "content": content})
        return msgs

    def _hle_judge(self, question: str, gold: str, response: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import json as _json

        prompt = self.JUDGE_PROMPT.format(question=question, correct_answer=gold, response=response)
        messages = [{"role": "user", "content": prompt}]
        judge_text = ""
        error_info = None
        try:
            judge_text = self.llm_judge.generate(messages, temperature=0.0, max_tokens=4096)
        except Exception as e:
            logger.warning("HLE judge LLM error: %s", e)
            error_info = str(e)
            judge_text = ""

        result = {
            "correct_answer": gold,
            "model_answer": None,
            "reasoning": None,
            "correct": "no",
            "confidence": 0,
            "raw_judge": judge_text,
        }
        try:
            m = re.search(r"\{[\s\S]*\}", judge_text)
            jtxt = m.group(0) if m else judge_text
            obj = _json.loads(jtxt)
            result["model_answer"] = obj.get("extracted_final_answer") or obj.get("extracted_answer")
            result["reasoning"] = obj.get("reasoning")
            corr = str(obj.get("correct", "no")).strip().lower()
            result["correct"] = "yes" if "yes" in corr else "no"
            try:
                result["confidence"] = int(obj.get("confidence", 0))
            except Exception:
                result["confidence"] = 0
        except Exception:
            try:
                m = re.search(r"extracted_final_answer\s*:\s*(.+)", judge_text, flags=re.I)
                if m:
                    result["model_answer"] = m.group(1).strip()
                m = re.search(r"correct\s*:\s*(yes|no)", judge_text, flags=re.I)
                if m:
                    result["correct"] = m.group(1).strip().lower()
                m = re.search(r"confidence\s*:\s*(\d+)", judge_text, flags=re.I)
                if m:
                    result["confidence"] = int(m.group(1))
            except Exception:
                pass
        try:
            log_meta = {"question": question, "gold": gold}
            if meta:
                log_meta.update(meta)
            if error_info:
                log_meta["error"] = error_info
            self._log_llm_call("judge", messages, judge_text, meta=log_meta)
        except Exception:
            logger.debug("Failed to log judge LLM call", exc_info=True)
        return result

    def _evaluate_row(self, row: pd.Series, *, phase: str) -> Dict[str, Any]:
        q = str(row["question"])
        gold = str(row["answer"])
        question_imgs_raw = self._collect_question_images(row)
        question_images_info: List[Tuple[str, str, str]] = []
        question_image_ids: List[str] = []
        for img in question_imgs_raw:
            reg = self._register_image(img)
            if reg:
                img_id, url = reg
                if img_id in question_image_ids:
                    continue
                question_image_ids.append(img_id)
                question_images_info.append((img_id, url, "question"))

        memory_ctx = None
        retrieved_ids: List[str] = []
        memory_image_ids: Set[str] = set()
        retrieved_topk = None
        row_start = time.time()
        timings: Dict[str, float] = {}
        if self.mem0_store and self.retrieve_k > 0:
            try:
                tau = float(getattr(self, "tau", 0.0))
            except Exception:
                tau = 0.0
            t_mem_start = time.time()
            mems = self.mem0_store.search(q, limit=self.retrieve_k, threshold=tau)
            timings["mem_search_sec"] = time.time() - t_mem_start
            retrieved_topk = [{"id": m.id, "score": m.score} for m in mems]
            t_ctx_start = time.time()
            memory_ctx, retrieved_ids, memory_image_ids = self._build_memory_context(mems, self.retrieve_k)
            timings["memory_ctx_build_sec"] = time.time() - t_ctx_start

        memory_images_info: List[Tuple[str, str, str]] = []
        if memory_image_ids:
            for img_id, url in self._fetch_images_by_ids(list(memory_image_ids)):
                memory_images_info.append((img_id, url, "memory"))

        images_info = question_images_info + memory_images_info
        answer_type = row.get("answer_type", None)
        messages = self._build_messages(
            q,
            memory_ctx=memory_ctx,
            answer_type=answer_type,
            question_image_ids=question_image_ids,
            images_info=images_info,
        )
        call_meta = {"question_id": row.get("id", None), "answer_type": answer_type, "phase": phase}
        gen_error = None
        try:
            t_llm_start = time.time()
            output = self.llm.generate(
                messages,
                temperature=self.temperature,
                # max_tokens=self.max_tokens,
            )
            timings["llm_generate_sec"] = time.time() - t_llm_start
        except Exception as e:
            logger.error("LLM error: %s", e)
            gen_error = str(e)
            output = ""
        if gen_error:
            call_meta["error"] = gen_error
        self._log_llm_call("solution", messages, output, meta=call_meta)
        t_judge_start = time.time()
        judge_res = self._hle_judge(q, gold, output or "", meta={"question_id": row.get("id", None)})
        timings["judge_sec"] = time.time() - t_judge_start
        timings["row_total_sec"] = time.time() - row_start
        try:
            self._log_timing(
                "evaluate_row",
                timings["row_total_sec"],
                {
                    "question_id": row.get("id", None),
                    "phase": phase,
                    "retrieved": len(retrieved_ids),
                    **timings,
                },
            )
        except Exception:
            logger.debug("Failed to log timing for row", exc_info=True)
        correct = True if str(judge_res.get("correct", "no")).lower() == "yes" else False

        rec: Dict[str, Any] = {
            "id": row.get("id", None),
            "question": q,
            "gold": gold,
            "raw_output": output,
            "correct": bool(correct),
            "judge_response": judge_res,
            "retrieved_ids": retrieved_ids,
            "image_ids": question_image_ids,
            "trajectory": f"QUESTION\\n{q}\\n\\nSOLUTION\\n{(output or '').strip()}\\n",
            "category": row.get("category", None),
        }
        if retrieved_topk is not None:
            rec["retrieved_topk"] = retrieved_topk
        return rec

    def _eval_split(
        self,
        df: pd.DataFrame,
        tag: str,
        step: Optional[int] = None,
        *,
        cum_key: Optional[str] = None,
        total: Optional[int] = None,
    ) -> Dict[str, float]:
        total = len(df)
        if total == 0:
            logger.warning("No rows in %s; skip.", tag)
            return {"acc": 0.0}
        results: List[Dict[str, Any]] = []
        correct_so_far = 0
        start = time.time()
        idxs = list(range(total))
        batches = [idxs[i:i + self.batch_size] for i in range(0, total, self.batch_size)]
        processed = 0
        for b in tqdm(batches, desc=f"Evaluating {tag}"):
            batch_results: List[Optional[Dict[str, Any]]] = [None] * len(b)
            with ThreadPoolExecutor(max_workers=min(len(b), self.batch_size)) as ex:
                fut2pos = {ex.submit(self._evaluate_row, df.iloc[i], phase=tag): pos for pos, i in enumerate(b)}
                for fut in as_completed(fut2pos):
                    pos = fut2pos[fut]
                    try:
                        batch_results[pos] = fut.result()
                    except Exception as e:
                        logger.warning("[%s] batch eval failed at item #%d: %s", tag, processed + pos + 1, e)
                        batch_results[pos] = None
            batch_valid = [r for r in batch_results if r is not None]
            results.extend(batch_valid)
            processed += len(batch_valid)
            correct_so_far += sum(1 for r in batch_valid if r.get("correct"))
            acc_so_far = correct_so_far / max(1, processed)
            logger.info("[%s] %d/%d | Acc so far: %.2f%%", tag, processed, total, acc_so_far * 100)

        acc = correct_so_far / max(1, len(results))
        elapsed = time.time() - start
        logger.info("[%s] Eval finished. Acc: %.2f%% | %d items | %.1fs", tag, acc * 100, total, elapsed)
        out_dir = self.output_dir / "hle_mem0"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tag))
        out_path = out_dir / f"hle_{safe_tag}_results_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        logger.info("Saved %s results to: %s", tag, out_path)
        per_item = {r["question"]: bool(r["correct"]) for r in results}
        metrics: Dict[str, float] = {"acc": float(acc), "per_item": per_item}
        if cum_key:
            total_count = total if total is not None else len(df)
            cum_acc, cum_correct = self._update_cumulative(cum_key, results, total_count)
            logger.info("[%s] Cumulative Acc: %.2f%% (%d/%d)", tag, cum_acc * 100, cum_correct, total_count)
            metrics["cum_acc"] = float(cum_acc)
        return metrics

    def _train_one_section(self, df: pd.DataFrame, sec_idx: int) -> Dict[str, float]:
        n = len(df)
        if n == 0:
            logger.info("No train data; skip training section %d", sec_idx)
            return {"acc": 0.0}
        idxs = list(range(n))
        batches = [idxs[i:i + self.batch_size] for i in range(0, n, self.batch_size)]
        all_recs: List[Dict[str, Any]] = []
        processed = 0
        correct_so_far = 0
        for b in tqdm(batches, desc=f"Training Section {sec_idx}/{self.num_sections}"):
            batch_results: List[Optional[Dict[str, Any]]] = [None] * len(b)
            with ThreadPoolExecutor(max_workers=min(len(b), self.batch_size)) as ex:
                fut2pos = {ex.submit(self._evaluate_row, df.iloc[i], phase="train"): pos for pos, i in enumerate(b)}
                for fut in as_completed(fut2pos):
                    pos = fut2pos[fut]
                    try:
                        batch_results[pos] = fut.result()
                    except Exception as e:
                        logger.warning("[train sec %d] batch eval failed at local pos %d: %s", sec_idx, pos, e)
                        batch_results[pos] = None
            batch_recs = [r for r in batch_results if r is not None]
            all_recs.extend(batch_recs)
            processed += len(batch_recs)
            correct_so_far += sum(1 for r in batch_recs if r.get("correct"))
            acc_so_far = correct_so_far / max(1, processed)
            logger.info("[train sec %d] %d/%d | Acc so far: %.2f%%", sec_idx, processed, n, acc_so_far * 100)

            if batch_recs:
                for rec in batch_recs:
                    exp = Experience(
                        benchmark="HLE",
                        task_id=str(rec.get("id")),
                        phase="train",
                        success=bool(rec.get("correct")),
                        task_text=rec.get("question", ""),
                        trajectory=rec.get("trajectory", ""),
                        metadata={
                            "category": rec.get("category"),
                            "image_ids": rec.get("image_ids", []),
                        },
                    )
                    try:
                        self.mem0_store.add_experience(exp, infer=False)
                    except Exception as e:
                        logger.warning("[train sec %d] mem0 add failed: %s", sec_idx, e)

        if not all_recs:
            return {"acc": 0.0}
        acc = correct_so_far / len(all_recs)
        logger.info("Section %d Train Acc: %.2f%%", sec_idx, acc * 100)
        total_count = self._cum_totals.get("train", len(df))
        cum_acc, cum_correct = self._update_cumulative("train", all_recs, total_count)
        logger.info("[train sec %d] Cumulative Acc: %.2f%% (%d/%d)", sec_idx, cum_acc * 100, cum_correct, total_count)
        out_dir = self.output_dir / "hle_mem0"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"train_sec_{sec_idx}")
        out_path = out_dir / f"hle_{safe_tag}_results_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        pd.DataFrame(all_recs).to_csv(out_path, index=False)
        logger.info("Saved train section %d results to: %s", sec_idx, out_path)
        per_item = {r["question"]: bool(r["correct"]) for r in all_recs}
        return {"acc": float(acc), "cum_acc": float(cum_acc), "per_item": per_item}

    def run(self) -> None:
        train_df, valid_df = self._load()
        self._cum_totals["train"] = len(train_df)
        self._cum_totals["valid"] = len(valid_df)

        if self.ckpt_eval_enabled:
            if len(valid_df) == 0:
                logger.warning("Valid set is empty; skip ckpt evaluation.")
                return
            self._eval_ckpt_sequence(valid_df)
            return

        start_sec = 1
        if self.ckpt_resume_enabled:
            start_sec = self._resume_from_ckpt()
            if start_sec > self.num_sections:
                logger.warning("Resume epoch exceeds num_sections; skip training.")
                return
        if len(valid_df) != 0 and start_sec == 1:
            self._eval_split(valid_df, tag="valid_initial", step=0, cum_key="valid", total=len(valid_df))

        sections = (
            [list(range(len(train_df)))]
            if self.num_sections <= 1
            else [list(range(len(train_df))) for _ in range(self.num_sections)]
        )
        for sec_idx in range(start_sec, len(sections) + 1):
            if len(train_df) != 0:
                self._train_one_section(train_df, sec_idx)
            if len(valid_df) != 0:
                self._eval_split(
                    valid_df,
                    tag=f"valid_sec_{sec_idx}",
                    step=sec_idx,
                    cum_key="valid",
                    total=len(valid_df),
                )
            if len(train_df) != 0:
                self._save_ckpt(sec_idx)

        try:
            with self._image_lock:
                self._persist_image_cache_unlocked()
        except Exception:
            logger.debug("Failed to persist image cache on shutdown", exc_info=True)


def _setup_logging(name: str) -> None:
    log_dir = REPO_ROOT / "logs"
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
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info("Logging configured. Log file: %s", log_filepath)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mem0 baseline on HLE")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "mem0_hle_config.yaml"))
    parser.add_argument("--dataset", type=str, help="HLE parquet path", default="/mnt/public/code/zst/memory/hle/test-00000-of-00001-filtered.parquet")
    parser.add_argument("--num_valid", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--judge_model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=['Computer Science/AI', 'Math', 'Biology/Medicine', 'Physics', 'Chemistry', 'Engineering', 'Humanities/Social Science', 'Other'],
        help="Filter HLE rows to these categories (space-separated list).",
    )
    parser.add_argument(
        "--category_ratio",
        type=float,
        default=1.0,
        help="Per-category sampling ratio (0-1) after filtering categories.",
    )
    args = parser.parse_args()

    cfg = MempConfig.from_yaml(args.config)
    _setup_logging(cfg.experiment.experiment_name)

    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S')
    log_dir = out_dir / "hle_mem0" / f"exp_{cfg.experiment.experiment_name}_{run_id}" / "local_cache"
    log_dir.mkdir(parents=True, exist_ok=True)

    llm = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_temperature=cfg.llm.temperature,
        default_max_tokens=cfg.llm.max_tokens,
        token_log_dir=str(log_dir),
    )
    llm_judge = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=args.judge_model,
        default_temperature=0.0,
        default_max_tokens=4096,
        token_log_dir=str(log_dir),
    ) if args.judge_model else None

    resume_user_id = None
    if getattr(cfg.experiment, "ckpt_resume_enabled", False):
        resume_path = getattr(cfg.experiment, "ckpt_resume_path", None)
        resume_epoch = getattr(cfg.experiment, "ckpt_resume_epoch", None)
        if resume_path and resume_epoch:
            try:
                root = Path(resume_path)
                if (root / "snapshot").is_dir():
                    candidate = root / "snapshot" / str(resume_epoch)
                elif (root / str(resume_epoch)).is_dir():
                    candidate = root / str(resume_epoch)
                else:
                    candidate = root
                meta_path = candidate / "snapshot_meta.json"
                if meta_path.is_file():
                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    resume_user_id = meta.get("user_id")
                    if resume_user_id:
                        logger.info("Resume user_id loaded from %s: %s", meta_path, resume_user_id)
            except Exception:
                logger.warning("Failed to load resume user_id", exc_info=True)

    user_id = resume_user_id or f"mem0_hle_{cfg.experiment.experiment_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    mem0_cfg = Mem0Config(
        mode="oss",
        user_id=user_id,
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
    )
    mem0_store = Mem0Store(mem0_cfg, log_callback=None)
    bench = HLEMem0Bench(
        name=cfg.experiment.experiment_name,
        llm=llm,
        llm_judge=llm_judge,
        selection=HLESelection(
            dataset_path=args.dataset,
            num_valid=(args.num_valid if args.num_valid > 0 else None),
            num_train=(args.num_train if args.num_train > 0 else None),
            categories=args.categories or getattr(cfg.experiment, "hle_categories", None),
            category_ratio=args.category_ratio if args.category_ratio is not None else getattr(cfg.experiment, "hle_category_ratio", None),
        ),
        output_dir=out_dir,
        mem0_store=mem0_store,
        run_id=run_id,
        temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
        max_tokens=(args.max_tokens if args.max_tokens is not None else (cfg.llm.max_tokens or 4096)),
        retrieve_k=cfg.memory.k_retrieve,
        num_sections=cfg.experiment.num_sections,
        batch_size=cfg.experiment.batch_size,
        dataset_ratio=getattr(cfg.experiment, "dataset_ratio", 1.0),
        random_seed=getattr(cfg.experiment, "random_seed", 42) or 42,
        train_valid_split=getattr(cfg.experiment, "train_valid_split", 0.8),
        ckpt_eval_enabled=getattr(cfg.experiment, "ckpt_eval_enabled", False),
        ckpt_eval_path=getattr(cfg.experiment, "ckpt_eval_path", None),
        ckpt_resume_enabled=getattr(cfg.experiment, "ckpt_resume_enabled", False),
        ckpt_resume_path=getattr(cfg.experiment, "ckpt_resume_path", None),
        ckpt_resume_epoch=getattr(cfg.experiment, "ckpt_resume_epoch", None),
    )
    mem0_store._log_callback = bench._log_mem0_event
    bench.tau = float(getattr(cfg.rl_config, "tau", 0.0))
    bench.run()


if __name__ == "__main__":
    main()
