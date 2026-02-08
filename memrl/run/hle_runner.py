from __future__ import annotations
import logging
import json
import hashlib
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
import time
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .base_runner import BaseRunner
from memrl.providers.llm import OpenAILLM
from memrl.service.memory_service import MemoryService

logger = logging.getLogger(__name__)


@dataclass
class HLESelection:
    valid_path: str
    train_path: Optional[str] = None
    num_valid: Optional[int] = None
    num_train: Optional[int] = None
    categories: Optional[List[str]] = None  # categories to keep
    category_ratio: Optional[float] = None  # per-category sampling ratio (0,1]


class HLERunner(BaseRunner):
    """HLE benchmark runner, mirroring AIME/MATH runners.

    Dataset expected columns: id, question, image (base64 or empty), answer
    """

    def __init__(
        self,
        name: str,
        llm: OpenAILLM,
        llm_judge: Optional[OpenAILLM],
        selection: 'HLESelection',
        output_dir: Path,
        memory_service: Optional[MemoryService] = None,
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
        baseline_mode: Optional[str] = None,
        baseline_k: int = 10,
    ) -> None:
        self.name = name
        self.llm = llm
        self.sel = selection
        self.output_dir = Path(output_dir)
        self.memory_service = memory_service
        self.llm_judge = llm_judge
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieve_k = max(0, int(retrieve_k))
        self.num_sections = num_sections
        self.batch_size = max(1, int(batch_size))
        self.dataset_ratio = float(dataset_ratio)
        self.random_seed = random_seed
        self.train_valid_split = float(train_valid_split)
        self.ckpt_eval_enabled = bool(ckpt_eval_enabled)
        self.ckpt_eval_path = str(ckpt_eval_path) if ckpt_eval_path else None
        self.ckpt_resume_enabled = ckpt_resume_enabled
        self.ckpt_resume_path = ckpt_resume_path
        self.ckpt_resume_epoch = ckpt_resume_epoch
        self.baseline_mode = (baseline_mode or "").strip().lower() or None
        self.baseline_k = max(1, int(baseline_k))

        self.run_id = run_id or time.strftime('%Y%m%d-%H%M%S')
        ts = self.run_id
        tb_dir = self.output_dir / "tensorboard" / f"exp_hle_{self.name}_{ts}"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"TensorBoard logs at: {tb_dir}")
        self.ck_dir = self.output_dir / "hle" / f"exp_{self.name}_{self.run_id}"
        self.log_dir = self.ck_dir / "local_cache"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.llm_log_path = self.log_dir / "llm_calls.jsonl"
        self._log_lock = threading.Lock()
        self._image_lock = threading.Lock()
        self._image_store: Dict[str, str] = {}  # image_id -> data_url
        self._image_hash_to_id: Dict[str, str] = {}
        self._image_id_counter = 0
        self._image_store_path = self.log_dir / "image_store.json"
        self._image_index_path = self.log_dir / "image_hash_index.json"
        self._load_image_cache()
        self.df_train: Optional[pd.DataFrame] = None
        self.df_valid: Optional[pd.DataFrame] = None
        self.train_cumulative_correct_map = {}
        self.valid_cumulative_correct_map = {}
        self._cum_state_path = self.log_dir / "cum_state.json"
        self._resume_section_start = 0
        # resume flags (optional; can be set after init if needed)
        self._cum_state_path = self.log_dir / "cum_state.json"
        self._resume_section_start = 0
        self._resume_from_ckpt_if_needed()

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

        # HLE judge prompt (from hle_eval)
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

    # ------------------------------
    # Resume helpers
    # ------------------------------
    def _load_cum_state(self, state_path: Optional[Path] = None):
        path = state_path or self._cum_state_path
        if path.exists():
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
                self.train_cumulative_correct_map = data.get("train_cum", {})
                self.valid_cumulative_correct_map = data.get("valid_cum", {})
                self._resume_section_start = int(data.get("next_section", 0))
                logger.info(
                    f"[Resume] Loaded cumulative state from {path}: train {len(self.train_cumulative_correct_map)}, "
                    f"valid {len(self.valid_cumulative_correct_map)}, next_section={self._resume_section_start}"
                )
            except Exception as e:
                logger.warning(f"[Resume] Failed to load cum_state from {path}: {e}")

    def _save_cum_state(self, next_section: int):
        try:
            data = {
                "train_cum": self.train_cumulative_correct_map,
                "valid_cum": self.valid_cumulative_correct_map,
                "next_section": int(next_section),
            }
            with open(self._cum_state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Resume] Failed to save cum_state: {e}")

    def _resume_from_ckpt_if_needed(self):
        # decide where to resume memory from
        resume_root = None
        if getattr(self, "ckpt_resume_enabled", False) and getattr(self, "ckpt_resume_path", None):
            resume_root = Path(self.ckpt_resume_path)
        else:
            # fallback: if current ck_dir has snapshot, use it
            if (self.ck_dir / "snapshot").exists():
                resume_root = self.ck_dir

        # load cumulative state (prefer resume root if available)
        if resume_root is not None:
            candidate_local = None
            if (resume_root / "local_cache").exists():
                candidate_local = resume_root / "local_cache"
            elif resume_root.parent.name == "snapshot" and (resume_root.parent.parent / "local_cache").exists():
                candidate_local = resume_root.parent.parent / "local_cache"
            if candidate_local and (candidate_local / "cum_state.json").exists():
                self._load_cum_state(candidate_local / "cum_state.json")
            else:
                self._load_cum_state()
        else:
            self._load_cum_state()

        if resume_root is None:
            return

        snapshot_root = resume_root / "snapshot" if (resume_root / "snapshot").is_dir() else resume_root
        ckpts = [p for p in snapshot_root.iterdir() if p.is_dir() and p.name.isdigit()]
        if not ckpts:
            return
        ckpts.sort(key=lambda p: int(p.name))
        target = ckpts[-1]
        if getattr(self, "ckpt_resume_epoch", None):
            try:
                target = next(p for p in ckpts if int(p.name) == int(self.ckpt_resume_epoch))
            except StopIteration:
                logger.warning(f"[Resume] ckpt epoch {self.ckpt_resume_epoch} not found, using last.")
        try:
            if self.memory_service:
                self.memory_service.load_checkpoint_snapshot(str(target), mem_cube_id=self.memory_service.default_cube_id)
                logger.info(f"[Resume] Loaded memory snapshot from {target}")
        except Exception as e:
            logger.warning(f"[Resume] Failed to load memory snapshot {target}: {e}")

    def _load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load dataset from a single source, apply dataset_ratio, then split into train/valid (e.g., 80/20).
        """

        # 1. 读取数据（统一来源）
        if not Path(self.sel.valid_path).exists():
            raise ValueError(f"HLE dataset path does not exist: {self.sel.valid_path}")
        df = pd.read_parquet(self.sel.valid_path)

        # 1.1 optional category filter / sampling
        df = self._filter_by_category(df)

        # 2. 按比例抽样整个数据集
        df = self._apply_dataset_ratio(df, "full")

        if len(df) == 0:
            raise ValueError("HLE dataset is empty after category filtering/sampling")

        # 3. 检查必要列
        for c in ['id', 'question', 'answer']:
            if c not in df.columns:
                raise ValueError(f"HLE dataset missing required column: {c}")

        df = df.reset_index(drop=True)

        n_total = len(df)

        # 4. 按类别划分 train/valid（每个 category 内按比例切分）
        split_ratio = getattr(self, "train_valid_split", 0.8)
        if "category" not in df.columns:
            raise ValueError("HLE dataset missing 'category' column for category-wise split")

        train_parts = []
        valid_parts = []
        for cat, group in df.groupby("category", sort=False):
            shuffled = group.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
            n_train = int(len(shuffled) * split_ratio)
            train_parts.append(shuffled.iloc[:n_train].copy())
            valid_parts.append(shuffled.iloc[n_train:].copy())

        train = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[:0].copy()
        valid = pd.concat(valid_parts, ignore_index=True) if valid_parts else df.iloc[:0].copy()

        # 5. 应用 num_train / num_valid 限制
        if self.sel.num_train:
            train = train.head(int(self.sel.num_train))
        if self.sel.num_valid:
            valid = valid.head(int(self.sel.num_valid))

        # 6. 保存结果
        self.df_train, self.df_valid = train.reset_index(drop=True), valid.reset_index(drop=True)
        logger.info(
            f"HLE loaded from single dataset: total={n_total}, train={len(self.df_train)}, valid={len(self.df_valid)}"
        )
        return self.df_train, self.df_valid

    def _apply_dataset_ratio(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        ratio = getattr(self, "dataset_ratio", 1.0)
        if df is None or df.empty or not (0 < ratio < 1):
            return df
        n_keep = max(1, int(len(df) * ratio))
        if n_keep >= len(df):
            return df
        sampled = df.sample(n=n_keep, random_state=self.random_seed).reset_index(drop=True)
        logger.info(
            "HLE %s split reduced via dataset_ratio %.2f: %d -> %d rows",
            split,
            ratio,
            len(df),
            len(sampled),
        )
        return sampled

    def _filter_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset by categories and optional per-category sampling ratio."""
        if df is None or df.empty:
            return df
        cats = self.sel.categories
        ratio = self.sel.category_ratio
        if cats:
            if 'category' not in df.columns:
                raise ValueError("HLE dataset missing 'category' column for category filtering")
            df = df[df['category'].isin(cats)].reset_index(drop=True)
            logger.info("HLE filtered categories %s -> %d rows", cats, len(df))
        if ratio is not None and 0 < ratio < 1:
            if 'category' not in df.columns:
                raise ValueError("HLE dataset missing 'category' column for category ratio sampling")
            def _sample_group(g: pd.DataFrame) -> pd.DataFrame:
                n_keep = max(1, int(len(g) * ratio))
                return g.sample(n=n_keep, random_state=self.random_seed)
            df = df.groupby('category', group_keys=False).apply(_sample_group).reset_index(drop=True)
            logger.info("HLE applied category_ratio %.2f -> %d rows", ratio, len(df))
        elif ratio is not None:
            logger.warning("category_ratio %.3f is out of (0,1); skip sampling", ratio)
        return df

    def _resolve_ckpt_dirs(self, ckpt_root: Path) -> List[Path]:
        """Resolve snapshot directories (numeric subfolders) from an experiment or snapshot root."""
        if (ckpt_root / "snapshot").is_dir():
            snapshot_root = ckpt_root / "snapshot"
        else:
            snapshot_root = ckpt_root
        if not snapshot_root.is_dir():
            raise ValueError(f"ckpt root does not exist: {snapshot_root}")
        ckpts = [p for p in snapshot_root.iterdir() if p.is_dir() and p.name.isdigit()]
        ckpts.sort(key=lambda p: int(p.name))
        return ckpts

    def _prune_valid_memories(self, valid_questions: Set[str]) -> None:
        """Remove validation questions from local memory indices to avoid leakage."""
        if not self.memory_service or not valid_questions:
            return
        dict_mem = getattr(self.memory_service, "dict_memory", None)
        if isinstance(dict_mem, dict):
            for q in list(dict_mem.keys()):
                if q in valid_questions:
                    dict_mem.pop(q, None)
        query_embeddings = getattr(self.memory_service, "query_embeddings", None)
        if isinstance(query_embeddings, dict):
            for q in list(query_embeddings.keys()):
                if q in valid_questions:
                    query_embeddings.pop(q, None)

    def _eval_ckpt_sequence(self, valid_df: pd.DataFrame) -> None:
        """Load historical checkpoints sequentially and evaluate on valid set."""
        if not self.memory_service:
            raise RuntimeError("memory_service is required for ckpt evaluation")
        if not self.ckpt_eval_path:
            raise ValueError("ckpt_eval_path is not set")
        ckpt_root = Path(self.ckpt_eval_path)
        ckpt_dirs = self._resolve_ckpt_dirs(ckpt_root)
        if not ckpt_dirs:
            raise ValueError(f"No checkpoint folders found under {ckpt_root}")
        valid_questions = set(valid_df["question"].astype(str).tolist())

        for idx, ckpt_dir in enumerate(ckpt_dirs, start=1):
            logger.info("Loading ckpt %s (%d/%d) for eval", ckpt_dir, idx, len(ckpt_dirs))
            self.memory_service.load_checkpoint_snapshot(str(ckpt_dir), mem_cube_id=self.memory_service.default_cube_id)
            self._prune_valid_memories(valid_questions)
            self._eval_split(valid_df, tag=f"valid_ckpt_{idx}", step=idx)

    def _baseline_task_key(self, data: Any) -> str:
        """Canonical key for pass@k/reflection loops; prefer id, fallback to question text."""
        candidate_id = None
        question = None
        try:
            candidate_id = data["id"]
        except Exception:
            candidate_id = None
        try:
            question = data["question"]
        except Exception:
            question = None
        if candidate_id is not None:
            try:
                if not pd.isna(candidate_id):
                    cid = str(candidate_id).strip()
                    if cid and cid.lower() != "nan":
                        return cid
            except Exception:
                cid = str(candidate_id).strip()
                if cid:
                    return cid
        return str(question or "")

    def _extract_solution_only(self, trajectory: str) -> str:
        if not trajectory:
            return ""
        if "SOLUTION" in trajectory:
            return trajectory.split("SOLUTION", 1)[1].strip()
        return trajectory.strip()


    def _format_reflection_note(self, question: str, trajectory: str, success: bool) -> str:
        status = "CORRECT" if success else "INCORRECT"

        solution_only = self._extract_solution_only(trajectory)

        note_parts = [
            "You attempted this question before.",
            f"Result: {status}",
            f"Question: {question}",
            "Previous attempt (solution only):",
            solution_only,
            "Reflect on mistakes or gaps, then solve the problem again with a better solution.",
        ]
        return "\n".join([p for p in note_parts if p])


    # ---------- Image helpers ----------
    def _register_image(self, image: Any) -> Optional[Tuple[str, str]]:
        """Convert raw image to data URL, cache in store, and return (image_id, data_url)."""
        if image is None:
            return None
        data_url = None
        if isinstance(image, str) and image.strip():
            data_url = image.strip()
        elif isinstance(image, dict) and 'bytes' in image:
            raw = image.get('bytes')
            if isinstance(raw, bytes):
                b64 = base64.b64encode(raw).decode('utf-8')
                data_url = f"data:image/jpeg;base64,{b64}"
        if not data_url:
            return None
        key = hashlib.md5(data_url.encode('utf-8')).hexdigest()
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
        """Return list of (image_id, data_url) for known ids."""
        imgs: List[Tuple[str, str]] = []
        for iid in image_ids or []:
            url = self._image_store.get(str(iid))
            if url:
                imgs.append((str(iid), url))
        return imgs

    def _persist_image_cache_unlocked(self) -> None:
        """Persist image store/index to log_dir (caller holds _image_lock)."""
        try:
            with open(self._image_store_path, "w", encoding="utf-8") as f:
                json.dump(self._image_store, f, ensure_ascii=False)
            with open(self._image_index_path, "w", encoding="utf-8") as f:
                payload = {
                    "hash_index": self._image_hash_to_id,
                    "counter": self._image_id_counter,
                }
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            logger.debug("Failed to persist image cache", exc_info=True)

    def _load_image_cache(self) -> None:
        """Load persisted image cache if available."""
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

    def _collect_question_images(self, row: pd.Series) -> List[Any]:
        """Collect raw image objects from the row (supports single image/image_preview)."""
        images: List[Any] = []
        if 'image' in row.index and pd.notna(row['image']) and str(row['image']).strip():
            images.append(row['image'])
        # if 'image_preview' in row.index and pd.notna(row['image_preview']):
        #     images.append(row['image_preview'])
        return images

    def _extract_mem_image_ids(self, mem: Dict[str, Any]) -> List[str]:
        md = mem.get("metadata")
        ids = []
        try:
            if hasattr(md, "model_extra"):
                ids = md.model_extra.get("image_ids") or []
            elif isinstance(md, dict):
                ids = md.get("image_ids") or []
        except Exception:
            pass
        try:
            return [str(x) for x in ids if x]
        except Exception:
            return []

    # ---------- Memory helpers ----------
    def _mem_success_flag(self, m: Dict[str, Any]) -> bool:
        md = m.get("metadata")
        try:
            if hasattr(md, "model_extra"):
                return bool(md.model_extra.get('success'))
            if isinstance(md, dict):
                return bool(md.get('success'))
        except Exception:
            pass
        return False

    def _build_memory_context(self, selected_mems: List[Dict[str, Any]], limit: int) -> Tuple[str, List[str], Set[str]]:
        if not selected_mems:
            return "", [], set()
        retrieved_ids: List[str] = []
        memory_image_ids: Set[str] = set()
        succ_blocks, fail_blocks = [], []
        for m in selected_mems[: max(0, limit) or len(selected_mems)]:
            mid = m.get('memory_id') or m.get('id')
            if mid:
                retrieved_ids.append(str(mid))
            content = m.get('content') or m.get('full_content') or ''
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

    # ---------- Prompt & eval ----------
    def _build_messages(
        self,
        question: str,
        memory_ctx: Optional[str] = None,
        answer_type: Optional[Any] = None,
        question_image_ids: Optional[List[str]] = None,
        images_info: Optional[List[Tuple[str, str, str]]] = None,
        reflection_note: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        answer_type_norm = ""
        if answer_type is not None:
            answer_type_norm = str(answer_type).strip().lower()
        system_prompt = (
            self.EXACT_ANSWER_SYSTEM_PROMPT
            if answer_type_norm == "exactmatch"
            else self.MULTIPLE_CHOICE_SYSTEM_PROMPT
        )
        # Compose multi-modal user content (text + list of images)
        legend = ""
        if images_info:
            lines = [f"{i+1}. [{img_id}] ({source})" for i, (img_id, _, source) in enumerate(images_info)]
            legend = "Attached images:\n" + "\n".join(lines)
        text_block = question if not legend else f"Now solve the following question: \n\n[Image IDs: {question_image_ids}]\n{question}\n\n{legend}"
        content: List[Dict[str, Any]] = [{"type": "text", "text": text_block}]
        if images_info:
            for img_id, url, source in images_info:
                content.append({"type": "text", "text": f"Image [{img_id}] ({source})"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if reflection_note:
            msgs.append({"role": "system", "content": reflection_note})
        if memory_ctx:
            msgs.append({"role": "system", "content": memory_ctx})
        msgs.append({"role": "user", "content": content})
        return msgs

    def _extract_answer(self, text: str) -> str:
        # Extract line starting with 'Answer:' then strip trailing punctuation
        m = re.search(r"^\s*Answer\s*:\s*(.+)$", text or "", flags=re.I|re.M)
        ans = (m.group(1) if m else (text or "")).strip()
        return re.sub(r"[\s\.]$", "", ans)

    def _log_llm_call(
        self,
        call_type: str,
        messages: Any,
        response: Any,
        meta: Optional[Dict[str, Any]] = None,
        parsed: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist each LLM interaction (inputs/outputs) to a local cache JSONL file."""
        entry = {
            "ts": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "type": call_type,
            "meta": meta or {},
            "messages": messages,
            "response": response,
        }
        if parsed is not None:
            entry["parsed"] = parsed
        try:
            payload = json.dumps(entry, ensure_ascii=False, default=str)
        except Exception as e:
            try:
                entry["messages"] = str(messages)
                payload = json.dumps(entry, ensure_ascii=False, default=str)
            except Exception:
                logger.debug("Failed to serialize LLM call log: %s", e)
                return
        try:
            with self._log_lock:
                with open(self.llm_log_path, "a", encoding="utf-8") as f:
                    f.write(payload + "\n")
        except Exception:
            logger.debug("Failed to write LLM call log", exc_info=True)

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
        # Try JSON parse
        try:
            # Extract JSON object substring if needed
            m = re.search(r"\{[\s\S]*\}", judge_text)
            jtxt = m.group(0) if m else judge_text
            obj = _json.loads(jtxt)
            # accept keys with slight variations
            result["model_answer"] = obj.get("extracted_final_answer") or obj.get("extracted_answer")
            result["reasoning"] = obj.get("reasoning")
            corr = str(obj.get("correct", "no")).strip().lower()
            result["correct"] = "yes" if "yes" in corr else "no"
            try:
                result["confidence"] = int(obj.get("confidence", 0))
            except Exception:
                result["confidence"] = 0
        except Exception:
            # Fallback regex parsing for 'correct:' and 'extracted_final_answer:'
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
            self._log_llm_call("judge", messages, judge_text, meta=log_meta, parsed=result)
        except Exception:
            logger.debug("Failed to log judge LLM call", exc_info=True)
        return result

    def _evaluate_row(self, row: pd.Series, reflection_note: Optional[str] = None) -> Dict[str, Any]:
        q = str(row['question'])
        gold = str(row['answer'])
        # Collect question images and register them
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
        retrieved_topk_queries = None
        memory_image_ids: Set[str] = set()
        if self.memory_service and self.retrieve_k > 0:
            try:
                # Align retrieval threshold knob across benchmarks: rl_config.sim_threshold (fallback tau).
                rl_cfg = getattr(self.memory_service, "rl_config", None)
                tau = float(getattr(rl_cfg, "sim_threshold", getattr(rl_cfg, "tau", 0.0)))
            except Exception:
                tau = 0.0
            try:
                ret = self.memory_service.retrieve_query(q, k=self.retrieve_k, threshold=tau)
                if isinstance(ret, tuple):
                    ret_result, retrieved_topk_queries = ret
                else:
                    ret_result, retrieved_topk_queries = ret, None
                selected_mems = ret_result.get('selected', []) if ret_result else []
                memory_ctx, retrieved_ids, memory_image_ids = self._build_memory_context(selected_mems, self.retrieve_k)
            except Exception as e:
                logger.warning("Memory retrieval failed: %s", e)

        # Resolve memory images from store
        memory_images_info: List[Tuple[str, str, str]] = []
        if memory_image_ids:
            for img_id, url in self._fetch_images_by_ids(list(memory_image_ids)):
                memory_images_info.append((img_id, url, "memory"))

        images_info = question_images_info + memory_images_info

        answer_type = row.get('answer_type', None)
        messages = self._build_messages(
            q,
            memory_ctx=memory_ctx,
            answer_type=answer_type,
            question_image_ids=question_image_ids,
            images_info=images_info,
            reflection_note=reflection_note,
        )
        call_meta = {
            "question_id": row.get('id', None),
            "answer_type": answer_type,
        }
        gen_error = None
        try:
            if self.llm.model != "gemini-3-pro-preview":
                kwargs = dict(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    )
            else:
                kwargs = dict(
                    messages=messages,
                    temperature=self.temperature
                    )
                
            if self.llm.model == "gpt-5.2":
                kwargs["reasoning_effort"] = "high"

            output = self.llm.generate(**kwargs)

        except Exception as e:
            logger.error("LLM error: %s", e)
            gen_error = str(e)
            output = ""
        if gen_error:
            call_meta["error"] = gen_error
        self._log_llm_call("solution", messages, output, meta=call_meta)
        judge_res = self._hle_judge(q, gold, output or "", meta={"question_id": row.get('id', None)})
        correct = True if str(judge_res.get("correct", "no")).lower() == "yes" else False

        rec: Dict[str, Any] = {
            "id": row.get('id', None),
            "question": q,
            "gold": gold,
            "raw_output": output,
            "correct": bool(correct),
            "judge_response": judge_res,
            "retrieved_ids": retrieved_ids,
            "image_ids": question_image_ids,
            "trajectory": f"QUESTION\n{q}\n\nSOLUTION\n{(output or '').strip()}\n",
        }
        if retrieved_topk_queries is not None:
            rec["retrieved_topk_queries"] = retrieved_topk_queries
        return rec

    def _eval_split(self, df: pd.DataFrame, tag: str, step: Optional[int] = None) -> Dict[str, float]:
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
                fut2pos = {ex.submit(self._evaluate_row, df.iloc[i]): pos for pos, i in enumerate(b)}
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
        try:
            if step is None:
                self.writer.add_scalar(f"Evaluation/Acc", acc)
            else:
                self.writer.add_scalar(f"Evaluation/Acc", acc, step)
            self.writer.flush()
        except Exception:
            pass
        out_dir = self.output_dir / "hle"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tag))
        out_path = out_dir / f"hle_{safe_tag}_results_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        logger.info("Saved %s results to: %s", tag, out_path)
        per_item = { r["question"]: bool(r["correct"]) for r in results}

        return {
            "acc": float(acc),
            "per_item": per_item
        }

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
                fut2pos = {ex.submit(self._evaluate_row, df.iloc[i]): pos for pos, i in enumerate(b)}
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
            if self.memory_service and batch_recs:
                try:
                    task_descriptions = [r["question"] for r in batch_recs]
                    trajectories = [r["trajectory"] for r in batch_recs]
                    successes = [bool(r["correct"]) for r in batch_recs]
                    retrieved_ids_list = [r.get("retrieved_ids") or [] for r in batch_recs]
                    retrieved_queries = [r.get("retrieved_topk_queries") for r in batch_recs]
                    metadatas = []
                    for rec_entry, s in zip(batch_recs, successes):
                        metadatas.append(
                            {
                                "source_benchmark": "HLE",
                                "success": s,
                                "q_value": 1.0 if s else 0.0,
                                "q_visits": 0,
                                "q_updated_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                                "last_used_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                                "reward_ma": 0.0,
                                "image_ids": rec_entry.get("image_ids", []),
                            }
                        )
                    self.memory_service.add_memories(
                        task_descriptions=task_descriptions,
                        trajectories=trajectories,
                        successes=successes,
                        retrieved_memory_queries=retrieved_queries,
                        retrieved_memory_ids_list=retrieved_ids_list,
                        metadatas=metadatas,
                    )
                    try:
                        self.memory_service.update_values([float(s) for s in successes], retrieved_ids_list)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning("[train sec %d] batch memory add/update failed: %s", sec_idx, e)
        if not all_recs:
            return {"acc": 0.0}
        acc = correct_so_far / len(all_recs)
        logger.info("Section %d Train Acc: %.2f%%", sec_idx, acc * 100)
        try:
            self.writer.add_scalar("Train/Section_Acc", acc, sec_idx)
            self.writer.flush()
        except Exception:
            pass 
        ckpt_meta = self.memory_service.save_checkpoint_snapshot(self.ck_dir, ckpt_id=sec_idx)
        logger.info(f" Saved ckpt: {ckpt_meta}")
        per_item = { r["question"]: bool(r["correct"]) for r in all_recs }

        return {
            "acc": float(acc),
            "per_item": per_item
        }

    def _baseline_eval_split(
        self,
        df: pd.DataFrame,
        desc: str,
        *,
        reflection_notes: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate a dataframe in batches, optionally injecting reflection notes per item."""
        if df is None or len(df) == 0:
            return []
        idxs = list(range(len(df)))
        batches = [idxs[i:i + self.batch_size] for i in range(0, len(idxs), self.batch_size)]
        results: List[Dict[str, Any]] = []
        for b in tqdm(batches, desc=desc):
            batch_results: List[Optional[Dict[str, Any]]] = [None] * len(b)
            with ThreadPoolExecutor(max_workers=min(len(b), self.batch_size)) as ex:
                fut2pos = {}
                for pos, i in enumerate(b):
                    row = df.iloc[i]
                    note = None
                    if reflection_notes:
                        key = self._baseline_task_key(row)
                        note = reflection_notes.get(key)
                    fut2pos[ex.submit(self._evaluate_row, row, note)] = pos
                for fut in as_completed(fut2pos):
                    pos = fut2pos[fut]
                    try:
                        batch_results[pos] = fut.result()
                    except Exception as e:
                        logger.warning("[baseline %s] item #%d failed: %s", desc, pos, e)
                        batch_results[pos] = None
            results.extend([r for r in batch_results if r is not None])
        return results

    def _run_passk_baseline(self, train_df: pd.DataFrame) -> None:
        total_tasks = len(train_df)
        if total_tasks == 0:
            logger.warning("No train data for pass@k baseline; aborting.")
            return
        solved: Set[str] = set()
        summary = []
        result_path = self.log_dir / "baseline_passk_results.jsonl"
        summary_path = self.log_dir / "baseline_passk_summary.json"

        for round_idx in range(1, self.baseline_k + 1):
            logger.info("Starting pass@k round %d/%d", round_idx, self.baseline_k)
            pending_idx = [
                i for i in range(total_tasks)
                if self._baseline_task_key(train_df.iloc[i]) not in solved
            ]
            if pending_idx:
                pending_df = train_df.iloc[pending_idx].reset_index(drop=True)
                trajectories = self._baseline_eval_split(pending_df, desc=f"pass@k round {round_idx}")
                for traj in trajectories:
                    key = self._baseline_task_key(traj)
                    if traj.get("correct") and key:
                        solved.add(str(key))
                    payload = {
                        "round": round_idx,
                        "baseline": "passk",
                        **traj,
                    }
                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            else:
                logger.info("All tasks already solved before round %d; skipping inference.", round_idx)
            cum_acc = (len(solved) / total_tasks) if total_tasks > 0 else 0.0
            summary.append({"round": round_idx, "cum_acc": cum_acc, "solved": len(solved), "total": total_tasks})
            try:
                self.writer.add_scalar("Baseline/PassK_Cumulative_Acc", cum_acc, round_idx)
            except Exception:
                pass

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _run_reflection_baseline(self, train_df: pd.DataFrame) -> None:
        total_tasks = len(train_df)
        if total_tasks == 0:
            logger.warning("No train data for reflection baseline; aborting.")
            return
        solved: Set[str] = set()
        summary = []
        reflection_notes: Dict[str, str] = {}
        result_path = self.log_dir / "baseline_reflection_results.jsonl"
        summary_path = self.log_dir / "baseline_reflection_summary.json"
        state_path = self.log_dir / "baseline_reflection_state.json"

        start_round = 1
        if state_path.exists():
            try:
                state = json.load(open(state_path, "r", encoding="utf-8"))
                solved = {str(x) for x in state.get("solved", [])}
                reflection_notes = {str(k): v for k, v in state.get("reflection_notes", {}).items()}
                last_completed = int(state.get("last_completed_round", 0))
                start_round = max(1, last_completed + 1)
                logger.info("Resuming reflection baseline from round %d", start_round)
            except Exception as e:
                logger.warning("Failed to load reflection baseline state from %s: %s", state_path, e)

        if start_round > self.baseline_k:
            logger.info("Reflection baseline already completed (last round %d).", start_round - 1)
            return

        for round_idx in range(start_round, self.baseline_k + 1):
            logger.info("Starting reflection round %d/%d", round_idx, self.baseline_k)
            pending_idx = [
                i for i in range(total_tasks)
                if self._baseline_task_key(train_df.iloc[i]) not in solved
            ]
            if pending_idx:
                pending_df = train_df.iloc[pending_idx].reset_index(drop=True)
                trajectories = self._baseline_eval_split(
                    pending_df,
                    desc=f"reflection round {round_idx}",
                    reflection_notes=reflection_notes,
                )
                for traj in trajectories:
                    key = self._baseline_task_key(traj)
                    if key:
                        reflection_notes[key] = self._format_reflection_note(
                            traj.get("question", ""),
                            traj.get("trajectory", ""),
                            bool(traj.get("correct")),
                        )
                        if traj.get("correct"):
                            solved.add(str(key))
                    payload = {
                        "round": round_idx,
                        "baseline": "reflection",
                        **traj,
                    }
                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            else:
                logger.info("All tasks already solved before round %d; skipping inference.", round_idx)
            cum_acc = (len(solved) / total_tasks) if total_tasks > 0 else 0.0
            logger.info("Reflection round %d completed. Cumulative Acc: %.2f%% (%d/%d)", round_idx, cum_acc * 100, len(solved), total_tasks)
            summary.append({"round": round_idx, "cum_acc": cum_acc, "solved": len(solved), "total": total_tasks})
            try:
                self.writer.add_scalar("Baseline/Reflection_Cumulative_Acc", cum_acc, round_idx)
            except Exception:
                pass
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "last_completed_round": round_idx,
                            "solved": sorted(solved),
                            "reflection_notes": reflection_notes,
                            "total": total_tasks,
                            "updated_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                logger.warning("Failed to save reflection baseline state to %s: %s", state_path, e)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def run(self):
        train_df, valid_df = self._load()

        if self.baseline_mode in {"passk", "reflection"}:
            if self.baseline_mode == "passk":
                self._run_passk_baseline(train_df)
            else:
                self._run_reflection_baseline(train_df)
            try:
                self.writer.close()
            except Exception:
                pass
            try:
                with self._image_lock:
                    self._persist_image_cache_unlocked()
            except Exception:
                logger.debug("Failed to persist image cache on shutdown", exc_info=True)
            return

        # If enabled, evaluate by loading historical checkpoints sequentially.
        if self.ckpt_eval_enabled:
            if len(valid_df) == 0:
                logger.warning("Valid set is empty; skip ckpt evaluation.")
                return
            self._eval_ckpt_sequence(valid_df)
            return

        # ------------------------------
        if not self.train_cumulative_correct_map:
            self.train_cumulative_correct_map = {}
        if not self.valid_cumulative_correct_map:
            self.valid_cumulative_correct_map = {}

        # ------------------------------
        if len(valid_df) != 0 and self._resume_section_start == 0:
            valid_res = self._eval_split(valid_df, tag="valid_initial", step=0)
            valid_per_item = valid_res["per_item"]

            for qid, correct in valid_per_item.items():
                self.valid_cumulative_correct_map[qid] = bool(correct)

            total_valid_items = len(self.valid_cumulative_correct_map)
            total_valid_correct = sum(1 for x in self.valid_cumulative_correct_map.values() if x)
            cumulative_valid_acc = total_valid_correct / max(1, total_valid_items)

            logger.info(
                f"[Valid] Initial Cumulative Acc: {cumulative_valid_acc * 100:.2f}% "
                f"({total_valid_correct}/{total_valid_items})"
            )
            try:
                self.writer.add_scalar("Valid/Cumulative_Acc", cumulative_valid_acc, 0)
            except Exception:
                pass

        start_section = max(1, int(self._resume_section_start) + 1)
        for sec_idx in range(start_section, self.num_sections + 1):

            # ------------------------------
            if len(train_df) != 0:
                res = self._train_one_section(train_df, sec_idx)
                train_per_item = res["per_item"]

                # ------------------------------
                for qid, correct in train_per_item.items():
                    if qid not in self.train_cumulative_correct_map:
                        self.train_cumulative_correct_map[qid] = False
                    if correct:
                        self.train_cumulative_correct_map[qid] = True

                total_items = len(self.train_cumulative_correct_map)
                total_correct = sum(1 for x in self.train_cumulative_correct_map.values() if x)
                cumulative_acc = total_correct / max(1, total_items)

                logger.info(
                    f"[Train] Cumulative Acc after section {sec_idx}: {cumulative_acc * 100:.2f}% "
                    f"({total_correct}/{total_items})"
                )
                try:
                    self.writer.add_scalar("Train/Cumulative_Acc", cumulative_acc, sec_idx)
                except Exception:
                    pass

            self._save_cum_state(sec_idx + 1)

            # ------------------------------
            if len(valid_df) != 0:
                valid_res = self._eval_split(valid_df, tag=f"valid_sec_{sec_idx}", step=sec_idx)
                valid_per_item = valid_res["per_item"]

                for qid, correct in valid_per_item.items():
                    if qid not in self.valid_cumulative_correct_map:
                        self.valid_cumulative_correct_map[qid] = False
                    if correct:
                        self.valid_cumulative_correct_map[qid] = True

                total_valid_items = len(self.valid_cumulative_correct_map)
                total_valid_correct = sum(1 for x in self.valid_cumulative_correct_map.values() if x)
                cumulative_valid_acc = total_valid_correct / max(1, total_valid_items)

                logger.info(
                    f"[Valid] Cumulative Acc after section {sec_idx}: {cumulative_valid_acc * 100:.2f}% "
                    f"({total_valid_correct}/{total_valid_items})"
                )

                try:
                    self.writer.add_scalar("Valid/Cumulative_Acc", cumulative_valid_acc, sec_idx)
                except Exception:
                    pass

        try:
            self.writer.close()
        except Exception:
            pass
        try:
            with self._image_lock:
                self._persist_image_cache_unlocked()
        except Exception:
            logger.debug("Failed to persist image cache on shutdown", exc_info=True)
