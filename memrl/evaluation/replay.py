"""Replay synthetic retrieval inputs without invoking or scoring an LLM.

The caller supplies a MemoryService populated only from ``adapt`` tasks, storage
provenance and measured provider usage. Unknown usage/cost stays unknown. This
checks retrieval isolation; it does not measure diagnosis or agent efficacy.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Callable, Dict, Optional, Union

from ..utils.task_id import extract_task_id


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_memoryeval(manifest: Dict[str, Any]) -> None:
    """Validate the memoryeval v1 synthetic patient/task partition contract."""
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or manifest.get("synthetic") is not True:
        raise ValueError("memoryeval requires schema_version=1 and synthetic=true")
    source = manifest.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("source must be an object")
    digest = source.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("source.sha256 must be a lowercase SHA256 digest")
    for field in ("format", "generator", "version"):
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError(f"source.{field} is required")
    partitions = manifest.get("partitions", {})
    if not isinstance(partitions, dict):
        raise ValueError("partitions must be an object")
    groups = []
    for split in ("train", "evaluation"):
        values = partitions.get(split)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"partitions.{split} must contain patient ID strings")
        if len(values) != len(set(values)):
            raise ValueError(f"partitions.{split} contains duplicate patients")
        groups.append(set(values))
    train, evaluation = groups
    if train & evaluation:
        raise ValueError("train and evaluation patients must be disjoint")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty list")
    task_ids, patient_splits = set(), {}
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each task must be an object")
        for field in ("task_id", "source_task_id", "patient_id", "query"):
            value = task.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"task.{field} is required (numeric zero is valid)")
        if any(isinstance(task[field], bool) or not isinstance(task[field], (str, int)) for field in ("task_id", "source_task_id")):
            raise ValueError("task IDs must be strings or integers")
        task_id = str(task["task_id"])
        if task_id in task_ids:
            raise ValueError("task IDs must be unique")
        task_ids.add(task_id)
        split, patient = task.get("split"), task["patient_id"]
        if split not in {"adapt", "replay", "heldout", "drift"}:
            raise ValueError("unknown task split")
        if not isinstance(task["query"], str) or not isinstance(patient, str):
            raise ValueError("query and patient_id must be strings")
        expected = train if split == "adapt" else evaluation
        if patient not in expected:
            raise ValueError("task patient does not belong to its partition")
        if patient in patient_splits and patient_splits[patient] != split:
            raise ValueError("a patient's encounters must stay in one split")
        patient_splits[patient] = split
    _sha256(manifest)  # reject non-JSON and nonfinite values before any retrieval


def _metadata(candidate: Dict[str, Any]) -> Dict[str, Any]:
    value = candidate.get("metadata")
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return vars(value) if value is not None and hasattr(value, "__dict__") else {}


def _usage_report(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    usage = dict(value or {})
    for key in ("model_calls", "embedding_calls", "input_tokens", "output_tokens", "cost_usd"):
        measured = usage.get(key)
        if measured is not None and (isinstance(measured, bool) or not isinstance(measured, (int, float)) or not math.isfinite(measured) or measured < 0):
            raise ValueError(f"provider_usage.{key} must be finite and nonnegative or null")
        usage.setdefault(key, None)
    usage.setdefault("provider", "unknown")
    usage["measurement"] = "caller_reported" if value is not None else "unavailable"
    return usage


def run_retrieval_replay(
    service: Any,
    manifest: Dict[str, Any],
    *,
    storage: Dict[str, Any],
    seed: int = 0,
    k: int = 5,
    provider_usage: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]] = None,
    source_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Return deterministic isolation records and provenance for a prebuilt store.

    Only adapt memories may be in the store. Any unprovenanced/evaluation patient
    candidate fails the record closed. Retrieval errors remain errors, never a
    successful empty prediction. No task target, trajectory or reward is scored.
    ``source_bytes`` verifies the declared upstream fixture digest when supplied;
    otherwise provenance explicitly marks it unverified.
    A usage callback is read after retrieval so it can report actual run counters.
    """
    validate_memoryeval(manifest)
    if type(seed) is not int or type(k) is not int or k < 1:
        raise ValueError("seed must be an integer and k must be a positive integer")
    if not all(isinstance(storage.get(key), str) and storage[key] for key in ("backend", "version", "integration")):
        raise ValueError("storage backend, version and integration evidence are required")
    if source_bytes is not None and hashlib.sha256(source_bytes).hexdigest() != manifest["source"]["sha256"]:
        raise ValueError("source bytes do not match the declared SHA256")
    usage = None if callable(provider_usage) else _usage_report(provider_usage)
    train = set(manifest["partitions"]["train"])
    adapt_sources = {(str(task["source_task_id"]), task["patient_id"]) for task in manifest["tasks"] if task["split"] == "adapt"}
    rng, records = random.Random(seed), []
    for task in manifest["tasks"]:
        if task["split"] == "adapt":
            continue
        source_id = str(task["source_task_id"])
        record = {
            "task_id": str(task["task_id"]), "source_task_id": source_id,
            "patient_id": task["patient_id"], "split": task["split"],
            "query_sha256": _sha256(task["query"]),
            "excluded_task_ids": [source_id], "actions": [], "candidate_ids": [],
            "failure_category": None,
        }
        try:
            response = service.retrieve_query(task["query"], k=k, exclude_task_ids=[source_id],
                                              rng=rng, raise_on_load_error=True)
            result = response[0] if isinstance(response, tuple) else response
            candidates = result["candidates"]
            for candidate in candidates:
                metadata = _metadata(candidate)
                ids = (candidate.get("task_id"), extract_task_id(metadata), metadata.get("source_task_id"))
                if any(value is not None and str(value) == source_id for value in ids):
                    record["failure_category"] = "source_task_leakage"
                    break
                if metadata.get("split") != "adapt" or metadata.get("patient_id") not in train:
                    record["failure_category"] = "partition_leakage"
                    break
                origin = metadata.get("source_task_id")
                if origin is None:
                    origin = extract_task_id(metadata)
                if origin is None or (str(origin), metadata.get("patient_id")) not in adapt_sources:
                    record["failure_category"] = "memory_provenance_error"
                    break
            record["candidate_ids"] = [str(c["memory_id"]) for c in candidates]
            if record["failure_category"] is None and any(str(action) not in record["candidate_ids"] for action in result["actions"]):
                record["failure_category"] = "invalid_retrieval_result"
            if record["failure_category"] is None:
                record["actions"] = list(result["actions"])
        except Exception as error:
            record["failure_category"] = "retrieval_error"
            record["error_type"] = type(error).__name__
        records.append(record)
    if callable(provider_usage):
        usage = _usage_report(provider_usage())
    return {
        "schema_version": 1, "synthetic": True, "purpose": "retrieval_isolation",
        "provenance": {"source": manifest["source"], "source_sha256_verified": source_bytes is not None,
                       "manifest_sha256": _sha256(manifest), "storage": dict(storage)},
        "configuration": {"seed": seed, "k": k, "excluded_id_field": "source_task_id"},
        "cost": usage, "records": records, "records_sha256": _sha256(records),
    }
