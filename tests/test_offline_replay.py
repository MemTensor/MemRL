"""Synthetic contract and actual retrieval replay; no model efficacy scoring."""
import copy
import hashlib
from types import SimpleNamespace
import unittest

from memrl.evaluation.replay import run_retrieval_replay, validate_memoryeval
from service_fixture import make_service


SOURCE = b"handwritten synthetic fixture v1\n"
STORAGE = {"backend": "in-memory-fixture", "version": "1", "integration": "fixture"}


def manifest():
    return {
        "schema_version": 1, "synthetic": True,
        "source": {"sha256": hashlib.sha256(SOURCE).hexdigest(), "format": "synthetic-test",
                   "generator": "handwritten", "version": "1", "seed": 0},
        "partitions": {"train": ["p-train"], "evaluation": ["p-eval"]},
        "tasks": [
            {"task_id": "train", "source_task_id": "train", "patient_id": "p-train",
             "split": "adapt", "query": "train", "trajectory": "synthetic target", "reward": 1},
            {"task_id": "eval", "source_task_id": 0, "patient_id": "p-eval",
             "split": "heldout", "query": "query", "trajectory": "unused target", "reward": 1},
        ],
    }


class OfflineReplayTests(unittest.TestCase):
    def service(self):
        return make_service({
            "self": SimpleNamespace(memory="self", metadata={"source_task_id": "0", "patient_id": "p-eval", "split": "heldout"}),
            "train": SimpleNamespace(memory="train", metadata={"task_id": "train", "source_task_id": "train", "patient_id": "p-train", "split": "adapt"}),
        }, {"query": [1., 0.], "self": [1., 0.], "train": [.8, .6]})

    def test_actual_retrieval_replay_is_deterministic_and_excludes_source(self):
        service = self.service()
        service.rl_config.epsilon = 1
        usage = {"provider": "synthetic-fixed-vectors", "model_calls": 0, "embedding_calls": 1}
        first = run_retrieval_replay(service, manifest(), storage=STORAGE, seed=7, k=1,
                                     provider_usage=usage, source_bytes=SOURCE)
        second = run_retrieval_replay(service, manifest(), storage=STORAGE, seed=7, k=1,
                                      provider_usage=usage, source_bytes=SOURCE)
        self.assertEqual(first, second)
        self.assertEqual(first["records"][0]["actions"], ["train"])
        self.assertEqual(first["records"][0]["excluded_task_ids"], ["0"])
        self.assertIsNone(first["records"][0]["failure_category"])
        self.assertTrue(first["provenance"]["source_sha256_verified"])
        self.assertIsNone(first["cost"]["cost_usd"])
        self.assertEqual(first["cost"]["model_calls"], 0)

    def test_unmeasured_usage_and_unverified_source_are_explicit(self):
        report = run_retrieval_replay(self.service(), manifest(), storage=STORAGE)
        self.assertFalse(report["provenance"]["source_sha256_verified"])
        self.assertEqual(report["cost"]["measurement"], "unavailable")
        self.assertIsNone(report["cost"]["model_calls"])
        self.assertIsNone(report["cost"]["cost_usd"])

    def test_partition_overlap_rejected_before_retrieval(self):
        data = manifest()
        data["partitions"]["evaluation"].append("p-train")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            run_retrieval_replay(None, data, storage=STORAGE)

    def test_patient_encounters_cannot_cross_evaluation_splits(self):
        data = manifest()
        data["tasks"].append(dict(data["tasks"][1], task_id="drift", split="drift"))
        with self.assertRaisesRegex(ValueError, "one split"):
            validate_memoryeval(data)

    def test_source_digest_mismatch_and_bad_usage_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            run_retrieval_replay(None, manifest(), storage=STORAGE, source_bytes=b"different")
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonnegative"):
                run_retrieval_replay(None, manifest(), storage=STORAGE, provider_usage={"cost_usd": value})

    def test_leakage_and_errors_are_failure_records(self):
        for category, metadata in (
            ("source_task_leakage", {"task_id": 0, "patient_id": "p-train", "split": "adapt"}),
            ("partition_leakage", {"task_id": "unrelated", "patient_id": "p-eval", "split": "heldout"}),
        ):
            with self.subTest(category=category):
                service = SimpleNamespace(retrieve_query=lambda *args, **kwargs: {
                    "actions": ["bad"], "candidates": [{"memory_id": "bad", "metadata": metadata}]
                })
                record = run_retrieval_replay(service, manifest(), storage=STORAGE)["records"][0]
                self.assertEqual(record["failure_category"], category)
                self.assertEqual(record["actions"], [])
        def fail(*args, **kwargs):
            raise OSError("synthetic unavailable storage")
        record = run_retrieval_replay(SimpleNamespace(retrieve_query=fail), manifest(), storage=STORAGE)["records"][0]
        self.assertEqual(record["failure_category"], "retrieval_error")
        self.assertEqual(record["error_type"], "OSError")
        self.assertEqual(record["actions"], [])

    def test_fixture_and_storage_version_changes_are_auditable(self):
        first = run_retrieval_replay(self.service(), manifest(), storage=STORAGE)
        data = copy.deepcopy(manifest())
        data["tasks"][1]["trajectory"] = "changed unused target"
        second = run_retrieval_replay(self.service(), data, storage=dict(STORAGE, version="2"))
        self.assertNotEqual(first["provenance"]["manifest_sha256"], second["provenance"]["manifest_sha256"])
        self.assertEqual(first["records_sha256"], second["records_sha256"])
        self.assertEqual(second["provenance"]["storage"]["version"], "2")

    def test_usage_callback_observes_actual_completed_retrieval(self):
        service = self.service()
        counts = {"embedding_calls": 0, "model_calls": 0}
        embed = service.embedding_provider.embed
        def counted(texts):
            counts["embedding_calls"] += 1
            return embed(texts)
        service.embedding_provider.embed = counted
        result = run_retrieval_replay(service, manifest(), storage=STORAGE,
                                     provider_usage=lambda: counts)
        self.assertEqual(result["cost"]["embedding_calls"], 1)
        self.assertEqual(result["cost"]["model_calls"], 0)

    def test_actions_outside_candidate_set_fail_closed(self):
        service = SimpleNamespace(retrieve_query=lambda *args, **kwargs: {
            "actions": ["unprovenanced"], "candidates": []
        })
        record = run_retrieval_replay(service, manifest(), storage=STORAGE)["records"][0]
        self.assertEqual(record["failure_category"], "invalid_retrieval_result")
        self.assertEqual(record["actions"], [])

    def test_missing_or_unknown_adaptation_origin_fails_closed(self):
        for origin in (None, "unknown-task"):
            metadata = {"patient_id": "p-train", "split": "adapt"}
            if origin is not None:
                metadata["source_task_id"] = origin
            service = SimpleNamespace(retrieve_query=lambda *args, **kwargs: {
                "actions": ["unknown"], "candidates": [{"memory_id": "unknown", "metadata": metadata}]
            })
            record = run_retrieval_replay(service, manifest(), storage=STORAGE)["records"][0]
            self.assertEqual(record["failure_category"], "memory_provenance_error")
            self.assertEqual(record["actions"], [])

    def test_actual_storage_failure_never_looks_like_successful_empty_result(self):
        for unavailable in ("all", "train", "none-result"):
            with self.subTest(unavailable=unavailable):
                service = self.service()
                service._mem_cache.clear()
                original_get = service.mos.get
                def get(**kwargs):
                    if unavailable == "none-result":
                        return None
                    if unavailable == "all" or kwargs["memory_id"] == unavailable:
                        raise OSError("synthetic storage unavailable")
                    return original_get(**kwargs)
                service.mos.get = get
                record = run_retrieval_replay(service, manifest(), storage=STORAGE)["records"][0]
                self.assertEqual(record["failure_category"], "retrieval_error")
                self.assertEqual(record["actions"], [])

    def test_legacy_source_aliases_and_first_failure_category_are_preserved(self):
        for alias in ("sample_index", "id"):
            metadata = {alias: 0, "patient_id": "p-train", "split": "adapt"}
            service = SimpleNamespace(retrieve_query=lambda *args, **kwargs: {
                "actions": ["outside-candidates"], "candidates": [{"memory_id": "leak", "metadata": metadata}]
            })
            data = manifest()
            data["tasks"][0]["source_task_id"] = 0
            record = run_retrieval_replay(service, data, storage=STORAGE)["records"][0]
            self.assertEqual(record["failure_category"], "source_task_leakage")
            self.assertEqual(record["actions"], [])


if __name__ == "__main__":
    unittest.main()
