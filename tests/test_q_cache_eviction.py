"""Q-cache eviction regressions using the real MemoryService implementation."""
import copy
import importlib
import json
import tempfile
import sys
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def load_service():
    modules = {}
    names = {
        "memos.configs.mem_os": ["MOSConfig"],
        "memos.configs.mem_cube": ["GeneralMemCubeConfig"],
        "memos.mem_os.main": ["MOS"],
        "memos.mem_cube.general": ["GeneralMemCube"],
        "memos.memories.textual.item": ["TextualMemoryItem", "TextualMemoryMetadata"],
        "memos.utils": [],
    }
    for name, types in names.items():
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            path = ".".join(parts[:end])
            if path not in modules:
                modules[path] = ModuleType(path)
                modules[path].__path__ = []
        for type_name in types:
            setattr(modules[name], type_name, type(type_name, (), {}))
    with patch.dict(sys.modules, modules):
        return importlib.import_module("memrl.service.memory_service")


service_module = load_service()


class Store:
    def __init__(self):
        self.items = {
            key: SimpleNamespace(memory=key, metadata={"q_value": q})
            for key, q in (("m", 0.0), ("other", 0.25))
        }
        self.fail = False

    def get(self, memory_id):
        return copy.deepcopy(self.items[memory_id])

    def update(self, memory_id, value):
        if self.fail:
            raise OSError("storage unavailable")
        self.items[memory_id] = SimpleNamespace(
            memory=value["memory"], metadata=value["metadata"]
        )


class ReadOnlyMetadata:
    def __init__(self, q_value):
        self._q_value = q_value

    @property
    def q_value(self):
        return self._q_value

    @q_value.setter
    def q_value(self, value):
        raise TypeError("metadata is immutable")


class QCacheEvictionTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        service = service_module.MemoryService.__new__(service_module.MemoryService)
        service.enable_value_driven = True
        service.rl_config = service_module.RLConfig(alpha=0.5, gamma=0.5, epsilon=0, topk=2)
        mos = SimpleNamespace(mem_cubes={"cube": SimpleNamespace(text_mem=self.store)})
        mos.get = lambda **kwargs: self.store.get(kwargs["memory_id"])
        service.mos = mos
        service.default_cube_id = "cube"
        service.user_id = "user"
        service._db_gate = threading.BoundedSemaphore(1)
        service._q_updater = service_module.QValueUpdater(
            mos,
            "user", service.rl_config, default_cube_id="cube"
        )
        service._q_cache = {}
        service._q_cache_max_size = 1
        service._mem_cache_max_size = 10
        service.dict_memory = {"query": ["m", "other"]}
        service.query_embeddings = {"query": [1.0, 0.0]}
        service.embedding_provider = SimpleNamespace(embed=lambda texts: [[1., 0.] for _ in texts])
        service._mem_cache = copy.deepcopy(self.store.items)
        service.weight_sim = 0.0
        service.weight_q = 1.0
        service.use_z_score_normalization = False
        service.dedup_by_task_id = False
        self.service = service

    def retrieve_q(self, memory_id):
        result = self.service.retrieve_query("query", k=2)[0]
        return next(c["q_estimate"] for c in result["candidates"] if c["memory_id"] == memory_id)

    def test_single_update_survives_q_cache_eviction(self):
        self.service.retrieve_query("query", k=2)
        self.assertEqual(self.service.update_value("m", -1), -0.5)
        self.service.update_value("other", 0)
        self.assertEqual(self.store.items["m"].metadata["q_value"], -0.5)
        self.assertEqual(self.retrieve_q("m"), -0.5)

    def test_batch_update_survives_q_cache_eviction_with_zero(self):
        self.service.retrieve_query("query", k=2)
        self.assertEqual(self.service.update_values([True], [["m"]]), {"m": 0.5})
        self.service.update_values([False], [["other"]])
        self.assertEqual(self.store.items["m"].metadata["q_value"], 0.5)
        self.assertEqual(self.retrieve_q("m"), 0.5)

    def test_failed_update_does_not_change_metadata_or_cache(self):
        self.service.retrieve_query("query", k=2)
        before_cache = self.service._q_cache.copy()
        before_metadata = copy.deepcopy(self.service._mem_cache["m"].metadata)
        self.store.fail = True
        with self.assertRaisesRegex(RuntimeError, "Failed to update Q-value: storage unavailable"):
            self.service.update_value("m", 1)
        self.assertEqual(self.service._q_cache, before_cache)
        self.assertEqual(self.service._mem_cache["m"].metadata, before_metadata)

        self.store.fail = False
        self.service._q_cache = before_cache.copy()
        self.store.fail = True
        self.assertEqual(self.service.update_values([True], [["m"]]), {"m": None})
        self.assertEqual(self.service._q_cache, before_cache)
        self.assertEqual(self.service._mem_cache["m"].metadata, before_metadata)

    def test_q_cache_snapshot_round_trip(self):
        self.service._q_cache = {"m": -0.5, "other": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            self.service._persist_local_caches(directory)
            restored = service_module.MemoryService.__new__(service_module.MemoryService)
            restored._q_cache = {}
            self.assertTrue(restored._restore_local_caches(directory + "/local_cache"))
            self.assertEqual(restored._q_cache, self.service._q_cache)
            with open(Path(directory) / "local_cache" / "q_cache.json", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"m": -0.5, "other": 0.0})

    def test_rejected_metadata_assignment_invalidates_fallback(self):
        self.service.retrieve_query("query", k=2)
        self.service._mem_cache["m"].metadata = ReadOnlyMetadata(0.0)
        self.assertEqual(self.service.update_value("m", -1), -0.5)
        self.service.update_value("other", 0)
        self.assertNotIn("m", self.service._mem_cache)
        self.assertEqual(self.retrieve_q("m"), -0.5)
        self.assertIn("m", self.service._mem_cache)

    def test_first_q_update_replaces_missing_metadata_value(self):
        # Imported or legacy memories need not already have a q_value field.
        for metadata in (SimpleNamespace(full_content="synthetic"), None):
            with self.subTest(metadata=metadata):
                self.store.items["m"].metadata = {"full_content": "synthetic"}
                self.service._q_cache.clear()
                self.service._mem_cache["m"] = SimpleNamespace(
                    memory="m", metadata=metadata
                )
                self.assertEqual(self.service.update_value("m", -1), -0.5)
                self.service._q_cache.clear()  # force the metadata/storage fallback
                self.assertEqual(self.retrieve_q("m"), -0.5)


if __name__ == "__main__":
    unittest.main()
