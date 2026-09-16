"""Real service/retrieval/updater code with only external MemOS types replaced."""
import copy
import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


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
    # Restore local imports too, so stub-backed modules do not leak to other tests.
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


class SingleUpdateCacheTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        service = service_module.MemoryService.__new__(service_module.MemoryService)
        service.enable_value_driven = True
        service.rl_config = service_module.RLConfig(alpha=0.5, gamma=0.5, epsilon=0, topk=1)
        text_mem = self.store
        mos = SimpleNamespace(mem_cubes={"cube": SimpleNamespace(text_mem=text_mem)})
        service._q_updater = service_module.QValueUpdater(
            mos, "user", service.rl_config, default_cube_id="cube"
        )
        service._q_cache = {}
        service._q_cache_max_size = 10
        service.dict_memory = {"query": ["m", "other"]}
        service.query_embeddings = {"query": [1.0, 0.0]}
        service.embedding_provider = SimpleNamespace(embed=lambda texts: [[1., 0.] for _ in texts])
        service._mem_cache = copy.deepcopy(self.store.items)
        service.weight_sim = 0.0
        service.weight_q = 1.0
        service.use_z_score_normalization = False
        service.dedup_by_task_id = False
        self.service = service

    def retrieve(self):
        return self.service.retrieve_query("query", k=1)[0]

    def test_retrieval_sees_two_successful_updates(self):
        self.assertEqual(self.retrieve()["actions"], ["other"])
        self.assertEqual(self.service.update_value("m", 1, next_max_q=2), 1.0)
        self.assertEqual(self.store.items["m"].metadata["q_value"], 1.0)
        self.assertEqual(self.retrieve()["actions"], ["m"])
        self.assertEqual(self.service.update_value("m", -1), 0.0)
        result = self.retrieve()
        self.assertEqual(result["actions"], ["other"])
        self.assertEqual(next(c["q_estimate"] for c in result["candidates"] if c["memory_id"] == "m"), 0.0)

    def test_full_cache_matches_batch_fifo(self):
        initial = {f"old{i}": float(i) for i in range(10)}
        self.service._q_cache = initial.copy()
        self.service.update_value("m", 1)
        single = self.service._q_cache.copy()
        self.store.items["m"].metadata = {"q_value": 0.0}
        self.service._q_cache = initial.copy()
        self.assertEqual(self.service.update_values([True], [["m"]]), {"m": 0.5})
        self.assertEqual(single, self.service._q_cache)
        self.assertNotIn("old0", single)
        self.assertEqual(len(single), 10)
        self.assertEqual(single["m"], 0.5)

    def test_storage_failure_preserves_cache_and_retrieval(self):
        before_result = self.retrieve()
        before_cache = self.service._q_cache.copy()
        before_store = copy.deepcopy(self.store.items)
        self.store.fail = True
        with self.assertRaisesRegex(RuntimeError, "Failed to update Q-value: storage unavailable"):
            self.service.update_value("m", 1)
        self.assertEqual(self.service._q_cache, before_cache)
        self.assertEqual(self.store.items, before_store)
        self.assertEqual(self.retrieve(), before_result)

    def test_noops_preserve_cache(self):
        self.retrieve()
        before = self.service._q_cache.copy()
        updater = Mock()
        self.service._q_updater = updater
        self.assertIsNone(self.service.update_value(None, 1))
        self.service.enable_value_driven = False
        self.assertIsNone(self.service.update_value("m", 1))
        updater.update.assert_not_called()
        self.service.enable_value_driven = True
        self.service._q_updater = None
        self.assertIsNone(self.service.update_value("m", 1))
        self.assertEqual(self.service._q_cache, before)

    def test_none_result_preserves_cache(self):
        self.retrieve()
        before = self.service._q_cache.copy()
        self.service._q_updater = SimpleNamespace(update=Mock(return_value=None))
        self.assertIsNone(self.service.update_value("m", 1))
        self.assertEqual(self.service._q_cache, before)


if __name__ == "__main__":
    unittest.main()
