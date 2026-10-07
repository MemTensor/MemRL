"""Behavioral exclusion regressions on actual retrieval/scoring code."""
import random
from types import SimpleNamespace
import unittest

from service_fixture import make_service


class TaskExclusionTests(unittest.TestCase):
    def setUp(self):
        self.items = {
            "self": SimpleNamespace(memory="same", metadata={"task_id": 0, "q_value": 99}),
            "alias": SimpleNamespace(memory="same", metadata={"task_id": "other", "source_task_id": "0"}),
            "eligible": SimpleNamespace(memory="next", metadata={"task_id": "train", "q_value": 0}),
            "unknown": SimpleNamespace(memory="last", metadata={"q_value": 0}),
        }
        self.service = make_service(self.items, {
            "query": [1., 0.], "same": [1., 0.], "next": [.8, .6], "last": [.6, .8]
        })

    def test_exclusion_refills_top_k_and_preserves_default(self):
        self.assertEqual(self.service.retrieve_query("query", k=1)[0]["actions"], ["self"])
        result, similarities = self.service.retrieve_query("query", k=1, exclude_task_ids=[0])
        self.assertEqual(result["actions"], ["eligible"])
        self.assertEqual([c["memory_id"] for c in result["candidates"]], ["eligible"])
        self.assertEqual(similarities, [("next", .8)])

    def test_excluded_memories_never_reach_exploration_or_normalization(self):
        self.service.rl_config.epsilon = 1
        self.service.rl_config.topk = 3
        for seed in range(5):
            result, _ = self.service.retrieve_query(
                "query", k=3, exclude_task_ids=["0"], rng=random.Random(seed)
            )
            self.assertEqual(set(result["actions"]), {"eligible", "unknown"})
            self.assertNotIn("self", [c["memory_id"] for c in result["candidates"]])

    def test_all_excluded_returns_empty_without_self_fallback(self):
        self.service.dict_memory = {"same": ["self", "alias"]}
        result, similarities = self.service.retrieve_query("query", exclude_task_ids=[0])
        self.assertEqual(result["actions"], [])
        self.assertEqual(result["candidates"], [])
        self.assertEqual(similarities, [])

    def test_rng_is_local_and_reproducible(self):
        self.service.rl_config.epsilon = 1
        state = random.getstate()
        first = self.service.retrieve_query("query", k=3, rng=random.Random(9))
        second = self.service.retrieve_query("query", k=3, rng=random.Random(9))
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), state)

    def test_optional_api_empty_paths_keep_pair_shape(self):
        result, similarities = self.service.retrieve_query("query", threshold=2, exclude_task_ids=[0])
        self.assertEqual(result["actions"], [])
        self.assertEqual(similarities, [])
        self.service.rl_config.q_min_threshold = 1000
        result, similarities = self.service.retrieve_query("query", exclude_task_ids=[0])
        self.assertEqual(result["actions"], [])
        self.assertEqual(len(similarities), 2)
        self.service.dict_memory.clear()
        result, similarities = self.service.retrieve_query("query", exclude_task_ids=[])
        self.assertEqual(result["actions"], [])
        self.assertEqual(similarities, [])
        # The historical default API's empty response stays backward compatible.
        self.assertIsInstance(self.service.retrieve_query("query"), dict)


if __name__ == "__main__":
    unittest.main()
