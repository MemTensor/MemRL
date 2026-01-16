import unittest


from memp.service.memory_service import MemoryService
from memp.service.value_driven import RLConfig


class TestAddMemoriesQInit(unittest.TestCase):
    def _make_ms(self) -> MemoryService:
        # Avoid full MemOS initialization: we only need add_memories()' metadata shaping.
        ms = MemoryService.__new__(MemoryService)
        ms.rl_config = RLConfig(q_init_pos=1.0, q_init_neg=-1.0)
        ms.add_similarity_threshold = 0.8
        ms.dict_memory = {}
        ms.query_embeddings = {}
        ms.embedding_provider = None

        def _capture_update_memories(
            self,
            task_descriptions,
            trajectories,
            successes,
            retrieved_ids_list,
            metadatas=None,
        ):
            # Return (task_desc, mem_id) tuples like updater.update_batch does.
            self._captured_metadatas = metadatas
            return [(td, f"mem_{i}") for i, td in enumerate(task_descriptions)]

        ms.update_memories = _capture_update_memories.__get__(ms, MemoryService)
        return ms

    def test_sets_q_value_by_success_when_missing(self) -> None:
        ms = self._make_ms()
        ms.add_memories(
            task_descriptions=["t_success", "t_failure"],
            trajectories=["traj1", "traj2"],
            successes=[True, False],
            retrieved_memory_queries=[None, None],
            retrieved_memory_ids_list=[None, None],
            metadatas=[{}, {}],
        )

        metas = getattr(ms, "_captured_metadatas", None)
        self.assertIsInstance(metas, list)
        self.assertEqual(float(metas[0]["q_value"]), 1.0)
        self.assertEqual(float(metas[1]["q_value"]), -1.0)

    def test_does_not_override_upstream_q_value(self) -> None:
        ms = self._make_ms()
        ms.add_memories(
            task_descriptions=["t1", "t2"],
            trajectories=["traj1", "traj2"],
            successes=[True, False],
            retrieved_memory_queries=[None, None],
            retrieved_memory_ids_list=[None, None],
            metadatas=[{"q_value": 123.0}, {"q_value": -5.0}],
        )

        metas = getattr(ms, "_captured_metadatas", None)
        self.assertIsInstance(metas, list)
        self.assertEqual(float(metas[0]["q_value"]), 123.0)
        self.assertEqual(float(metas[1]["q_value"]), -5.0)


if __name__ == "__main__":
    unittest.main()

