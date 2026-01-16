import types
import unittest


class TestAddMemoryWrapper(unittest.TestCase):
    def test_add_memory_delegates_to_add_memories_and_returns_first_mem_id(self):
        # Import inside test to ensure it uses the activated conda env deps (memos, etc).
        from memp.service.memory_service import MemoryService

        svc = MemoryService.__new__(MemoryService)  # bypass heavy __init__

        captured = {}

        def fake_add_memories(
            self,
            *,
            task_descriptions,
            trajectories,
            successes,
            retrieved_memory_queries=None,
            retrieved_memory_ids_list=None,
            metadatas=None,
        ):
            captured["task_descriptions"] = task_descriptions
            captured["trajectories"] = trajectories
            captured["successes"] = successes
            captured["retrieved_memory_queries"] = retrieved_memory_queries
            captured["retrieved_memory_ids_list"] = retrieved_memory_ids_list
            captured["metadatas"] = metadatas
            # add_memories() currently returns list of (task_description, mem_id)
            return [("q", "mem_123")]

        svc.add_memories = types.MethodType(fake_add_memories, svc)

        mem_id = svc.add_memory(
            task_description="q",
            trajectory="traj",
            success=True,
            retrieved_memory_query=[("q_old", 0.9)],
            retrieved_memory_ids=["m0"],
            metadata={"source_benchmark": "X"},
        )

        self.assertEqual(mem_id, "mem_123")
        self.assertEqual(captured["task_descriptions"], ["q"])
        self.assertEqual(captured["trajectories"], ["traj"])
        self.assertEqual(captured["successes"], [True])
        self.assertEqual(captured["retrieved_memory_queries"], [[("q_old", 0.9)]])
        self.assertEqual(captured["retrieved_memory_ids_list"], [["m0"]])
        self.assertEqual(captured["metadatas"], [{"source_benchmark": "X"}])

    def test_add_memory_allows_metadata_none(self):
        from memp.service.memory_service import MemoryService

        svc = MemoryService.__new__(MemoryService)

        def fake_add_memories(self, **kwargs):
            return [("q", "mem_1")]

        svc.add_memories = types.MethodType(fake_add_memories, svc)

        mem_id = svc.add_memory(
            task_description="q",
            trajectory="traj",
            success=False,
            retrieved_memory_query=None,
            retrieved_memory_ids=None,
            metadata=None,
        )
        self.assertEqual(mem_id, "mem_1")


if __name__ == "__main__":
    unittest.main()

