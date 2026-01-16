import tempfile
import unittest
from pathlib import Path


class _DummyLLM:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, *, messages, temperature, max_tokens):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        # Return fenced python to exercise extraction.
        return "```python\nprint('ok')\n```"


class _DummyRLConfig:
    sim_threshold = 0.42
    q_init_pos = 1.0
    q_init_neg = 0.0


class _DummyMemoryService:
    def __init__(self) -> None:
        self.rl_config = _DummyRLConfig()
        self.retrieve_query_calls = []
        self.update_values_calls = []
        self.add_memories_calls = []

    def retrieve_query(self, task_description: str, k: int = 5, threshold: float = 0.0):
        self.retrieve_query_calls.append(
            {"task_description": task_description, "k": k, "threshold": threshold}
        )
        selected = [
            {
                "memory_id": "m0",
                "content": "Task: X\n\nDo Y",
                "similarity": 0.9,
            }
        ]
        topk_queries = [("q0", 0.9)]
        return {"selected": selected, "actions": ["m0"], "candidates": selected, "simmax": 0.9}, topk_queries

    def update_values(self, successes, retrieved_ids_list):
        self.update_values_calls.append(
            {"successes": successes, "retrieved_ids_list": retrieved_ids_list}
        )
        return {}

    def add_memories(
        self,
        *,
        task_descriptions,
        trajectories,
        successes,
        retrieved_memory_queries,
        retrieved_memory_ids_list,
        metadatas,
    ):
        self.add_memories_calls.append(
            {
                # Copy lists because the runner reuses & clears buffers after flushing.
                "task_descriptions": list(task_descriptions),
                "trajectories": list(trajectories),
                "successes": list(successes),
                "retrieved_memory_queries": list(retrieved_memory_queries),
                "retrieved_memory_ids_list": list(retrieved_memory_ids_list),
                "metadatas": list(metadatas),
            }
        )
        return []


class TestBCBAlignment(unittest.TestCase):
    def test_bcb_runner_uses_retrieve_query_and_add_memories(self) -> None:
        from memp.run.bcb_runner import BCBRunner, BCBSelection

        class _NoEvalRunner(BCBRunner):
            def _evaluate_one(self, *, task, code):
                return {"task_id": task.get("task_id", "t"), "status": "PASS"}

        llm = _DummyLLM()
        mem = _DummyMemoryService()
        sel = BCBSelection(subset="hard", split="instruct")

        with tempfile.TemporaryDirectory() as td:
            r = _NoEvalRunner(
                root=Path(td),
                selection=sel,
                llm=llm,
                memory_service=mem,
                output_dir=td,
                model_name="dummy",
                num_epochs=1,
                retrieve_k=3,
                retrieve_threshold=0.99,  # legacy arg, should not affect threshold used
                rl_enabled=True,
                bcb_repo=str(Path(__file__).resolve().parents[1] / "3rdparty" / "bigcodebench-main"),
            )
            r._problems = {
                "BigCodeBench/1": {
                    "task_id": "BigCodeBench/1",
                    "instruct_prompt": "Write a function foo().",
                    "complete_prompt": "def foo():\n    ...",
                }
            }

            out = r._run_phase(
                epoch=1,
                phase="train",
                task_ids=["BigCodeBench/1"],
                epoch_dir=td,
                update_memory=True,
            )
            self.assertEqual(out["pass"], 1)

        # Retrieval uses the unified threshold knob.
        self.assertEqual(len(mem.retrieve_query_calls), 1)
        self.assertEqual(mem.retrieve_query_calls[0]["k"], 3)
        self.assertAlmostEqual(mem.retrieve_query_calls[0]["threshold"], 0.42)

        # Train path updates Q and adds memory via add_memories() (dict_memory alignment).
        self.assertEqual(len(mem.update_values_calls), 1)
        self.assertEqual(len(mem.add_memories_calls), 1)
        call = mem.add_memories_calls[0]
        self.assertEqual(call["task_descriptions"], ["Write a function foo()."])
        self.assertEqual(call["retrieved_memory_ids_list"], [["m0"]])
        self.assertEqual(call["retrieved_memory_queries"], [[("q0", 0.9)]])

        # LLM got a system message containing retrieved memory context.
        self.assertEqual(len(llm.calls), 1)
        msgs = llm.calls[0]["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Retrieved Memory Context", msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()
