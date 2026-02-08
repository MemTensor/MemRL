import sys
import tempfile
import unittest
from pathlib import Path

# Ensure local package imports work even when pytest import-mode/sys.path differs.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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
        self.enable_value_driven = False
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
                "metadata": {"outcome": "success", "task_id": "BigCodeBench/0"},
            }
        ]
        return ({"selected": selected, "simmax": 0.9}, [(task_description, 0.9)])

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
        retrieved_memory_queries=None,
        retrieved_memory_ids_list=None,
        metadatas=None,
    ):
        self.add_memories_calls.append(
            {
                "task_descriptions": list(task_descriptions or []),
                "trajectories": list(trajectories or []),
                "successes": list(successes or []),
                "retrieved_memory_queries": list(retrieved_memory_queries or []),
                "retrieved_memory_ids_list": list(retrieved_memory_ids_list or []),
                "metadatas": list(metadatas or []),
            }
        )
        return {}


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
                retrieve_threshold=0.42,
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

        # Retrieval uses MemoryService.retrieve_query() with the explicit threshold knob.
        self.assertEqual(len(mem.retrieve_query_calls), 1)
        self.assertEqual(mem.retrieve_query_calls[0]["k"], 3)
        self.assertAlmostEqual(mem.retrieve_query_calls[0]["threshold"], 0.42)

        # Train path writes memories via add_memories() and updates Q in batch.
        self.assertEqual(len(mem.add_memories_calls), 1)
        self.assertEqual(len(mem.update_values_calls), 1)
        call = mem.add_memories_calls[0]
        self.assertEqual(call["task_descriptions"], ["Write a function foo()."])
        # Trajectory is the model's raw output (preferred path).
        self.assertIn("```python", call["trajectories"][0])
        self.assertEqual(call["retrieved_memory_ids_list"], [["m0"]])

        # LLM got a system message containing retrieved memory context.
        self.assertEqual(len(llm.calls), 1)
        msgs = llm.calls[0]["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Retrieved Memory Context", msgs[0]["content"])

    def test_bcb_runner_fallback_trajectory_when_raw_output_empty(self) -> None:
        from memp.run.bcb_runner import BCBRunner, BCBSelection

        class _EmptyLLM:
            def __init__(self) -> None:
                self.calls = []

            def generate(self, *, messages, temperature, max_tokens):
                self.calls.append({"messages": messages})
                return ""  # trigger fallback trajectory

        class _MemWithIdOnly(_DummyMemoryService):
            def retrieve_query(self, task_description: str, k: int = 5, threshold: float = 0.0):
                self.retrieve_query_calls.append(
                    {"task_description": task_description, "k": k, "threshold": threshold}
                )
                selected = [
                    {
                        "id": "m_id_only",
                        "content": "X",
                        "similarity": 0.9,
                        "metadata": {"outcome": "failure", "task_id": "BigCodeBench/0"},
                    }
                ]
                return ({"selected": selected, "simmax": 0.9}, [(task_description, 0.9)])

        class _FailEvalRunner(BCBRunner):
            def _evaluate_one(self, *, task, code):
                return {
                    "task_id": task.get("task_id", "t"),
                    "status": "FAIL",
                    "error": "boom",
                }

        llm = _EmptyLLM()
        mem = _MemWithIdOnly()
        sel = BCBSelection(subset="hard", split="instruct")

        with tempfile.TemporaryDirectory() as td:
            r = _FailEvalRunner(
                root=Path(td),
                selection=sel,
                llm=llm,
                memory_service=mem,
                output_dir=td,
                model_name="dummy",
                num_epochs=1,
                retrieve_k=1,
                retrieve_threshold=0.1,
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
            self.assertEqual(out["pass"], 0)

        self.assertEqual(len(mem.add_memories_calls), 1)
        call = mem.add_memories_calls[0]
        self.assertEqual(call["retrieved_memory_ids_list"], [["m_id_only"]])
        traj = call["trajectories"][0]
        self.assertIn("[STEP 1] TASK PROMPT", traj)
        self.assertIn("Write a function foo().", traj)
        self.assertIn("[STEP 2] MEMORY RETRIEVAL", traj)
        self.assertIn("selected_memory_ids: ['m_id_only']", traj)
        self.assertIn("[STEP 4] EVALUATION RESULT", traj)
        self.assertIn("status: FAIL", traj)
        self.assertIn("boom", traj)


if __name__ == "__main__":
    unittest.main()
