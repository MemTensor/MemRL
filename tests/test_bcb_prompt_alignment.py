import sys
from pathlib import Path

# Ensure local package imports work even when pytest import-mode/sys.path differs.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class _FakeLLM:
    def __init__(self) -> None:
        self.last_messages = None

    def generate(self, *args, **kwargs):
        # MemRL runners call with keyword arguments.
        self.last_messages = kwargs.get("messages") if "messages" in kwargs else (args[0] if args else None)
        return "```python\nprint('hi')\n```"


class _DummyMem:
    rl_config = None


def _meta_full_content(meta) -> str:
    if meta is None:
        return ""
    if hasattr(meta, "model_dump"):
        d = meta.model_dump()
        if isinstance(d, dict) and d.get("full_content"):
            return str(d["full_content"])
    extra = getattr(meta, "model_extra", None)
    if isinstance(extra, dict) and extra.get("full_content"):
        return str(extra["full_content"])
    return ""


def test_bcb_generate_code_includes_memoryrl_system_prompt_and_context_format():
    from memrl.run.bcb_runner import BCBRunner, BCBSelection, DEFAULT_SYSTEM_PROMPT

    llm = _FakeLLM()
    repo_root = Path(__file__).resolve().parents[1]
    runner = BCBRunner(
        root=".",
        selection=BCBSelection(),
        llm=llm,
        memory_service=_DummyMem(),
        output_dir="./tmp_test_out",
        model_name="test",
        num_epochs=1,
        bcb_repo=str(repo_root / "3rdparty" / "bigcodebench-main"),
    )

    mems = [
        {
            "content": "[MEMORY TYPE] SUCCESS_PROCEDURE\n[TASK]\nfoo\n",
            "metadata": {"outcome": "success", "task_id": "BigCodeBench/1"},
        }
    ]
    mem_ctx = runner._format_memory_context(mems)
    out = runner._generate_code("USER_PROMPT", memory_context=mem_ctx)
    assert "print('hi')" in out

    msgs = llm.last_messages
    assert isinstance(msgs, list) and len(msgs) >= 2
    assert msgs[0]["role"] == "system"
    assert DEFAULT_SYSTEM_PROMPT.strip() in msgs[0]["content"]
    assert "# Relevant Code Examples from Memory" in msgs[0]["content"]
    assert "## Example 1 [SUCCESS]" in msgs[0]["content"]
    assert "Task: BigCodeBench/1" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "USER_PROMPT"


def test_adjustment_failure_memory_does_not_store_full_trajectory():
    # NOTE: We intentionally do not import memrl.service.* here because the test
    # environment may not have the optional MemOS dependency installed.
    #
    # What we ultimately care about for BCB prompt alignment is that failure
    # memories injected into the LLM context do NOT contain the full failed
    # trajectory. In MemRL we enforce this at injection time for BCB.
    from memrl.run.bcb_runner import BCBRunner

    failed_traj = "FAILED_TRAJECTORY_SHOULD_NOT_BE_STORED"
    legacy_failure_blob = (
        "TASK REFLECTION:\n"
        "Task: do something\n\n"
        f"{failed_traj}\n"
        "Reflection: Root Cause: X; Pattern: Y; Correct: Z.\n"
    )
    coerced = BCBRunner._coerce_bcb_memory_content(
        raw_content=legacy_failure_blob,
        outcome="failure",
        task_description="do something",
    )
    assert "FAILURE_REFLECTION" in coerced
    assert "Root Cause" in coerced
    assert failed_traj not in coerced
