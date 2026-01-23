import json
from pathlib import Path


def test_llb_jsonl_tracer_writes_one_line_per_task(tmp_path: Path):
    # Import should succeed after implementation.
    from memp.trace.llb_jsonl import LLBJsonlTracer
    from memp.trace.tracing_llm import TracingLLMProvider

    out = tmp_path / "trace.jsonl"
    tracer = LLBJsonlTracer(path=out, sample_filter="1")

    class FakeProvider:
        def generate(self, messages, **kwargs):
            return "ok"

    provider = TracingLLMProvider(FakeProvider(), tracer=tracer)

    with tracer.task(sample_index="s1", run_meta={"run_id": "r1"}, task_description="t"):
        # Simulate a single model call with system + user messages.
        provider.generate(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hi"},
            ],
            temperature=0.0,
        )

    # Tracer writes one JSON line for the task.
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["sample_index"] == "s1"
    assert obj["prompt"]["full_system_prompt"] == "SYS"
    assert obj["llm_calls"][0]["system_prompt_id"] == obj["prompt"]["system_prompt_id"]
    assert obj["llm_calls"][0]["messages_wo_system"] == [{"role": "user", "content": "hi"}]
    assert obj["llm_calls"][0]["response_text"] == "ok"


def test_llb_jsonl_tracer_sample_filter_limit(tmp_path: Path):
    from memp.trace.llb_jsonl import LLBJsonlTracer

    out = tmp_path / "trace.jsonl"
    tracer = LLBJsonlTracer(path=out, sample_filter="2")  # trace only 2 tasks

    def run_one(idx: str):
        with tracer.task(sample_index=idx, run_meta={"run_id": "r1"}, task_description="t"):
            pass

    run_one("a")
    run_one("b")
    run_one("c")  # should not be traced

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    got = [json.loads(x)["sample_index"] for x in lines]
    assert got == ["a", "b"]


def test_llb_jsonl_tracer_sample_filter_list(tmp_path: Path):
    from memp.trace.llb_jsonl import LLBJsonlTracer

    out = tmp_path / "trace.jsonl"
    tracer = LLBJsonlTracer(path=out, sample_filter="x, z")

    for idx in ["x", "y", "z"]:
        with tracer.task(sample_index=idx, run_meta={"run_id": "r1"}, task_description="t"):
            pass

    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["sample_index"] for x in lines] == ["x", "z"]

