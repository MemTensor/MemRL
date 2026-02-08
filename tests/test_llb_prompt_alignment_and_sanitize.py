from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure local package imports work even when pytest import-mode/sys.path differs.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _opt_import(name: str):
    """Best-effort import that turns missing modules into assertion failures (not test errors)."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        # Preserve the underlying exception for debugging in assertion messages.
        return ("__missing__", name, repr(e))


def _assert_imported(mod, name: str):
    if isinstance(mod, tuple) and mod[:2] == ("__missing__", name):
        raise AssertionError(f"failed to import {name}: {mod[2]}")
    assert mod is not None
    return mod


def test_llb_prompt_modules_exist() -> None:
    # RED: these modules should exist after we align MemRL with memory_rl/dev/feat-mdp-llb.
    _assert_imported(_opt_import("memrl.lifelongbench_eval.prompts"), "memrl.lifelongbench_eval.prompts")
    _assert_imported(_opt_import("memrl.lifelongbench_eval.sanitize"), "memrl.lifelongbench_eval.sanitize")
    _assert_imported(_opt_import("memrl.lifelongbench_eval.memory_context"), "memrl.lifelongbench_eval.memory_context")


def test_llb_system_prompt_is_task_consistent() -> None:
    prompts = _opt_import("memrl.lifelongbench_eval.prompts")
    prompts = _assert_imported(prompts, "memrl.lifelongbench_eval.prompts")

    db = prompts.build_llb_system_prompt(task="db_bench")
    os = prompts.build_llb_system_prompt(task="os_interaction")

    assert db == (
        prompts.DEFAULT_SYSTEM_PROMPT
        + "\n\n"
        + prompts.LLB_DB_STRICT_OUTPUT_FORMAT_CONSTRAINT
    )
    assert os == (
        prompts.DEFAULT_SYSTEM_PROMPT
        + "\n\n"
        + prompts.LLB_OS_STRICT_OUTPUT_FORMAT_CONSTRAINT
    )


def test_llb_memory_sanitizer_strips_env_preamble() -> None:
    sanitize = _opt_import("memrl.lifelongbench_eval.sanitize")
    sanitize = _assert_imported(sanitize, "memrl.lifelongbench_eval.sanitize")

    raw = "\n".join(
        [
            "user: I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.",
            "assistant: ok",
            "user: real task",
            "assistant: done",
        ]
    )
    cleaned = sanitize.sanitize_llb_env_preamble(raw)
    assert "I will ask you a question" not in cleaned
    assert cleaned.startswith("assistant: ok")

    embedded = "\n".join(
        [
            "Task: demo",
            "",
            "TRAJECTORY:",
            "user: I will provide you with a task to perform on a Linux (Ubuntu) system.",
            "assistant: ok",
            "user: real task",
            "assistant: done",
        ]
    )
    cleaned2 = sanitize.sanitize_llb_env_preamble(embedded)
    assert "Linux (Ubuntu) system" not in cleaned2
    assert cleaned2.startswith("Task: demo")

    unchanged = "user: please do X\nassistant: ok"
    assert sanitize.sanitize_llb_env_preamble(unchanged) == unchanged


def test_llb_memory_context_no_env_preamble() -> None:
    memory_context = _opt_import("memrl.lifelongbench_eval.memory_context")
    memory_context = _assert_imported(memory_context, "memrl.lifelongbench_eval.memory_context")

    blob = "\n".join(
        [
            "Task: demo",
            "",
            "TRAJECTORY:",
            "user: I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.",
            "assistant: ok",
            "user: real task",
            "assistant: done",
        ]
    )
    ctx = memory_context.format_llb_memory_context(
        {"successed": [{"metadata": {}, "content": blob}]},
        task="db_bench",
    )
    assert "I will ask you a question" not in ctx


def test_llb_prompt_order_system_before_memory_and_constraints_last() -> None:
    prompts = _opt_import("memrl.lifelongbench_eval.prompts")
    memory_context = _opt_import("memrl.lifelongbench_eval.memory_context")
    prompts = _assert_imported(prompts, "memrl.lifelongbench_eval.prompts")
    memory_context = _assert_imported(memory_context, "memrl.lifelongbench_eval.memory_context")

    ctx = memory_context.format_llb_memory_context(
        {"successed": [{"metadata": {}, "content": "assistant: ok"}]},
        task="db_bench",
    )
    full = prompts.build_llb_prompt_with_memory(
        task="db_bench",
        base_prompt=prompts.DEFAULT_SYSTEM_PROMPT,
        memory_context=ctx,
    )

    assert full.find("You are an execution-focused AI agent") != -1
    assert full.find("[Retrieved Memory Context]") != -1
    assert full.find("STRICT OUTPUT FORMAT (LLB-DB") != -1
    assert full.find("You are an execution-focused AI agent") < full.find(
        "[Retrieved Memory Context]"
    )
    assert full.find("[Retrieved Memory Context]") < full.find(
        "STRICT OUTPUT FORMAT (LLB-DB"
    )
    assert full.count("STRICT OUTPUT FORMAT") == 1
    assert full.rstrip().endswith(
        prompts.LLB_DB_STRICT_OUTPUT_FORMAT_CONSTRAINT.splitlines()[-1]
    )

    ctx_os = memory_context.format_llb_memory_context(
        {"successed": [{"metadata": {}, "content": "assistant: ok"}]},
        task="os_interaction",
    )
    full_os = prompts.build_llb_prompt_with_memory(
        task="os_interaction",
        base_prompt=prompts.DEFAULT_SYSTEM_PROMPT,
        memory_context=ctx_os,
    )
    assert full_os.find("[Retrieved Memory Context]") < full_os.find(
        "STRICT OUTPUT FORMAT (LLB-OS"
    )
    assert full_os.count("STRICT OUTPUT FORMAT") == 1
    assert full_os.rstrip().endswith(
        prompts.LLB_OS_STRICT_OUTPUT_FORMAT_CONSTRAINT.splitlines()[-1]
    )
