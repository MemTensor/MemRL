# Proposal: Fix `MemoryService.add_memory()` by Converting It to a Thin Wrapper

Date: 2026-01-15  
Scope: `memp/service/memory_service.py` (single-method behavior change)  
Related: `code_review/memp_service_review.md`

## Background / Problem

The service currently has two entrypoints for adding memories:

- `MemoryService.add_memory(...)` (single item)
- `MemoryService.add_memories(...)` (batch)

In the current code, the **single-item** API is effectively broken and risky:

- It mutates `metadata` with `metadata |= {...}` which crashes when `metadata is None`.
- It references `self.rl_config.q_init`, which does not exist in the current `RLConfig` definitions (they use `q_init_pos/q_init_neg`).
- Repository-wide search indicates there are **no call sites** of `add_memory(...)` today, meaning it is currently “dead” but still a future foot-gun.

We want to keep the public API surface (do not delete), but make it safe and consistent with the batch path.

## Goal

Make `add_memory(...)` a **thin wrapper** around `add_memories(...)`:

- No behavior divergence between single-item and batch code paths.
- Avoid all current crashes (`metadata=None`, nonexistent RL fields).
- Preserve existing signature and return type (`Optional[str]` memory_id).
- No warnings (per decision: compatibility mode).

## Non-Goals

- Refactor/repair retriever implementations (`memp/service/retrievers.py`) in this change.
- Change memory semantics (bucket matching, RL/Q update logic, etc.).
- Introduce new dependencies (e.g., pytest); we will use `unittest`.

## Proposed Change (Design)

### New `add_memory` behavior

`add_memory(...)` will:

1. Build a 1-element batch payload:
   - `task_descriptions=[task_description]`
   - `trajectories=[trajectory]`
   - `successes=[bool(success)]`
   - `retrieved_memory_queries=[retrieved_memory_query]`
   - `retrieved_memory_ids_list=[retrieved_memory_ids]`
   - `metadatas=[metadata]`
2. Call `self.add_memories(...)`.
3. Extract the first result’s `mem_id` and return it.

### Return-shape handling

Today, `add_memories(...)` returns a list of `(task_description, mem_id)` tuples (see its internal usage in runners).

The wrapper will:

- Return `None` if results is empty.
- Return the second element of the first tuple if it matches that shape.
- Fail closed (return `None`) if the return shape is unexpected.

### Error handling

Maintain the current `try/except` behavior and log via the existing `[add_memory] Error: ...` print path (to keep changes minimal and avoid changing logging policy).

## Test Plan (TDD)

Because `pytest` is not available in the `memory` conda env, we use `unittest`.

Add a focused unit test that does not require initializing MemOS:

- Create `MemoryService` instance via `MemoryService.__new__(MemoryService)` (bypass `__init__`).
- Monkeypatch `add_memories` with a stub that records arguments and returns `[("q", "mem_123")]`.
- Assert:
  - `add_memory()` returns `"mem_123"`.
  - The wrapper forwards args correctly.
  - `metadata=None` is accepted and does not crash.

File: `tests/test_add_memory_wrapper.py` (already added; currently failing before implementation).

Run command (in `conda activate memory` environment):

```bash
python -m unittest -q tests.test_add_memory_wrapper
```

Expected:
- Before code change: tests fail because current `add_memory` crashes and returns `None`.
- After code change: tests pass.

## Rollout / Risk

Risk is low:
- No known call sites today.
- Implementation is a wrapper; primary behavior stays defined by `add_memories`.
- If any external user code relies on the old (broken) behavior, this is still a strict improvement (previously it crashed / returned None).

## Acceptance Criteria

- `python -m unittest -q tests.test_add_memory_wrapper` passes under `conda activate memory`.
- `add_memory(metadata=None)` no longer crashes.
- `add_memory(...)` no longer references `self.rl_config.q_init`.
- No changes to `add_memories(...)` semantics.

