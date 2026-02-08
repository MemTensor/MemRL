## Why

Resume-from-checkpoint loads a snapshot mem cube and switches `MemoryService.default_cube_id`, but value-driven subcomponents created during `MemoryService.__init__` (e.g., the Q-value updater / curator) can remain bound to the pre-resume cube id. This causes reads/updates to hit the wrong cube and produces repeated `Memory with ID <uuid> not found` errors.

## What Changes

- Ensure `MemoryService` has a single source of truth for the active mem cube id, and that all subcomponents that talk to MemOS are re-bound when the active cube changes.
- On `load_checkpoint_snapshot(...)`, after switching to the snapshot cube, synchronize value-driven components (e.g., `_q_updater`, `_curator`) to use the new `default_cube_id`.
- On other cube-switching paths (e.g., `switch_to_cube_timestamp(...)`), apply the same synchronization logic to prevent future mismatches.
- Add focused tests/verification to prevent regressions (resume + RL/Q-update path should not attempt `get()` from a different cube).

## Capabilities

### New Capabilities

- `active-cube-alignment`: When `MemoryService` switches to a different mem cube (via checkpoint resume or explicit cube switching), all internal components that read/write memories must operate against the same active `mem_cube_id`.

### Modified Capabilities

<!-- none -->

## Impact

- `memp/service/memory_service.py`: checkpoint loading and cube switching must re-bind value-driven components after updating `default_cube_id`.
- `memp/service/value_driven.py`: no API changes expected, but its `default_cube_id` must be updated when the active cube changes.
- Runners that use resume + RL updates (e.g., LLB) should stop emitting large volumes of `Memory with ID ... not found` after resume.

