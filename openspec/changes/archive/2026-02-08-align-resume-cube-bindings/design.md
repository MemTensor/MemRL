## Context

`MemoryService` creates and registers a timestamped MemOS mem cube during initialization (`default_cube_id = cube_<user>_<ts>`). When resuming from a checkpoint, `MemoryService.load_checkpoint_snapshot(...)` loads a snapshot cube and switches `MemoryService.default_cube_id` to the snapshot cube id.

However, value-driven subcomponents (notably `QValueUpdater` and `MemoryCurator`) are created in `MemoryService.__init__` and store their own `default_cube_id`. After resume, these subcomponents can remain bound to the pre-resume cube id while the rest of the system uses the snapshot cube id. This causes MemOS `get/update` calls to hit the wrong cube and results in repeated `Memory with ID ... not found` errors during RL/Q-update paths.

Constraints:
- Keep the change minimal and localized (方案 A): do not redesign retrieval, do not unload existing cubes, and do not introduce new persistence formats.
- Preserve existing public APIs; only fix internal cube-binding correctness.

## Goals / Non-Goals

**Goals:**
- Ensure that after `load_checkpoint_snapshot(...)` completes, all value-driven components that call into MemOS operate on the same active `mem_cube_id` as `MemoryService.default_cube_id`.
- Apply the same alignment when `MemoryService` switches cubes via other supported entry points (e.g., `switch_to_cube_timestamp(...)`).
- Add unit-level regression coverage to prevent cube-binding drift from reappearing.

**Non-Goals:**
- Changing the retrieval strategy behavior (e.g., forcing `mos.search` to scope to a specific cube).
- Unregistering/unloading other cubes after resume (can be considered later if cross-cube retrieval becomes a separate issue).
- Addressing unrelated `None` memory id indexing issues unless required for the cube-binding fix.

## Decisions

1) Centralize “active cube switch” side-effects in `MemoryService`.

- Decision: introduce a small internal helper (method or function) that synchronizes cube-bound subcomponents to match `MemoryService.default_cube_id`:
  - `MemoryService._q_updater.default_cube_id` (if present)
  - `MemoryService._curator.default_cube_id` and `MemoryService._curator.q_updater.default_cube_id` (if present)
- Rationale: the bug occurs because `default_cube_id` is stored in multiple places. A single helper reduces the chance of future switch paths forgetting to rebind.
- Alternatives:
  - Recreate `_q_updater` / `_curator` objects after resume. This is also viable, but updating the cube id in-place is simpler, avoids any subtle state loss, and matches the minimal-scope goal.

2) Apply the helper in the two cube-switching entry points.

- `load_checkpoint_snapshot(...)`: after registering the snapshot cube and setting `default_cube_id`, call the helper to rebind subcomponents.
- `switch_to_cube_timestamp(...)`: after switching `default_cube_id`, call the helper as well.

3) Regression testing approach.

- Add a unit test that does not depend on a live MemOS backend by testing the helper logic directly:
  - Create a lightweight dummy object with `default_cube_id` fields matching `QValueUpdater`/`MemoryCurator` shape.
  - Verify that after invoking the helper, all component `default_cube_id` values match `MemoryService.default_cube_id`.
- Rationale: avoids brittle integration tests and keeps CI fast.

## Risks / Trade-offs

- [Risk] Other code paths may create additional cube-bound components in the future and forget to include them in the helper. -> Mitigation: keep the helper narrowly documented as the mandatory place to add new cube-bound components; add a test that asserts all known components are synchronized.
- [Trade-off] Retrieval (`mos.search`) is not scoped to a cube id today, so cross-cube search results could still exist. -> Mitigation: treat as separate follow-up; this change focuses on fixing the value-driven update path mismatch that causes the observed burst of errors.
