## 1. Active Cube Binding Sync

- [x] 1.1 Add a small internal helper in `MemoryService` to sync cube-bound subcomponents (update `_q_updater` / `_curator` cube ids to match `default_cube_id`)
- [x] 1.2 Invoke the helper at the end of `load_checkpoint_snapshot(...)` after `default_cube_id` is set to the snapshot cube id
- [x] 1.3 Invoke the helper in `switch_to_cube_timestamp(...)` after `default_cube_id` is set to the timestamp cube id
- [x] 1.4 Add lightweight debug logging (once per switch) showing old/new `default_cube_id` and whether value-driven components were updated

## 2. Tests

- [x] 2.1 Add a unit test that verifies cube binding alignment after calling the helper (covers `_q_updater` and `_curator` when present)
- [x] 2.2 Add a unit test that verifies alignment logic is a no-op (and does not raise) when value-driven is disabled or components are missing

## 3. Verification

- [x] 3.1 Run the existing unit tests suite (or at least snapshot + new tests) to ensure no regressions
- [x] 3.2 Run a targeted unit test under the `memory` python environment to ensure cube bindings stay aligned after a cube switch (covers `_q_updater` / `_curator` sync)
