## ADDED Requirements

### Requirement: Value-driven components follow the active mem cube
When `MemoryService` switches its active MemOS cube (by updating `default_cube_id`), all value-driven subcomponents that read/write memories MUST use the same active `mem_cube_id`.

#### Scenario: Resume-from-checkpoint aligns cube bindings
- **WHEN** `MemoryService.load_checkpoint_snapshot(snapshot_root)` completes successfully and sets `MemoryService.default_cube_id` to the loaded snapshot cube id
- **THEN** `MemoryService._q_updater.default_cube_id` (when present) MUST equal `MemoryService.default_cube_id`
- **THEN** `MemoryService._curator.default_cube_id` (when present) MUST equal `MemoryService.default_cube_id`

#### Scenario: Switching to a historical cube aligns cube bindings
- **WHEN** `MemoryService.switch_to_cube_timestamp(timestamp)` completes successfully and sets `MemoryService.default_cube_id` to the selected timestamp cube id
- **THEN** `MemoryService._q_updater.default_cube_id` (when present) MUST equal `MemoryService.default_cube_id`
- **THEN** `MemoryService._curator.default_cube_id` (when present) MUST equal `MemoryService.default_cube_id`

### Requirement: Cube binding alignment is safe when value-driven is disabled
If value-driven features are disabled, cube switching MUST NOT require value-driven subcomponents to exist, and cube-switching operations MUST complete without raising errors due to missing `_q_updater` / `_curator`.

#### Scenario: Resume without value-driven components
- **WHEN** `MemoryService.enable_value_driven` is `false` and `MemoryService.load_checkpoint_snapshot(snapshot_root)` completes successfully
- **THEN** the operation MUST NOT fail due to missing value-driven subcomponents

