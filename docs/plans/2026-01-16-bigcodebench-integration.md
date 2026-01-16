# BigCodeBench (BCB) Multi-Epoch Memory Mode Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-class BigCodeBench (BCB) benchmark integration that runs multi-epoch memory training/validation using the current `memp.service.MemoryService`, with a unified entrypoint style (`run/run_bcb.py`) consistent with existing benchmarks.

**Architecture:** Introduce a new `BCBRunner` under `memp/run/` that orchestrates multi-epoch `train -> val` loops. Reuse most of the existing BCB-specific logic (prompting, decoding, evaluation via `bigcodebench.eval.untrusted_check`) by migrating it into `memp/bigcodebench_eval/`, but remove the old "self-initializing" service/provider code and inject `OpenAILLM/OpenAIEmbedder/MemoryService` from the unified script entrypoint.

**Tech Stack:** Python 3.10, existing `OpenAILLM/OpenAIEmbedder`, `MemoryService`, vendored `3rdparty/bigcodebench-main` for official evaluation.

---

### Task 1: Create/verify worktree-friendly repo hygiene

**Files:**
- Modify: `.gitignore`

**Step 1: Ensure `.worktrees/` is ignored**

Run: `git check-ignore -v .worktrees`
Expected: shows `.worktrees/` matched by `.gitignore`.

**Step 2: Allow committing `docs/` and `tests/` for open-source readiness**

Edit `.gitignore` to NOT ignore `docs/` and `tests/`.

**Step 3: Commit**

Run:
```bash
git add .gitignore
git commit -m "chore: track docs and tests for open-source"
```

---

### Task 2: Vendor BigCodeBench into current repo

**Files:**
- Create: `3rdparty/bigcodebench-main/` (vendored directory)

**Step 1: Copy the vendor tree**

Run (from repo root):
```bash
cp -a /mnt/public/code/jq/memory_rl/bigcodebench-main 3rdparty/bigcodebench-main
```

**Step 2: Add a minimal README**

Create `3rdparty/bigcodebench-main/README_MEMRL_VENDOR.md` documenting:
- source path used for vendoring
- expected upstream origin (BigCodeBench)
- how it is imported (sys.path insertion in our runner)

**Step 3: Commit**

Run:
```bash
git add 3rdparty/bigcodebench-main
git commit -m "vendor: add bigcodebench-main under 3rdparty"
```

---

### Task 3: Add BCB evaluation package (ported from old repo, trimmed)

**Files:**
- Create: `memp/bigcodebench_eval/__init__.py`
- Create: `memp/bigcodebench_eval/task_wrappers.py`
- Create: `memp/bigcodebench_eval/bcb_adapter.py`
- Create: `memp/bigcodebench_eval/eval_utils.py`

**Step 1: Port dataset utilities**

Copy/adapt from `/mnt/public/code/jq/memory_rl/memp/bigcodebench_eval/task_wrappers.py`, but ensure:
- default data dir is repo-local: `data/bigcodebench/...` (documented)
- error message gives explicit download instructions
- no dependency on old service/env helpers

**Step 2: Port decoder adapter**

Copy/adapt from `/mnt/public/code/jq/memory_rl/memp/bigcodebench_eval/bcb_adapter.py`, but ensure:
- import path for bigcodebench is `3rdparty/bigcodebench-main` (not repo root)
- the only MemoryService APIs used are:
  - `retrieve(...)` / `retrieve_value_aware(...)`
  - `update_memory(...)`
  - `update_values(...)` (NOT `update_values_batch`, which does not exist in current service)

**Step 3: Extract official eval wrapper**

Create `memp/bigcodebench_eval/eval_utils.py` containing:
- `_run_untrusted_check_with_hard_timeout(...)` (ported)
- any sanitize helper needed for consistent evaluation (ported if referenced)

**Step 4: Add a small unit test for path resolution**

Test: `tests/test_bcb_vendor_path.py`
- Verifies `eval_utils` resolves `3rdparty/bigcodebench-main` and can import `bigcodebench` when present.
- The test should skip (not fail) if the vendor dir is missing.

**Step 5: Commit**

Run:
```bash
git add memp/bigcodebench_eval tests
git commit -m "feat: add bigcodebench_eval helpers (adapter, eval utils, dataset wrappers)"
```

---

### Task 4: Implement multi-epoch BCB runner under `memp/run/`

**Files:**
- Create: `memp/run/bcb_runner.py`
- Modify: `memp/run/__init__.py`

**Step 1: Implement `BCBRunner` skeleton**

`BCBRunner` should accept:
- `root: Path`
- `memory_service: MemoryService`
- `llm_provider: OpenAILLM`
- `exp_name: str`
- `subset/split`
- `num_epochs: int`
- `train_ratio/seed`
- `retrieve_k`, `temperature`, `max_tokens`
- `output_dir`

**Step 2: Implement epoch loop**

For epoch `e=1..num_epochs`:
- Train phase: generate code for train tasks, evaluate, write memories for each task via `memory_service.update_memory(...)`
- Val phase: generate+evaluate val tasks; DO NOT update memory by default
- Save per-epoch metrics under:
  - `<run_dir>/epoch{e}/train/metrics.json`
  - `<run_dir>/epoch{e}/val/metrics.json`

**Step 3: Implement snapshots (current service API)**

At end of each epoch:
- Call `memory_service.save_checkpoint_snapshot(epoch_dir, ckpt_id=str(e))`
This should produce: `<epoch_dir>/snapshot/<e>/...`

At end of run:
- Call `memory_service.save_checkpoint_snapshot(run_dir, ckpt_id=\"final\")`

**Step 4: Add smoke test for snapshot call contract**

Test: `tests/test_bcb_snapshot_api.py`
- Create a tiny fake MemoryService (or monkeypatch) exposing `save_checkpoint_snapshot(target_ck_dir, ckpt_id)`
- Run a minimal `BCBRunner` with `num_epochs=2` and assert snapshot calls use the correct ckpt_id values.

**Step 5: Commit**

Run:
```bash
git add memp/run/bcb_runner.py memp/run/__init__.py tests
git commit -m "feat: add multi-epoch BCBRunner with per-epoch snapshots"
```

---

### Task 5: Add unified entrypoint `run/run_bcb.py`

**Files:**
- Create: `run/run_bcb.py`

**Step 1: Follow the existing benchmark script style**

Mirror `run/run_hle.py` patterns:
- Insert repo root into `sys.path`
- `--config` loads `MempConfig.from_yaml(...)`
- Build `OpenAILLM/OpenAIEmbedder`
- Write temp `mos_config.json` (sqlite user manager under a temp dir)
- Initialize `MemoryService` with:
  - `strategy_config=StrategyConfiguration(BuildStrategy(...), RetrieveStrategy(...), UpdateStrategy(...))`
  - `enable_value_driven=cfg.experiment.enable_value_driven`
  - `rl_config=cfg.rl_config`

**Step 2: Construct and run**

Instantiate `BCBRunner(...)` and call `run()`.

**Step 3: Commit**

Run:
```bash
git add run/run_bcb.py
git commit -m \"feat: add run_bcb entrypoint consistent with other benchmarks\"
```

---

### Task 6: Verification and docs

**Files:**
- Modify: `README.md`

**Step 1: Add usage snippet**

Document:
```bash
python run/run_bcb.py --config <your_yaml> --subset hard --split instruct --epochs 3
```

**Step 2: Run minimal local checks**

Run:
```bash
python -m compileall memp run
pytest -q -o addopts=
```
Expected:
- compileall succeeds
- pytest passes (or if thirdparty tests interfere, run `pytest -q -o addopts= tests`)

**Step 3: Commit**

Run:
```bash
git add README.md
git commit -m \"docs: document BigCodeBench runner\"
```

