from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure local package imports work even when pytest import-mode/sys.path differs.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_config_rlconfig_has_sim_threshold_default() -> None:
    # Config-level RLConfig should expose sim_threshold so YAML/JSON config can set it.
    from memp.configs.config import RLConfig

    cfg = RLConfig()
    assert hasattr(cfg, "sim_threshold")
    assert cfg.sim_threshold == 0.5


def test_llb_runner_uses_sim_threshold_for_retrieve_query_threshold() -> None:
    # Avoid importing memp.service.* in unit tests (may require external memos deps).
    runner_path = _ROOT / "memp" / "run" / "llb_rl_runner.py"
    txt = runner_path.read_text(encoding="utf-8")

    # Ensure retrieve_query threshold is driven by sim_threshold (with safe fallback).
    # NOTE: keep the regex simple; we only care that the retrieve_query(...) call's
    # threshold expression references `sim_threshold` (it may also include a fallback).
    pattern = r"retrieve_query\([\s\S]*?threshold=[\s\S]*?sim_threshold"
    assert re.search(pattern, txt) is not None
