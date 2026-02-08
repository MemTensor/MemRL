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
    from memrl.configs.config import RLConfig

    cfg = RLConfig()
    assert hasattr(cfg, "sim_threshold")
    assert cfg.sim_threshold == 0.5


def test_llb_runner_uses_sim_threshold_for_retrieve_query_threshold() -> None:
    # Avoid importing memrl.service.* in unit tests (may require external memos deps).
    runner_path = _ROOT / "memrl" / "run" / "llb_rl_runner.py"
    txt = runner_path.read_text(encoding="utf-8")

    # Ensure retrieve_query threshold is driven by sim_threshold (with safe fallback).
    #
    # The implementation may either pass the expression inline, or compute a local
    # variable (e.g., `thr = ...`) and pass `threshold=thr`. We accept both forms.
    has_sim_threshold_source = (
        re.search(r"retrieve_query\([\s\S]*?threshold\s*=\s*[\s\S]*?sim_threshold", txt)
        is not None
    ) or (re.search(r"\bthr\s*=\s*[\s\S]*?sim_threshold", txt) is not None)
    assert has_sim_threshold_source

    # Sanity: retrieve_query must be called with a threshold argument.
    assert re.search(r"retrieve_query\([\s\S]*?threshold\s*=", txt) is not None
