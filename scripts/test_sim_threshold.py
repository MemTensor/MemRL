"""Regression checks for benchmark retrieval-similarity thresholds."""

import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memrl.configs.config import RLConfig  # noqa: E402


def _load_rl_config(config_name: str) -> RLConfig:
    config_path = PROJECT_ROOT / "configs" / config_name
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return RLConfig(**config["rl_config"])


class SimThresholdTests(unittest.TestCase):
    def test_alfworld_retrieval_threshold_matches_paper_value(self) -> None:
        config = _load_rl_config("rl_alf_config.yaml")

        self.assertAlmostEqual(config.tau, 0.62)
        self.assertAlmostEqual(config.sim_threshold, 0.62)
        self.assertAlmostEqual(config.resolve_sim_threshold(), 0.62)

    def test_bigcodebench_retrieval_threshold_matches_paper_value(self) -> None:
        config = _load_rl_config("rl_bcb_config.yaml")

        self.assertAlmostEqual(config.sim_threshold, 0.38)
        self.assertAlmostEqual(config.resolve_sim_threshold(), 0.38)

    def test_hle_retrieval_threshold_matches_paper_value(self) -> None:
        config = _load_rl_config("rl_hle_config.yaml")

        self.assertAlmostEqual(config.tau, 0.25)
        self.assertAlmostEqual(config.sim_threshold, 0.25)
        self.assertAlmostEqual(config.resolve_sim_threshold(), 0.25)

    def test_llb_resolves_task_specific_thresholds(self) -> None:
        config = _load_rl_config("rl_llb_config.yaml")
        cases = {
            "os": 0.50,
            "os_interaction": 0.50,
            "OS_INTERACTION": 0.50,
            "db": 0.37,
            "db_bench": 0.37,
            " DB_BENCH ": 0.37,
        }

        for task, expected in cases.items():
            with self.subTest(task=task):
                self.assertAlmostEqual(
                    config.resolve_sim_threshold(task), expected
                )

    def test_unknown_llb_task_uses_global_fallback(self) -> None:
        config = _load_rl_config("rl_llb_config.yaml")

        self.assertAlmostEqual(config.resolve_sim_threshold("kg"), 0.50)
        self.assertAlmostEqual(config.resolve_sim_threshold(None), 0.50)


if __name__ == "__main__":
    unittest.main()
