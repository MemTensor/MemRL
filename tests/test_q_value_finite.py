"""Exercise the real updater with an in-memory persistence boundary; no MemOS service."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def load_module():
    stub = ModuleType("memos.mem_os.main")
    stub.MOS = object
    spec = importlib.util.spec_from_file_location(
        "value_driven_under_test",
        Path(__file__).resolve().parents[1] / "memrl/service/value_driven.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"memos.mem_os.main": stub, spec.name: module}):
        spec.loader.exec_module(module)
    return module


vd = load_module()


class Store:
    def __init__(self, metadata):
        self.metadata = metadata.copy()
        self.writes = []

    def get(self, memory_id):
        return SimpleNamespace(memory="synthetic", metadata=self.metadata)

    def update(self, memory_id, value):
        self.writes.append(value)
        self.metadata = value["metadata"]


class FiniteUpdateTests(unittest.TestCase):
    def make_updater(self, metadata=None, **config):
        store = Store(metadata or {"q_value": 0.0})
        updater = vd.QValueUpdater(None, "test", vd.RLConfig(**config))
        updater._get_text_mem = lambda: store
        return updater, store

    def test_nonfinite_inputs_never_write(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for field in ("reward", "next_max_q", "alpha", "gamma", "q_floor", "q_value", "reward_ma"):
                with self.subTest(value=value, field=field):
                    metadata = {"q_value": 0.0}
                    config, args = {}, {"reward": 1.0}
                    if field in ("q_value", "reward_ma"):
                        metadata[field] = value
                    elif field in ("reward", "next_max_q"):
                        args[field] = value
                    else:
                        config[field] = value
                    updater, store = self.make_updater(metadata, **config)
                    with self.assertRaises(ValueError):
                        updater.update("m", **args)
                    self.assertEqual(store.writes, [])

    def test_finite_overflow_never_writes(self):
        cases = [
            ({"gamma": 2.0}, {"q_value": 0.0}, 1e308, 1e308),
            ({"alpha": 2.0}, {"q_value": 0.0}, 1e308, None),
            ({"alpha": 2.0}, {"q_value": 0.0, "reward_ma": -1e308}, 1e308, None),
        ]
        for config, metadata, reward, next_q in cases:
            with self.subTest(config=config, metadata=metadata):
                updater, store = self.make_updater(metadata, **config)
                with self.assertRaises(ValueError):
                    updater.update("m", reward, next_q)
                self.assertEqual(store.writes, [])

    def test_valid_repeated_updates_and_floor(self):
        updater, store = self.make_updater(alpha=0.5, gamma=0.5, q_floor=-1.0)
        self.assertEqual(updater.update("m", -2.0), -1.0)
        self.assertEqual(updater.update("m", 2.0, 2.0), 1.0)
        self.assertEqual(store.metadata["q_visits"], 2)
        self.assertEqual(store.metadata["reward_ma"], 0.5)
        self.assertEqual(store.metadata["last_reward"], 2.0)


if __name__ == "__main__":
    unittest.main()
