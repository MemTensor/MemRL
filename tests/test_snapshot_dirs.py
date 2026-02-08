import os
import tempfile
import unittest


from memrl.service.memory_service import _resolve_snapshot_dirs


class TestResolveSnapshotDirs(unittest.TestCase):
    def test_falls_back_when_qdrant_dir_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            meta = {"cube_dir": os.path.join(root, "cube_override"), "qdrant_dir": None, "checkpoint_id": 7}
            cube_dir, qdrant_dir, ckpt = _resolve_snapshot_dirs(root, meta)
            self.assertEqual(cube_dir, meta["cube_dir"])
            self.assertEqual(qdrant_dir, os.path.join(root, "qdrant"))
            self.assertEqual(ckpt, 7)

    def test_falls_back_when_qdrant_dir_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            meta = {"qdrant_dir": ""}
            cube_dir, qdrant_dir, ckpt = _resolve_snapshot_dirs(root, meta)
            self.assertEqual(cube_dir, os.path.join(root, "cube"))
            self.assertEqual(qdrant_dir, os.path.join(root, "qdrant"))
            self.assertEqual(ckpt, 0)


if __name__ == "__main__":
    unittest.main()

