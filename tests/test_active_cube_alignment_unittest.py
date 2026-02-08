import unittest


from memrl.service.memory_service import MemoryService


class _Dummy:
    """Minimal object to exercise MemoryService cube-binding sync logic."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestActiveCubeAlignment(unittest.TestCase):
    def test_sync_updates_q_updater_and_curator(self) -> None:
        q_updater = _Dummy(default_cube_id="cube_old")
        curator_q_updater = _Dummy(default_cube_id="cube_old2")
        curator = _Dummy(default_cube_id="cube_old", q_updater=curator_q_updater)

        svc = _Dummy(default_cube_id="cube_new", _q_updater=q_updater, _curator=curator)

        # Call the method on the dummy "service" instance.
        MemoryService._sync_cube_bound_components(  # type: ignore[arg-type]
            svc, old_cube_id="cube_old", reason="test"
        )

        self.assertEqual(q_updater.default_cube_id, "cube_new")
        self.assertEqual(curator.default_cube_id, "cube_new")
        self.assertEqual(curator_q_updater.default_cube_id, "cube_new")

    def test_sync_is_noop_when_components_missing(self) -> None:
        svc = _Dummy(default_cube_id="cube_new")

        # Should not raise even if value-driven components were never created.
        MemoryService._sync_cube_bound_components(  # type: ignore[arg-type]
            svc, old_cube_id="cube_old", reason="test-missing"
        )


if __name__ == "__main__":
    unittest.main()

