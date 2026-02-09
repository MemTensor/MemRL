from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_min_runner(tmp_path: Path, monkeypatch, summary_writer):
    # Import inside helper so monkeypatching works reliably.
    import memrl.run.bcb_runner as bcb_runner

    monkeypatch.setattr(bcb_runner, "SummaryWriter", summary_writer, raising=False)

    sel = bcb_runner.BCBSelection(subset="hard", split="instruct", train_ratio=0.7, seed=42)

    class _LLM:
        pass

    class _Mem:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    bcb_repo = repo_root / "3rdparty" / "bigcodebench-main"

    return bcb_runner.BCBRunner(
        root=tmp_path,
        selection=sel,
        llm=_LLM(),
        memory_service=_Mem(),
        output_dir=str(tmp_path / "results" / "bcb" / "run1"),
        model_name="dummy",
        num_epochs=1,
        run_validation=False,
        retrieve_k=1,
        bcb_repo=str(bcb_repo),
    )


def test_bcb_tensorboard_noop_when_unavailable(tmp_path, monkeypatch, caplog):
    runner = _make_min_runner(tmp_path, monkeypatch, summary_writer=None)
    # No-op writer should exist and be safe to call.
    runner.writer.add_scalar("bcb/train/processed", 1, global_step=1)
    runner.writer.close()
    assert any("TensorBoard is not available" in r.message for r in caplog.records)


def test_bcb_tensorboard_logdir_convention(tmp_path, monkeypatch):
    created = {}

    class _StubWriter:
        def __init__(self, *, log_dir: str):
            created["log_dir"] = log_dir

        def add_scalar(self, *args, **kwargs):
            return

        def close(self):
            return

    _make_min_runner(tmp_path, monkeypatch, summary_writer=_StubWriter)

    log_dir = Path(created["log_dir"])
    # Convention: <root>/logs/tensorboard/...
    assert log_dir.parts[-3:-1] == ("logs", "tensorboard")
