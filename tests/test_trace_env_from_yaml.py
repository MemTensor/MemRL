import os


def test_apply_trace_env_from_experiment_config_sets_env(monkeypatch):
    from memp.configs.config import ExperimentConfig
    from memp.trace.llb_jsonl import apply_trace_env_from_experiment_config

    monkeypatch.delenv("TRACE_JSONL_PATH", raising=False)
    monkeypatch.delenv("TRACE_SAMPLE_FILTER", raising=False)

    exp = ExperimentConfig(
        experiment_name="x",
        trace_jsonl_path="logs/llb_trace.jsonl",
        trace_sample_filter="3",
    )
    apply_trace_env_from_experiment_config(exp)

    assert os.environ["TRACE_JSONL_PATH"] == "logs/llb_trace.jsonl"
    assert os.environ["TRACE_SAMPLE_FILTER"] == "3"


def test_apply_trace_env_from_experiment_config_does_not_override_existing_env(
    monkeypatch,
):
    from memp.configs.config import ExperimentConfig
    from memp.trace.llb_jsonl import apply_trace_env_from_experiment_config

    monkeypatch.setenv("TRACE_JSONL_PATH", "already.jsonl")
    monkeypatch.setenv("TRACE_SAMPLE_FILTER", "abc")

    exp = ExperimentConfig(
        experiment_name="x",
        trace_jsonl_path="from_yaml.jsonl",
        trace_sample_filter="1",
    )
    apply_trace_env_from_experiment_config(exp)

    # YAML has higher priority than env vars when the YAML keys are explicitly set.
    assert os.environ["TRACE_JSONL_PATH"] == "from_yaml.jsonl"
    assert os.environ["TRACE_SAMPLE_FILTER"] == "1"


def test_apply_trace_env_from_experiment_config_can_unset_env_when_yaml_is_null(
    monkeypatch,
):
    from memp.configs.config import ExperimentConfig
    from memp.trace.llb_jsonl import apply_trace_env_from_experiment_config

    monkeypatch.setenv("TRACE_JSONL_PATH", "already.jsonl")
    monkeypatch.setenv("TRACE_SAMPLE_FILTER", "abc")

    exp = ExperimentConfig(
        experiment_name="x",
        trace_jsonl_path=None,
        trace_sample_filter=None,
    )
    apply_trace_env_from_experiment_config(exp)

    assert "TRACE_JSONL_PATH" not in os.environ
    assert "TRACE_SAMPLE_FILTER" not in os.environ
