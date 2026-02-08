def test_llmconfig_has_api_version_field():
    from memrl.configs.config import LLMConfig

    cfg = LLMConfig()
    # run/run_llb.py expects this attribute to exist (may be None).
    assert hasattr(cfg, "api_version")


def test_embeddingconfig_has_api_version_field():
    from memrl.configs.config import EmbeddingConfig

    cfg = EmbeddingConfig()
    # run/run_llb.py expects this attribute to exist (may be None).
    assert hasattr(cfg, "api_version")


def test_memoryconfig_has_checkpoint_fields():
    from memrl.configs.config import MemoryConfig

    cfg = MemoryConfig()
    # run/run_llb.py expects these attributes to exist (may be unset).
    assert hasattr(cfg, "load_from_checkpoint")
    assert hasattr(cfg, "checkpoint_path")
