# scripts/run_agent_experiment.py
import sys
import os
from pathlib import Path
import logging
import tempfile
import shutil
import json
import argparse

# --- Setup Project Path ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# --- Setup LLB Path ---
# 确保 LLB 在 sys.path 中，以便导入其组件
LLB_ROOT = project_root / "3rdparty" / "LifelongAgentBench"
if not LLB_ROOT.exists():
    raise RuntimeError(f"LLB directory not found: {LLB_ROOT}")

# Python 3.10 兼容：为 enum.StrEnum 提供兜底实现
try:
    import enum as _enum

    if not hasattr(_enum, "StrEnum"):

        class _StrEnum(str, _enum.Enum):
            pass

        _enum.StrEnum = _StrEnum  # type: ignore[attr-defined]
    import typing as _typing

    if not hasattr(_typing, "reveal_type"):

        def _noop_reveal_type(x):
            return x

        _typing.reveal_type = _noop_reveal_type  # type: ignore[attr-defined]
    if not hasattr(_typing, "Self"):
        _typing.Self = object  # type: ignore[attr-defined]
except Exception:
    pass

if str(LLB_ROOT) not in sys.path:
    sys.path.insert(0, str(LLB_ROOT))

# --- Import all our components ---
from memp.configs.config import MempConfig
from memp.service.memory_service import MemoryService
from memp.service.strategies import (
    BuildStrategy,
    RetrieveStrategy,
    UpdateStrategy,
    StrategyConfiguration,
)
from memp.providers.llm import OpenAILLM
from memp.providers.embedding import OpenAIEmbedder
from memp.run.llb_rl_runner import LLBRunner


# (The setup_logging function remains the same)
def setup_logging(project_root: Path, name: str):
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    import time

    log_filename = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Logging configured. Log file: {log_filepath}")


logger = logging.getLogger(__name__)


def main():
    """
    Main function to initialize all components and start the runner,
    using the correct MemoryService initialization method.
    """
    # --- Experiment Configuration ---

    try:
        # --- 1. INITIALIZE ALL COMPONENTS ---
        logger.info("Initializing all components...")

        # Load Config and Providers
        config_path = project_root / "configs" / "rl_llb_config.yaml"
        config = MempConfig.from_yaml(str(config_path))
        setup_logging(project_root, config.experiment.experiment_name)

        # Use a temporary directory for all runtime artifacts (like mos_config.json and the DB)
        temp_dir = tempfile.mkdtemp(prefix="memp_live_agent_run_")
        logger.info(f"Using temporary directory for runtime artifacts: {temp_dir}")

        llm_provider = OpenAILLM(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            default_temperature=config.llm.temperature,
            default_max_tokens=config.llm.max_tokens,
            provider=config.llm.provider,
            api_version=config.llm.api_version,
        )
        embedding_provider = OpenAIEmbedder(
            api_key=config.embedding.api_key,
            base_url=config.embedding.base_url,
            model=config.embedding.model,
            provider=config.embedding.provider,
            api_version=config.embedding.api_version,
        )

        # --- Use the detailed MemoryService setup from your demo script ---
        logger.info("Configuring and initializing MemoryService...")
        user_id = f"live_agent_exp_{os.getpid()}"
        build_strategy = BuildStrategy(config.memory.build_strategy)
        retrieve_strategy = RetrieveStrategy(config.memory.retrieve_strategy)
        update_strategy = UpdateStrategy(config.memory.update_strategy)

        # 1. Create the mos_config dictionary
        # Note: MemOS config validation does not allow 'api_version', so we rely on env vars.
        if config.llm.provider == "azure":
            os.environ["OPENAI_API_TYPE"] = "azure"
        if config.llm.api_version:
            os.environ["OPENAI_API_VERSION"] = config.llm.api_version

        mos_config = {
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": config.llm.model,
                    "api_key": config.llm.api_key,
                    "api_base": config.llm.base_url,
                },
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": config.llm.model,
                            "api_key": config.llm.api_key,
                            "api_base": config.llm.base_url,
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": config.embedding.provider,
                            "model_name_or_path": config.embedding.model,
                            "api_key": config.embedding.api_key,
                            "base_url": config.embedding.base_url,
                        },
                    },
                    "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
                },
            },
            "user_manager": {
                "backend": "sqlite",
                "config": {"db_path": os.path.join(temp_dir, "users.db")},
            },
            "top_k": 5,
        }

        # 2. Write the config to a temporary JSON file
        mos_config_path = os.path.join(temp_dir, "mos_config.json")
        with open(mos_config_path, "w") as f:
            json.dump(mos_config, f)

        # 3. rl_config:

        enable_value_driven = config.experiment.enable_value_driven
        rl_config = config.rl_config

        logger.info("Config:\n%s", config.model_dump_json(indent=2))

        # 4. Initialize MemoryService with the config path and providers
        memory_service = MemoryService(
            mos_config_path=mos_config_path,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            strategy_config=StrategyConfiguration(
                build_strategy, retrieve_strategy, update_strategy
            ),
            user_id=user_id,
            num_workers=config.experiment.batch_size,
            max_keywords=config.memory.max_keywords,
            add_similarity_threshold=config.memory.add_similarity_threshold,
            enable_value_driven=enable_value_driven,
            rl_config=rl_config,
        )

        # Load from checkpoint if configured
        resumed_section = 0  # Default: start from beginning
        if config.memory.load_from_checkpoint and config.memory.checkpoint_path:
            checkpoint_path = Path(config.memory.checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = project_root / checkpoint_path

            if checkpoint_path.exists():
                logger.info(f"Loading memory from checkpoint: {checkpoint_path}")
                try:
                    resumed_section = memory_service.load_checkpoint_snapshot(
                        str(checkpoint_path)
                    )
                    logger.info(
                        f"✓ Checkpoint loaded successfully from section {resumed_section}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
                    raise
            else:
                logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
                logger.warning("Starting with fresh memory service")

        logger.info("All components initialized successfully.")

        # Initialize the Runner with the fully constructed components
        # Note: LLBRunner will create LanguageModelAgent instances internally using the adapter
        runner = LLBRunner(
            root=project_root,
            memory_service=memory_service,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            exp_name=config.experiment.experiment_name,
            random_seed=config.experiment.random_seed,
            num_section=config.experiment.num_sections,
            batch_size=config.experiment.batch_size,
            max_steps=config.experiment.max_steps,
            rl_config=rl_config,
            bon=config.experiment.bon,
            retrieve_k=config.memory.k_retrieve,
            mode=config.experiment.mode,
            # enable_value_driven=enable_value_driven,
            task=config.experiment.task,
            split_file=config.experiment.split_file,
            valid_interval=config.experiment.valid_interval,
            test_interval=config.experiment.test_interval,
            train_set_ratio=config.experiment.train_set_ratio,
            start_section=resumed_section,  # Resume from the checkpoint section
            algorithm=config.experiment.algorithm,
            val_before_train=config.experiment.val_before_train,
            valid_file=config.experiment.valid_file,  # Get from config
        )
        # --- RUN THE EXPERIMENT ---
        runner.run()

    except Exception as e:
        logger.error(
            f"An unhandled error occurred during the experiment: {e}", exc_info=True
        )
    finally:
        # --- 3. CLEANUP ---
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
