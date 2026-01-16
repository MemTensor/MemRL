import argparse
import json as _json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memp.configs.config import MempConfig
from memp.providers.llm import OpenAILLM
from memp.providers.embedding import OpenAIEmbedder
from memp.service.memory_service import MemoryService
from memp.service.strategies import (
    BuildStrategy,
    RetrieveStrategy,
    UpdateStrategy,
    StrategyConfiguration,
)
from memp.run.bcb_runner import BCBRunner, BCBSelection

DEFAULT_SPLIT_FILES = {
    "hard": project_root / "configs" / "bigcodebench" / "splits" / "hard_seed42.json",
    "full": project_root / "configs" / "bigcodebench" / "splits" / "full_seed123.json",
}


def setup_logging(project_root: Path, name: str) -> None:
    log_dir = project_root / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
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

    logging.info("Logging configured. Log file: %s", log_filepath)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run BigCodeBench (BCB) multi-epoch memory benchmark"
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "rl_bcb_config.yaml"),
    )
    p.add_argument("--subset", type=str, default="hard", choices=["hard", "full"])
    p.add_argument(
        "--split", type=str, default="instruct", choices=["instruct", "complete"]
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split_file",
        type=str,
        default=None,
        help=(
            "Path to a JSON split file containing train_ids/val_ids. "
            "If omitted, uses legacy split files under configs/bigcodebench/splits/."
        ),
    )
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument(
        "--bcb_repo",
        type=str,
        default=str(project_root / "3rdparty" / "bigcodebench-main"),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override cfg.experiment.output_dir",
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--retrieve_k", type=int, default=None)
    p.add_argument("--retrieve_threshold", type=float, default=None)
    p.add_argument("--eval_timeout", type=float, default=60.0)
    p.add_argument("--untrusted_hard_timeout", type=float, default=120.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(project_root, "bcb")
    logger = logging.getLogger(__name__)

    cfg = MempConfig.from_yaml(args.config)

    if args.split_file is None:
        default_split = DEFAULT_SPLIT_FILES.get(args.subset)
        if default_split is not None and default_split.exists():
            args.split_file = str(default_split)

    out_root = Path(args.output_dir or cfg.experiment.output_dir or "./results").resolve()
    out_dir = out_root / "bigcodebench_eval" / f"{args.split}_{args.subset}" / "memory"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{time.strftime('%Y%m%d_%H%M%S')}_{cfg.llm.model.replace('/', '_')}"
        f"_rl-{'on' if cfg.experiment.enable_value_driven else 'off'}"
    )
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Providers
    token_log_dir = str((project_root / "logs" / "bcb").resolve())
    llm = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_temperature=(
            args.temperature if args.temperature is not None else cfg.llm.temperature
        ),
        default_max_tokens=(args.max_tokens if args.max_tokens is not None else cfg.llm.max_tokens),
        token_log_dir=token_log_dir,
    )
    embedder = OpenAIEmbedder(
        api_key=cfg.embedding.api_key,
        base_url=cfg.embedding.base_url,
        model=cfg.embedding.model,
        max_text_len=getattr(cfg.embedding, "max_text_len", 8196),
    )

    # MemOS config JSON for MemoryService (consistent with other runners).
    temp_dir = tempfile.mkdtemp(prefix="memp_bcb_run_")
    user_id = f"bcb_{os.getpid()}"
    mos_config = {
        "chat_model": {
            "backend": "openai",
            "config": {
                "model_name_or_path": cfg.llm.model,
                "api_key": cfg.llm.api_key,
                "api_base": cfg.llm.base_url,
            },
        },
        "mem_reader": {
            "backend": "simple_struct",
            "config": {
                "llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": cfg.llm.model,
                        "api_key": cfg.llm.api_key,
                        "api_base": cfg.llm.base_url,
                    },
                },
                "embedder": {
                    "backend": "universal_api",
                    "config": {
                        "provider": cfg.embedding.provider,
                        "model_name_or_path": cfg.embedding.model,
                        "api_key": cfg.embedding.api_key,
                        "base_url": cfg.embedding.base_url,
                    },
                },
                "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
            },
        },
        "user_manager": {
            "backend": "sqlite",
            "config": {"db_path": os.path.join(temp_dir, "users.db")},
        },
        "top_k": int(
            args.retrieve_k if args.retrieve_k is not None else cfg.memory.k_retrieve
        ),
    }

    mos_config_path = os.path.join(temp_dir, "mos_config.json")
    with open(mos_config_path, "w", encoding="utf-8") as f:
        _json.dump(mos_config, f)

    memsvc = MemoryService(
        mos_config_path=mos_config_path,
        llm_provider=llm,
        embedding_provider=embedder,
        strategy_config=StrategyConfiguration(
            BuildStrategy(cfg.memory.build_strategy),
            RetrieveStrategy(cfg.memory.retrieve_strategy),
            UpdateStrategy(cfg.memory.update_strategy),
        ),
        user_id=user_id,
        num_workers=cfg.experiment.batch_size,
        max_keywords=cfg.memory.max_keywords,
        add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
        enable_value_driven=cfg.experiment.enable_value_driven,
        rl_config=cfg.rl_config,
        db_max_concurrency=4,
        sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", 0.1856827586889267),
        sim_norm_std=getattr(cfg.memory, "sim_norm_std", 0.09407906234264374),
    )

    sel = BCBSelection(
        subset=args.subset,
        split=args.split,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        split_file=args.split_file,
        data_path=args.data_path,
    )

    runner = BCBRunner(
        root=project_root,
        selection=sel,
        llm=llm,
        memory_service=memsvc,
        output_dir=str(run_dir),
        model_name=cfg.llm.model,
        num_epochs=int(args.epochs),
        temperature=(
            args.temperature if args.temperature is not None else cfg.llm.temperature
        ),
        max_tokens=(
            args.max_tokens if args.max_tokens is not None else (cfg.llm.max_tokens or 1280)
        ),
        retrieve_k=(args.retrieve_k if args.retrieve_k is not None else cfg.memory.k_retrieve),
        retrieve_threshold=(
            args.retrieve_threshold if args.retrieve_threshold is not None else 0.2
        ),
        rl_enabled=cfg.experiment.enable_value_driven,
        bcb_repo=args.bcb_repo,
        untrusted_hard_timeout_s=float(args.untrusted_hard_timeout),
        eval_timeout_s=float(args.eval_timeout),
    )

    logger.info("BCB run_dir: %s", run_dir)
    runner.run()


if __name__ == "__main__":
    main()
