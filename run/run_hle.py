import sys
import os
from pathlib import Path
import logging
import argparse
import json as _json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memp.configs.config import MempConfig
from memp.providers.llm import OpenAILLM
from memp.providers.embedding import OpenAIEmbedder
from memp.service.memory_service import MemoryService
from memp.service.strategies import BuildStrategy, RetrieveStrategy, UpdateStrategy, StrategyConfiguration
from memp.run.hle_runner import HLERunner, HLESelection


def setup_logging(project_root: Path, name: str):
    log_dir = project_root / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    import time
    log_filename = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Logging configured. Log file: {log_filepath}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HLE benchmark with memory-agent")
    p.add_argument("--config", type=str, default=str(project_root / "configs" / "rl_hle_config.yaml"))
    default_parquet = str(project_root.parent / "hle" / "test-00000-of-00001-filtered.parquet")
    p.add_argument("--valid", type=str, default=default_parquet)
    # Use the same parquet as training set by default
    p.add_argument("--train", type=str, default=default_parquet)
    p.add_argument("--num_valid", type=int, default=0)
    p.add_argument("--num_train", type=int, default=0)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--judge_model", type=str, default='gpt-4o-2024-08-06')
    p.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=['Computer Science/AI', 'Math', 'Biology/Medicine', 'Physics', 'Chemistry', 'Engineering', 'Humanities/Social Science', 'Other'],
        help="Filter HLE rows to these categories (space-separated list).",
    )
    p.add_argument(
        "--category_ratio",
        type=float,
        default=1.0,
        help="Per-category sampling ratio (0-1) after filtering categories.",
    )
    return p.parse_args()


def main():
    logger = logging.getLogger(__name__)
    args = parse_args()
    try:
        cfg = MempConfig.from_yaml(args.config)
        setup_logging(project_root, cfg.experiment.experiment_name)

        out_dir = Path(cfg.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime('%Y%m%d-%H%M%S')
        log_dir = out_dir / "hle" / f"exp_{cfg.experiment.experiment_name}_{run_id}" / "local_cache"
        log_dir.mkdir(parents=True, exist_ok=True)

        llm = OpenAILLM(
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            default_temperature=cfg.llm.temperature,
            default_max_tokens=cfg.llm.max_tokens,
            token_log_dir=str(log_dir),
        )
        embedder = OpenAIEmbedder(
            api_key=cfg.embedding.api_key,
            base_url=cfg.embedding.base_url,
            model=cfg.embedding.model,
            max_text_len=getattr(cfg.embedding, "max_text_len", 4096),
            token_log_dir=str(log_dir),
        )
        # Optional separate judge LLM
        llm_judge = None
        if args.judge_model:
            llm_judge = OpenAILLM(
                api_key=cfg.llm.api_key,
                base_url=cfg.llm.base_url,
                model=args.judge_model,
                default_temperature=0.0,
                default_max_tokens=4096,
                token_log_dir=str(log_dir),
            )

        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="memp_hle_run_")
        user_id = f"hle_{os.getpid()}"
        mos_config = {
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": cfg.llm.model,
                    "api_key": cfg.llm.api_key,
                    "api_base": cfg.llm.base_url
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
                            "provider": "openai",
                            "model_name_or_path": cfg.embedding.model,
                            "api_key": cfg.embedding.api_key,
                            "base_url": cfg.embedding.base_url,
                        },
                    },
                    "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
                },
            },
            "user_manager": {"backend": "sqlite", "config": {"db_path": os.path.join(temp_dir, "users.db")}},
            "top_k": 5,
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
            add_similarity_threshold=getattr(cfg.memory, 'add_similarity_threshold', 0.9),
            enable_value_driven=cfg.experiment.enable_value_driven,
            rl_config=cfg.rl_config,
            db_max_concurrency=4,
            sim_norm_mean=getattr(cfg.memory, 'sim_norm_mean', 0.1856827586889267),
            sim_norm_std=getattr(cfg.memory, 'sim_norm_std', 0.09407906234264374),
        )

        sel = HLESelection(
            valid_path=args.valid,
            train_path=args.train,
            num_valid=(args.num_valid if args.num_valid and args.num_valid > 0 else None),
            num_train=(args.num_train if args.num_train and args.num_train > 0 else None),
            categories=args.categories or getattr(cfg.experiment, "hle_categories", None),
            category_ratio=args.category_ratio if args.category_ratio is not None else getattr(cfg.experiment, "hle_category_ratio", None),
        )

        runner = HLERunner(
            name=cfg.experiment.experiment_name,
            llm=llm,
            llm_judge=llm_judge,
            selection=sel,
            output_dir=out_dir,
            memory_service=memsvc,
            run_id=run_id,
            temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
            max_tokens=(args.max_tokens if args.max_tokens is not None else (cfg.llm.max_tokens or 4096)),
            retrieve_k=cfg.memory.k_retrieve,
            num_sections=cfg.experiment.num_sections,
            batch_size=cfg.experiment.batch_size,
            dataset_ratio=getattr(cfg.experiment, "dataset_ratio", 1.0),
            random_seed=getattr(cfg.experiment, "random_seed", 42) or 42,
            train_valid_split=getattr(cfg.experiment, "train_valid_split", 0.8),
            ckpt_eval_enabled=getattr(cfg.experiment, "ckpt_eval_enabled", False),
            ckpt_eval_path=getattr(cfg.experiment, "ckpt_eval_path", None),
            ckpt_resume_enabled=getattr(cfg.experiment, "ckpt_resume_enabled", False),
            ckpt_resume_path=getattr(cfg.experiment, "ckpt_resume_path", None),
            ckpt_resume_epoch=getattr(cfg.experiment, "ckpt_resume_epoch", None),
        )
        runner.run()
    except Exception as e:
        logger.error(f"HLE run failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
