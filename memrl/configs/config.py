"""
Configuration management for the Memp system.

This module defines Pydantic models for configuration management,
supporting both YAML and JSON configuration files.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
import yaml
import json


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    
    provider: str = Field(default="openai", description="LLM provider name")
    api_key: str = Field(default="sk-8", 
                        description="API key for authentication")
    base_url: Optional[str] = Field(default="https://api.openai.com/v1", description="Base URL for API")
    # Optional Azure/OpenAI-style API versioning. Some runners pass this through.
    api_version: Optional[str] = Field(default=None, description="Optional API version for some providers (e.g. Azure OpenAI)")
    model: str = Field(default="gpt-4.1-mini", description="Model name")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Generation temperature")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum tokens")
    
    @field_validator('api_key')
    @classmethod
    def api_key_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('API key cannot be empty')
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding provider."""
    
    provider: str = Field(default="openai", description="Embedding provider name")
    api_key: str = Field(default="sk-8",
                        description="API key for authentication")
    base_url: Optional[str] = Field(default="https://api.openai.com/v1", description="Base URL for API")
    api_version: Optional[str] = Field(default=None, description="Optional API version for some providers (e.g. Azure OpenAI)")
    model: str = Field(default="text-embedding-3-large", description="Embedding model name")
    max_text_len: int = Field(
        default=8196,
        ge=0,
        description="Maximum characters per query before chunked embedding (0 disables chunking)",
    )
    
    @field_validator('api_key')
    @classmethod
    def api_key_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('API key cannot be empty')
        return v


class MemoryConfig(BaseModel):
    """Configuration for memory system."""
    
    # Strategy configuration
    build_strategy: str = Field(default="proceduralization", 
                               description="Build strategy: trajectory, script, proceduralization")
    retrieve_strategy: str = Field(default="query",
                                  description="Retrieve strategy: random, query, avefact") 
    update_strategy: str = Field(default="adjustment",
                                description="Update strategy: vanilla, validation, adjustment")
    
    # Memory parameters
    k_retrieve: int = Field(default=1, gt=-1, description="Number of memories to retrieve")
    max_keywords: int = Field(default=8, gt=0, description="Maximum keywords for AveFact")
    confidence_threshold: float = Field(default=0.0, ge=0, le=1, 
                                       description="Minimum similarity threshold")
    memory_confidence: float = Field(default=100.0, ge=0, le=100,
                                    description="Confidence score for new memories")
    add_similarity_threshold: float = Field(default=0.9,
                                    description="similarity_threshold for add memory")     
    # MemOS configuration
    mos_config_path: str = Field(default="configs/mos_config.json",
                                description="Path to MemOS configuration file")
    user_id: str = Field(default="memp_user", description="User ID for memory management")
    sim_norm_mean: float = Field(default=0, description="Mean for similarity normalization")
    sim_norm_std: float = Field(default=0, description="Standard deviation for similarity normalization")

    # Optional checkpoint loading for runners that support resuming memory state.
    load_from_checkpoint: bool = Field(default=False, description="Whether to load memory state from a checkpoint snapshot")
    checkpoint_path: Optional[str] = Field(default=None, description="Path to a memory checkpoint snapshot to load (if enabled)")


class EnvironmentConfig(BaseModel):
    """Configuration for environment-specific settings."""
    
    # ALFWorld settings  
    alfworld_config_path: str = Field(default="configs/base_config.yaml",
                                     description="Path to ALFWorld configuration")
    alfworld_env_type: str = Field(default="AlfredTWEnv",
                                  description="ALFWorld environment type")


class ExperimentConfig(BaseModel):
    """Configuration for experimental settings."""

    # Experiment parameters
    experiment_name: str = Field(..., description="Name for the experiment trail")
    algorithm: str = Field(
        default="rl", description="Algorithm to run: memp, rl, mdp, rlmdp, slow_rl"
    )
    val_before_train: bool = Field(
        default=True, description="Whether to run validation before training starts"
    )
    enable_value_driven: bool = Field(default=False, description="Whether to use rl")
    random_seed: Optional[int] = Field(
        default=42, description="Random seed for reproducibility"
    )
    mode: str = Field(
        default="train", description="Control whether to only train or test"
    )

    # Optional tracing (LLB JSONL) controlled by YAML (YAML overrides env vars when set).
    trace_jsonl_path: Optional[str] = Field(
        default=None,
        description="If set, enable LLB per-task JSONL tracing and write to this path.",
    )
    trace_sample_filter: Optional[str] = Field(
        default=None,
        description="Optional task filter for tracing: digits=first N tasks; or comma-separated sample_index list.",
    )

    # LLB retrieval behavior toggles (used only by LLB runner entrypoints).
    llb_use_z_score_normalization: bool = Field(
        default=True,
        description=(
            "LLB only: whether to z-score normalize similarity/q before hybrid scoring. "
            "Set false to rank by raw similarity + raw q (closer to legacy behavior)."
        ),
    )
    llb_q_floor: Optional[float] = Field(
        default=None,
        description=(
            "LLB only: if set, overrides rl_config.q_floor for LLB runs. "
            "Mimics memory_rl's q_floor to prevent Q from dropping below this value "
            "(applied during both Q update and retrieval scoring)."
        ),
    )
    llb_dedup_by_task_id: bool = Field(
        default=False,
        description=(
            "LLB only: when selecting the final top-k retrieved memories, deduplicate by task_id "
            "(fallback to sample_index/id; if still missing, treated as unique). "
            "This mimics memory_rl's task-level Phase-B behavior and reduces same-task repeats."
        ),
    )

    # LLB-specific parameters
    task: str = Field(default="db", description="Task type for LLB: db, os, kg")
    split_file: str = Field(default="", description="Path to LLB dataset split file")
    valid_file: Optional[str] = Field(
        default=None, description="Path to LLB validation dataset file"
    )

    num_sections: int = Field(default=5, description="Number of sections to split the training data into")
    batch_size: int = Field(default=32, description="Number of parallel environments for sampling")
    max_steps: int = Field(default=30, description="Max steps per episode during training and evaluation")
    valid_interval: int = Field(default=1, description="Run evaluation on the validation set every N sections. Set to 0 to disable.")
    test_interval: int = Field(default=1, description="Run evaluation on the test set every N sections. Set to 0 to disable.")
    dataset_ratio: float = Field(default=0.7, description="Proportion of files randomly selected for training (rest used for validation)")
    few_shot_path: str = Field(default='data/alfworld/alfworld_examples.json', description="Path for alfworld examples")
    bon: int = Field(default=0, description="Run BoN-evaluation on the val/test for N trails")
    hle_categories: Optional[List[str]] = Field(default=None, description="Subset of HLE categories to keep")
    hle_category_ratio: Optional[float] = Field(default=None, description="Per-category sampling ratio (0,1]")
    train_valid_split: float = Field(default=0.8, description="Ratio to split training and validation sets")
    ckpt_eval_enabled: bool = Field(default=False, description="Whether to evaluate by loading historical checkpoints")
    ckpt_eval_path: Optional[str] = Field(default=None, description="Path to experiment or snapshot directory for ckpt eval")
    ckpt_resume_enabled: bool = Field(default=False, description="Whether to resume training from a checkpoint snapshot")
    ckpt_resume_path: Optional[str] = Field(default=None, description="Path to experiment or snapshot directory for ckpt resume")
    ckpt_resume_epoch: Optional[int] = Field(default=None, description="Epoch index (1-based) to resume from")
    baseline_mode: Optional[str] = Field(default=None, description="Baseline mode: passk or reflection")
    baseline_k: int = Field(default=10, description="Baseline rounds (k) for pass@k/reflection")
    # Output settings
    output_dir: str = Field(default="./results", description="Directory for experiment outputs")
    save_trajectories: bool = Field(default=True, description="Save detailed trajectories")
    save_memories: bool = Field(default=True, description="Save memory snapshots")

    # Logging settings
    enable_logging: bool = Field(default=True, description="Enable detailed logging")
    log_level: str = Field(default="INFO", description="Logging level")

class RLConfig(BaseModel):
    """Configuration for reinforcement learning parameters."""

    epsilon: float = Field(default=0.1, description="ε-greedy exploration probability")
    tau: float = Field(default=0.35, description="Unknown detection threshold on similarity")
    alpha: float = Field(default=0.3, description="Q-learning step size (learning rate)")
    gamma: float = Field(default=0.0, description="Discount factor (default 0 for single-step)")
    q_init_pos: float = Field(default=0.0, description="Optimistic initialization for positive Q-values")
    q_init_neg: float = Field(default=0.0, description="Initialization for negative Q-values")
    q_floor: Optional[float] = Field(
        default=None,
        description=(
            "Minimum allowed Q value (optional). "
            "Only applied by runners that explicitly enable it (e.g., LLB via experiment.llb_q_floor)."
        ),
    )
    success_reward: float = Field(default=1.0, description="Reward for successful outcome")
    failure_reward: float = Field(default=-1.0, description="Reward for failure outcome")
    # Retrieval filtering threshold used by runners when calling MemoryService.retrieve_query(...).
    # (Kept separate from `tau` to avoid conflating unknown-detection vs retrieval filtering.)
    sim_threshold: float = Field(default=0.5, description="Similarity threshold for retrieval filtering")
    topk: int = Field(default=5, description="Candidate set size for value-aware selection")
    novelty_threshold: float = Field(default=0.85, description="Similarity threshold to treat as non-novel (merge)")
    recency_boost: float = Field(default=0.0, description="Optional recency weight for prioritization")
    reward_merge_gain: float = Field(default=0.1, description="Gain for attributing success to close memories")
    q_min_threshold: float = Field(default=-0.8, description="Threshold for q min")
    weight_sim: float = Field(default=0.5, description="Weight for similarity in combined score")
    weight_q: float = Field(default=0.5, description="Weight for Q-value in combined score")

class MempConfig(BaseModel):
    """Main configuration class for the Memp system."""
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    rl_config: RLConfig = Field(default_factory=RLConfig)
    # Global settings
    project_name: str = Field(default="memp", description="Project name")
    version: str = Field(default="0.1.0", description="Project version")
    
    model_config = ConfigDict(extra="forbid")  # Don't allow extra fields
        
    @classmethod
    def from_yaml(cls, config_path: str) -> "MempConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            MempConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                config_data = {}
                
            return cls(**config_data)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")
    
    @classmethod
    def from_json(cls, config_path: str) -> "MempConfig":
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            MempConfig instance
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            return cls(**config_data)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")
    
    def to_yaml(self, output_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML configuration
        """
        config_dict = self.model_dump()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_json(self, output_path: str, indent: int = 2) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            output_path: Path to save JSON configuration
            indent: JSON indentation level
        """
        config_dict = self.model_dump()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=indent)
    
    def get_strategy_config(self):
        """Get strategy configuration for MemoryService."""
        from ..service.strategies import StrategyConfiguration
        
        return StrategyConfiguration.from_strings(
            build=self.memory.build_strategy,
            retrieve=self.memory.retrieve_strategy,
            update=self.memory.update_strategy
        )
    
    def validate_paths(self) -> None:
        """
        Validate that all specified paths exist.
        
        Raises:
            FileNotFoundError: If required paths don't exist
        """
        paths_to_check = [
            (self.memory.mos_config_path, "MemOS config file"),
            (self.environment.alfworld_config_path, "ALFWorld config file"),
        ]
        
        # Only check TravelPlanner data dir if it's not the default relative path
        if not self.environment.travelplanner_data_dir.startswith("../"):
            paths_to_check.append((self.environment.travelplanner_data_dir, "TravelPlanner data directory"))
        
        for path, description in paths_to_check:
            if not Path(path).exists():
                print(f"Warning: {description} not found at {path}")
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        strategy_str = f"{self.memory.build_strategy}+{self.memory.retrieve_strategy}+{self.memory.update_strategy}"
        return f"MempConfig(strategy={strategy_str}, llm={self.llm.model}, embedding={self.embedding.model})"
