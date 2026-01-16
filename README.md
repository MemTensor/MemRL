# MemRL
Paper: “MEMRL: SELF-EVOLVING AGENTS VIA RUNTIME REINFORCEMENT LEARNING ON EPISODIC MEMORY” Open-Source Code

## BigCodeBench (BCB)

Run multi-epoch BCB memory benchmark:

```bash
python run/run_bcb.py --config configs/rl_bcb_config.yaml --subset hard --split instruct --epochs 3
```

Dataset is expected under `data/bigcodebench/` (JSONL). See the error message from the runner for a download command.
Train/val split defaults to the legacy split files under `configs/bigcodebench/splits/` (override via `--split_file`).
