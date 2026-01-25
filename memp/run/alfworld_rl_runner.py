# memp/run/alfworld_runner.py
import logging
from pathlib import Path
from typing import Dict, Set, Any
import os
import yaml
import time
import textworld
import textworld.agents
import textworld.gym
import numpy as np
import pandas as pd
import json
import random
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from tqdm import tqdm
from .base_runner import BaseRunner
from memp.envs.alfworld_env import AlfWorldEnv
from memp.agent.memp_agent import MempAgent
from memp.agent.history import EpisodeHistory
from memp.service.memory_service import MemoryService
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv, AlfredDemangler, AlfredInfos, AlfredExpert
from memp.service.value_driven import RLConfig

MAX_RETRIES = 4 
RETRY_DELAY = 2  

logger = logging.getLogger(__name__)

def load_config_from_path(config_path: str, params=None):
    assert os.path.exists(config_path), f"Invalid config file: {config_path}"
    with open(config_path) as reader:
        config = yaml.safe_load(reader)
    if params is not None:
        for param in params:
            fqn_key, value = param.split("=")
            entry_to_change = config
            keys = fqn_key.split(".")
            for k in keys[:-1]:
                entry_to_change = entry_to_change[k]
            entry_to_change[keys[-1]] = value
    return config

class AlfworldRunner(BaseRunner):
    """
    A Runner that prepares batches of environments for a large-scale experiment.
    It handles loading, splitting the dataset, and creating all necessary
    environment instances upfront.
    """
    def __init__(self, agent: MempAgent, root: str, env_config: str, memory_service: MemoryService, exp_name: str,
                 num_section: int, batch_size: int, max_steps: int, rl_config, ck_dir:str, retrieve_k: int=1, mode: str='train',
                 valid_interval: int=2, test_interval: int=2, dataset_ratio: float=1.0, random_seed: int=42, bon: int=0,
                 ckpt_resume_enabled: bool = False, ckpt_resume_path: Optional[str] = None, ckpt_resume_epoch: Optional[int] = None,
                 baseline_mode: Optional[str] = None, baseline_k: int = 10):
        self.agent = agent
        self.root = root
        self.memory_service = memory_service
        self.exp_name = exp_name
        self.random_seed = random_seed
        self.num_section = num_section
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.retrieve_k = retrieve_k
        self.mode = mode
        self.env_config_path = env_config # Store path for AlfWorldEnv wrapper
        self.env_config = load_config_from_path(env_config) # Load config for AlfredTWEnv
        self.valid_interval = valid_interval
        self.test_interval = test_interval
        self.dataset_ratio = dataset_ratio
        self.bon = bon
        self.results_log = []
        self.ckpt_resume_enabled = ckpt_resume_enabled
        self.ckpt_resume_path = ckpt_resume_path
        self.ckpt_resume_epoch = ckpt_resume_epoch
        self.baseline_mode = (baseline_mode or "").strip().lower() or None
        self.baseline_k = max(1, int(baseline_k))
        
        self.rl_config: Optional[RLConfig] = rl_config

        # --- [Dataset] Initialize game files ---
        env_controller = AlfredTWEnv(self.env_config, train_eval='train')
        all_train_game_files = env_controller.game_files

        if not 0.0 < self.dataset_ratio <= 1.0:
            raise ValueError(f"dataset_ratio must be between 0.0 and 1.0, but got {self.dataset_ratio}")

        if self.dataset_ratio < 1.0:
            num_total_train = len(all_train_game_files)
            num_to_sample = int(num_total_train * self.dataset_ratio)
            
            logger.info(f"Randomly sampling {num_to_sample} games from the {num_total_train} training games ({self.dataset_ratio:.2%})...")
            
            # Set a seed for reproducibility of the random sample
            random.seed(self.random_seed)
            self.train_game_files = random.sample(all_train_game_files, k=num_to_sample)
        else:
            # If ratio is 1.0, use the full dataset
            logger.info(f"Using the full training set of {len(all_train_game_files)} games.")
            self.train_game_files = all_train_game_files

        env_controller = AlfredTWEnv(self.env_config, train_eval='eval_in_distribution')
        self.valid_game_files = env_controller.game_files

        env_controller = AlfredTWEnv(self.env_config, train_eval='eval_out_of_distribution')
        self.test_game_files = env_controller.game_files

        # --- [TENSORBOARD] Initialize SummaryWriter ---
        # Create a unique, timestamped directory for this experiment's logs
        tb_log_dir = self.root / "logs" / "tensorboard" / f"exp_{self.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
        self.ck_dir = ck_dir
        if self.ckpt_resume_enabled and self.ckpt_resume_path:
            resume_root = Path(self.ckpt_resume_path)
            if resume_root.name == "snapshot":
                self.ck_dir = resume_root.parent
            elif resume_root.parent.name == "snapshot":
                self.ck_dir = resume_root.parent.parent
            elif (resume_root / "snapshot").exists():
                self.ck_dir = resume_root
        self.local_cache_dir = self.ck_dir / "local_cache"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self._cum_state_path = self.local_cache_dir / "cum_state.json"
        self._cum_success_ids: Set[str] = set()
        self._cum_total = len(set(self.train_game_files))
        self._resume_section_start = self._resume_from_ckpt()

    def _analyze_and_report_results(self):
        """
        Analyzes and reports the final results for both training and evaluation,
        including success rates and average steps for all phases.
        """
        if not self.results_log:
            logger.warning("No results were logged. Cannot perform analysis.")
            return

        logger.info("\n" + "#"*20 + " FULL EXPERIMENT FINISHED - FINAL RESULTS " + "#"*20)
        results_df = pd.DataFrame(self.results_log)
        
        # --- Training Performance ---
        train_df = results_df[results_df['mode'].isin(['build', 'update'])]
        if not train_df.empty:
            overall_success_rate = train_df['success'].mean()
            logger.info("\n--- Training Performance (on Train Set) ---")
            logger.info(f"Total Training Trajectories: {len(train_df)}")
            logger.info(f"Overall Success Rate: {overall_success_rate:.2%}")

            section_performance = train_df.groupby('section').agg(
                success_rate=('success', 'mean'),
                avg_steps=('steps', 'mean')
            ).reset_index()
            logger.info("\n>>> Training Performance by Section <<<")
            print(section_performance.to_string(index=False, formatters={'success_rate': '{:.2%}'.format}))
        
        # --- Evaluation Performance ---
        eval_df = results_df[~results_df['mode'].isin(['build', 'update'])]
        if not eval_df.empty:
            logger.info("\n--- Evaluation Performance Summary ---")

            # Pivot table for Success Rate on Eval Sets
            logger.info("\n>>> Success Rate (%) by Evaluation Set <<<")
            # In eval logs, the 'success' column already holds the rate
            eval_success_summary = eval_df.pivot_table(index='after_section', columns='mode', values='success')
            with pd.option_context('display.float_format', '{:.2%}'.format):
                print(eval_success_summary)
            
            # Pivot table for Average Steps on Success on Eval Sets
            logger.info("\n>>> Average Steps on Success by Evaluation Set <<<")
            # In eval logs, the 'steps' column holds the average steps on success
            eval_steps_summary = eval_df.pivot_table(index='after_section', columns='mode', values='steps')
            with pd.option_context('display.float_format', '{:.2f}'.format):
                print(eval_steps_summary)
            
        # --- Save results to a CSV file ---
        log_dir = self.root / "logs"
        log_dir.mkdir(exist_ok=True)
        results_csv_path = log_dir / f"experiment_results_{self.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"\nDetailed results saved to: {results_csv_path}")

    def envs_spilt(
        self, 
        game_files, 
        task_type: str
    ) -> List[List[List[str]]]:
        """
        Use the full dataset for each section (no splitting by data). num_section
        only controls how many passes (sections) we run; every section sees the
        entire training set. Batching is done only by mini-batch size.
        """
        logger.info(f"Preparing full dataset batches for task type: '{task_type}'...")
        
        if not game_files:
            raise ValueError(f"No game files found for task_type '{task_type}'. Check your config paths.")

        # Each section uses the whole dataset; num_section controls number of passes.
        section_splits = [game_files for _ in range(self.num_section)]

        game_list_by_section = []
        for i, section_games in enumerate(section_splits):
            section_games = list(section_games)
            
            num_mini_batches = int(np.ceil(len(section_games) / self.batch_size))
            mini_batch_splits = []
            
            for j in range(num_mini_batches):
                start_index = j * self.batch_size
                end_index = start_index + self.batch_size
                mini_batch = section_games[start_index:end_index]
                if mini_batch:
                    mini_batch_splits.append(mini_batch)
            
            game_list_by_section.append(mini_batch_splits)
            logger.info(
                f"Section {i+1}: {len(section_games)} games, split into "
                f"{len(mini_batch_splits)} mini-batches of size <= {self.batch_size}."
            )

        return game_list_by_section
    
    def envs_built(self, mini_batch_games: List[str], task_type: str) -> List[AlfWorldEnv]:
        """
        Receives a 2D list of game files for a SINGLE section and creates a dedicated,
        parallel gym environment for each mini-batch within that section.

        Args:
            section_mini_batches (List[List[str]]): The split game files for one section,
                                                    structured as [mini_batch][game_file].
            task_type (str): The dataset split being used.

        Returns:
            List[AlfWorldEnv]: A 1D list of fully initialized AlfWorldEnv wrappers,
                               where each element is a parallel environment for one mini-batch.
        """
        logger.info(f"Building environment instances for the current section batch...")
        
        # This logic is based on the AlfredTWEnv.init_env source you provided
        domain_randomization = self.env_config["env"]["domain_randomization"]
        if task_type != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        expert_type = self.env_config["env"]["expert_type"]
        training_method = self.env_config["general"]["training_method"]

        if training_method == "dqn":
            max_nb_steps_per_episode = self.env_config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps_per_episode = self.env_config["dagger"]["training"]["max_nb_steps_per_episode"]
            expert_plan = True if task_type == "train" else False
            if expert_plan:
                wrappers.append(AlfredExpert(expert_type))
                request_infos.extras.append("expert_plan")
        else:
            raise NotImplementedError
        
            # The actual batch size for this env is the number of games in its mini-batch
        current_batch_size = len(mini_batch_games)
        env_id = textworld.gym.register_games(
            mini_batch_games, 
            request_infos,
            batch_size=current_batch_size,
            auto_reset=False,
            asynchronous=True,
            max_episode_steps=max_nb_steps_per_episode,
            wrappers=wrappers
        )
        # Launch the underlying Gym environment
        underlying_env = textworld.gym.make(env_id)
        
        # Wrap it with our AlfWorldEnv for a consistent interface
        env_wrapper = AlfWorldEnv(
            config_path=self.env_config_path, 
            preconfigured_env=underlying_env, 
            batch_size=current_batch_size
        )
        
        logger.info("Environment instances for this section batch have been built successfully.")
        return env_wrapper

    def _resolve_resume_dir(self) -> Optional[Path]:
        if not self.ckpt_resume_path:
            return None
        base = Path(self.ckpt_resume_path)
        if self.ckpt_resume_epoch is not None:
            if base.name == "snapshot":
                return base / str(self.ckpt_resume_epoch)
            if (base / "snapshot").exists():
                return base / "snapshot" / str(self.ckpt_resume_epoch)
            return base / str(self.ckpt_resume_epoch)
        return base

    def _load_cum_state(self, path: Optional[Path] = None) -> None:
        path = path or self._cum_state_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ids = payload.get("success_ids", [])
            self._cum_success_ids = {str(x) for x in ids if x}
            total = payload.get("total")
            if isinstance(total, int) and total > 0:
                self._cum_total = total
        except Exception:
            logger.warning("Failed to load cumulative acc state from %s", path, exc_info=True)

    def _persist_cum_state(self, path: Optional[Path] = None) -> None:
        path = path or self._cum_state_path
        payload = {
            "success_ids": sorted(self._cum_success_ids),
            "total": self._cum_total,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("Failed to persist cumulative acc state to %s", path, exc_info=True)

    def _update_cum_success(self, trajectories: List[Dict[str, Any]]) -> None:
        for traj in trajectories:
            key = (
                traj.get("gamefile")
                or traj.get("task_id")
                or traj.get("task_description")
            )
            if not key:
                continue
            if traj.get("success"):
                self._cum_success_ids.add(str(key))

    def _current_cum_acc(self) -> float:
        if self._cum_total <= 0:
            return 0.0
        return len(self._cum_success_ids) / self._cum_total

    def _sanitize_reflection_trajectory(self, trajectory: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        cleaned: List[Dict[str, str]] = []
        task_start_prefix = "Now, it's your turn to solve a new task."
        last_task_start_idx = None

        for idx, msg in enumerate(trajectory or []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if role != "user":
                continue
            if not isinstance(content, str):
                content = str(content)
            if content.strip().startswith(task_start_prefix):
                last_task_start_idx = idx

        if last_task_start_idx is not None:
            for msg in (trajectory or [])[last_task_start_idx + 1 :]:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    continue
                if not isinstance(content, str):
                    content = str(content)
                if content.strip().startswith("You attempted this task before."):
                    continue
                if content.strip().startswith("Here is an example of how to solve the task:"):
                    continue
                if role == "user" and content.strip().startswith("Observation:"):
                    cleaned.append({"role": "user", "content": content})
                elif role == "assistant":
                    cleaned.append({"role": "assistant", "content": content})
            return cleaned

        # Fallback when task marker is missing: keep only obs/action, drop example header.
        for msg in trajectory or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                continue
            if not isinstance(content, str):
                content = str(content)
            if content.strip().startswith("You attempted this task before."):
                continue
            if content.strip().startswith("Here is an example of how to solve the task:"):
                continue
            if role == "user" and content.strip().startswith("Observation:"):
                cleaned.append({"role": "user", "content": content})
            elif role == "assistant":
                cleaned.append({"role": "assistant", "content": content})
        return cleaned

    def _format_reflection_note(self, trajectory: List[Dict[str, Any]], success: bool) -> str:
        status = "CORRECT" if success else "INCORRECT"
        sanitized = self._sanitize_reflection_trajectory(trajectory)
        try:
            traj_text = json.dumps(sanitized, ensure_ascii=False, default=str)
        except Exception:
            traj_text = str(sanitized)
        return (
            "You attempted this task before.\n"
            f"Result: {status}\n"
            "Previous trajectory (observations/actions only):\n"
            f"{traj_text}\n\n"
            "Reflect on mistakes or improvements and solve the task again with a better plan."
        )

    def _sample_from_batch_baseline(
        self,
        mini_batch_env: AlfWorldEnv,
        *,
        reflection_notes: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        completed_experiences = []
        current_bs = mini_batch_env.batch_size
        active_slots = list(range(current_bs))
        messages_per_slot: List[List[Dict]] = [[] for _ in range(current_bs)]
        steps_per_slot: List[List[Dict]] = [0 for _ in range(current_bs)]

        results = mini_batch_env.reset()
        current_task_descs = ['\n'.join(res['obs'].split('\n\n')[1:]) for res in results]
        current_observations = ['\n'.join(res['obs'].split('\n\n')[1:]) for res in results]
        task_types = ['/'.join(res['info']['extra.gamefile'].split('/')[-3:-1]) for res in results]
        current_gamefiles = [
            res.get('info', {}).get('extra.gamefile') or res.get('info', {}).get('gamefile')
            for res in results
        ]

        for i in range(current_bs):
            messages = self.agent._construct_messages(
                task_description=current_task_descs[i],
                retrieved_memories={},
                task_type=task_types[i]
            )
            if reflection_notes:
                note = reflection_notes.get(current_gamefiles[i])
                if note:
                    messages.insert(-1, {"role": "system", "content": note})
            messages_per_slot[i] = messages

        for step in tqdm(range(self.max_steps), desc="Sampling mini-batch (baseline)"):
            if not messages_per_slot:
                break
            if not active_slots:
                logger.info("All active tasks finished. Ending batch early.")
                break
            slots_to_act_on = active_slots

            actions_dict = {}
            with ThreadPoolExecutor(max_workers=len(slots_to_act_on)) as executor:
                future_to_slot = {}
                for i in slots_to_act_on:
                    def submit_with_retry(slot_idx=i):
                        for attempt in range(1, MAX_RETRIES + 1):
                            try:
                                return self.agent.act(
                                    observation=current_observations[slot_idx],
                                    history_messages=messages_per_slot[slot_idx],
                                    first_step=(step == 0)
                                )
                            except Exception as e:
                                logger.warning(
                                    f"[Sampling Retry] Slot {slot_idx} attempt {attempt}/{MAX_RETRIES} failed: {e}"
                                )
                                if attempt < MAX_RETRIES:
                                    time.sleep(RETRY_DELAY)
                                else:
                                    logger.error(f"[Sampling Abort] Slot {slot_idx} all retries failed.")
                                    return "inventory"
                    future_to_slot[executor.submit(submit_with_retry)] = i

                for future in as_completed(future_to_slot):
                    slot_idx = future_to_slot[future]
                    try:
                        actions_dict[slot_idx] = future.result()
                    except Exception as e:
                        logger.error(f"[Sampling Fatal] Slot {slot_idx} raised unhandled exception: {e}")
                        actions_dict[slot_idx] = "inventory"

            steps_per_slot += np.ones(len(steps_per_slot))
            actions = ["look"] * current_bs
            for slot_idx, action in actions_dict.items():
                actions[slot_idx] = action

            valid_actions = []
            for i, act in enumerate(actions):
                if act is None:
                    valid_actions.append("look")
                elif not isinstance(act, str) or not act.strip():
                    valid_actions.append("look")
                else:
                    valid_actions.append(act)
            actions = valid_actions

            step_results = mini_batch_env.step(actions)

            newly_finished_slots = []
            slots_to_check = active_slots
            for i in slots_to_check:
                result = step_results[i]
                current_observations[i] = result['obs']
                info = result.get("info", {}) or {}
                gamefile = info.get("extra.gamefile") or info.get("gamefile")
                if gamefile:
                    current_gamefiles[i] = gamefile

                if result['done']:
                    success = result.get('reward', 0) > 0
                    completed_experiences.append({
                        "task_description": current_task_descs[i],
                        "trajectory": messages_per_slot[i],
                        "success": success,
                        "steps": steps_per_slot[i],
                        "gamefile": current_gamefiles[i],
                    })
                    newly_finished_slots.append(i)

            if newly_finished_slots:
                active_slots = [s for s in active_slots if s not in newly_finished_slots]

        final_slots_to_check = active_slots
        for i in final_slots_to_check:
            if messages_per_slot[i] and not any(
                exp['task_description'] == current_task_descs[i]
                and exp['trajectory'] == messages_per_slot[i]
                for exp in completed_experiences
            ):
                completed_experiences.append({
                    "task_description": current_task_descs[i],
                    "trajectory": messages_per_slot[i],
                    "success": False,
                    "steps": steps_per_slot[i],
                    "gamefile": current_gamefiles[i],
                })

        return completed_experiences

    def _run_passk_baseline(self) -> None:
        total_tasks = len(self.train_game_files)
        solved: Set[str] = set()
        summary = []
        result_path = self.local_cache_dir / "baseline_passk_results.jsonl"
        summary_path = self.local_cache_dir / "baseline_passk_summary.json"
        train_sections_data = self.envs_spilt(self.train_game_files, 'train')

        for round_idx in range(1, self.baseline_k + 1):
            logger.info("Starting pass@k round %d/%d", round_idx, self.baseline_k)
            for section_data in train_sections_data:
                for mini_batch_games in tqdm(section_data, desc=f"pass@k round {round_idx}"):
                    pending_games = [g for g in mini_batch_games if g not in solved]
                    if not pending_games:
                        continue
                    mini_batch_env = self.envs_built(pending_games, 'train')
                    trajectories = self._sample_from_batch_baseline(mini_batch_env)
                    mini_batch_env.close()
                    for traj in trajectories:
                        if traj.get("success"):
                            key = traj.get("gamefile") or traj.get("task_description")
                            if key:
                                solved.add(str(key))
                        payload = {
                            "round": round_idx,
                            "baseline": "passk",
                            **traj,
                        }
                        with open(result_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            cum_acc = (len(solved) / total_tasks) if total_tasks > 0 else 0.0
            summary.append({"round": round_idx, "cum_acc": cum_acc, "solved": len(solved), "total": total_tasks})
            logger.info("pass@k round %d cumulative acc: %.2f%%", round_idx, cum_acc * 100)
            self.writer.add_scalar("Baseline/PassK_Cumulative_Acc", cum_acc, round_idx)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _run_reflection_baseline(self) -> None:
        total_tasks = len(self.train_game_files)
        solved: Set[str] = set()
        summary = []
        result_path = self.local_cache_dir / "baseline_reflection_results.jsonl"
        summary_path = self.local_cache_dir / "baseline_reflection_summary.json"
        state_path = self.local_cache_dir / "baseline_reflection_state.json"
        train_sections_data = self.envs_spilt(self.train_game_files, 'train')
        reflection_notes: Dict[str, str] = {}

        start_round = 1
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                solved = {str(x) for x in state.get("solved", [])}
                reflection_notes = {
                    str(k): v for k, v in state.get("reflection_notes", {}).items()
                }
                last_completed = int(state.get("last_completed_round", 0))
                start_round = max(1, last_completed + 1)
                logger.info("Resuming reflection baseline from round %d", start_round)
            except Exception:
                logger.warning("Failed to load reflection baseline state from %s", state_path, exc_info=True)

        if start_round > self.baseline_k:
            logger.info("Reflection baseline already completed (last round %d).", start_round - 1)
            return

        for round_idx in range(start_round, self.baseline_k + 1):
            logger.info("Starting reflection round %d/%d", round_idx, self.baseline_k)
            for section_data in train_sections_data:
                for mini_batch_games in tqdm(section_data, desc=f"reflection round {round_idx}"):
                    pending_games = [g for g in mini_batch_games if g not in solved]
                    if not pending_games:
                        continue
                    mini_batch_env = self.envs_built(pending_games, 'train')
                    trajectories = self._sample_from_batch_baseline(
                        mini_batch_env,
                        reflection_notes=reflection_notes,
                    )
                    mini_batch_env.close()
                    for traj in trajectories:
                        key = traj.get("gamefile") or traj.get("task_description")
                        if key:
                            key = str(key)
                            reflection_notes[key] = self._format_reflection_note(
                                traj.get("trajectory", []),
                                bool(traj.get("success")),
                            )
                        if traj.get("success") and key:
                            solved.add(str(key))
                        payload = {
                            "round": round_idx,
                            "baseline": "reflection",
                            **traj,
                        }
                        with open(result_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            cum_acc = (len(solved) / total_tasks) if total_tasks > 0 else 0.0
            summary.append({"round": round_idx, "cum_acc": cum_acc, "solved": len(solved), "total": total_tasks})
            logger.info("reflection round %d cumulative acc: %.2f%%", round_idx, cum_acc * 100)
            self.writer.add_scalar("Baseline/Reflection_Cumulative_Acc", cum_acc, round_idx)
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "last_completed_round": round_idx,
                            "solved": sorted(solved),
                            "reflection_notes": reflection_notes,
                            "total": total_tasks,
                            "updated_at": datetime.now().isoformat(),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception:
                logger.warning("Failed to save reflection baseline state to %s", state_path, exc_info=True)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _resume_from_ckpt(self) -> int:
        if not self.ckpt_resume_enabled:
            return 1
        snapshot_dir = self._resolve_resume_dir()
        if not snapshot_dir or not snapshot_dir.exists():
            logger.warning("Resume enabled but snapshot not found: %s", snapshot_dir)
            return 1
        loaded = False
        if hasattr(self.memory_service, "load_checkpoint_snapshot"):
            try:
                self.memory_service.load_checkpoint_snapshot(
                    str(snapshot_dir), local_cache_dir=str(self.local_cache_dir)
                )
                loaded = True
            except TypeError:
                try:
                    self.memory_service.load_checkpoint_snapshot(str(snapshot_dir))
                    loaded = True
                except Exception:
                    logger.warning("Failed to load checkpoint snapshot from %s", snapshot_dir, exc_info=True)
            except Exception:
                logger.warning("Failed to load checkpoint snapshot from %s", snapshot_dir, exc_info=True)
        if loaded:
            self._load_cum_state(snapshot_dir / "local_cache" / "cum_state.json")
        else:
            return 1
        resume_epoch = self.ckpt_resume_epoch
        if resume_epoch is None:
            try:
                resume_epoch = int(snapshot_dir.name)
            except Exception:
                resume_epoch = None
        return (resume_epoch + 1) if resume_epoch else 1

    def process_retrieve_mems(self, retrieved_mems_per_slot):
        processed_mems_per_slot = []
        for i, mems_for_one_slot in enumerate(retrieved_mems_per_slot):
            success_mems = []
            failed_mems = []

            for mem in mems_for_one_slot:
                is_success = mem['metadata'].model_extra.get('success', False)
                if is_success:
                    success_mems.append(mem)
                else:
                    failed_mems.append(mem)

            final_mems = {}
            if success_mems:
                final_mems['successed'] = success_mems
            if failed_mems:
                final_mems['failed'] = failed_mems

            processed_mems_per_slot.append(final_mems)

        return processed_mems_per_slot

    def _sample_from_batch(self, mini_batch_env: AlfWorldEnv) -> List[Dict]:
        """
        Runs one parallel environment (a mini-batch), managing the full conversational
        history (messages list) for each parallel game and feeding it to the ReAct agent.
        """
        completed_experiences = []
        current_bs = mini_batch_env.batch_size
        active_slots = list(range(current_bs))
        messages_per_slot: List[List[Dict]] = [[] for _ in range(current_bs)]
        steps_per_slot: List[List[Dict]] = [0 for _ in range(current_bs)]

        results = mini_batch_env.reset()
        current_task_descs = ['\n'.join(res['obs'].split('\n\n')[1:]) for res in results]
        # The first observation is part of the initial prompt, not a separate step
        current_observations = ['\n'.join(res['obs'].split('\n\n')[1:]) for res in results]
        task_types = ['/'.join(res['info']['extra.gamefile'].split('/')[-3:-1]) for res in results]
        current_gamefiles = [
            res.get('info', {}).get('extra.gamefile') or res.get('info', {}).get('gamefile')
            for res in results
        ]

        # --- Retrieve initial memories for the batch ---
        logger.info(f"Retrieving initial memories (k={self.retrieve_k}) for the batch in parallel...")
        with ThreadPoolExecutor(max_workers=current_bs) as executor:
            future_re_mems = [
                executor.submit(
                    self.memory_service.retrieve_query,
                    desc,
                    k=self.retrieve_k,
                    # Align retrieval threshold knob across benchmarks: rl_config.sim_threshold (fallback tau).
                    threshold=getattr(self.rl_config, "sim_threshold", getattr(self.rl_config, "tau", 0.0))
                )
                for desc in current_task_descs
            ]

            retrieved_mems_per_slot = []
            retrieved_queries_per_slot = []

            for future in future_re_mems:
                result = future.result()
                if isinstance(result, tuple): 
                    mem, topk_queries = result[0]['selected'], result[1]
                else:
                    mem, topk_queries = [], []

                retrieved_mems_per_slot.append(mem)
                retrieved_queries_per_slot.append(topk_queries)

        retrieved_mems_per_slot = self.process_retrieve_mems(retrieved_mems_per_slot)

        logger.info("Constructing initial ReAct prompts for each game...")
        for i in range(current_bs):
            messages_per_slot[i] = self.agent._construct_messages(
                task_description=current_task_descs[i],
                retrieved_memories=retrieved_mems_per_slot[i],
                task_type=task_types[i]
            )

        for step in tqdm(range(self.max_steps), desc="Sampling mini-batch (ReAct)"):
            if not messages_per_slot: # Break if all tasks are somehow finished
                break
            if not active_slots:
                    logger.info("All active tasks finished. Ending batch early.")
                    break
            # --- Determine which slots need an action ---
            slots_to_act_on = active_slots
            
            actions_dict = {}

            with ThreadPoolExecutor(max_workers=len(slots_to_act_on)) as executor:
                future_to_slot = {}

                for i in slots_to_act_on:
                    def submit_with_retry(slot_idx=i):
                        for attempt in range(1, MAX_RETRIES + 1):
                            try:
                                return self.agent.act(
                                    observation=current_observations[slot_idx],
                                    history_messages=messages_per_slot[slot_idx],
                                    first_step=(step == 0)
                                )
                            except Exception as e:
                                logger.warning(
                                    f"[Sampling Retry] Slot {slot_idx} attempt {attempt}/{MAX_RETRIES} failed: {e}"
                                )
                                if attempt < MAX_RETRIES:
                                    time.sleep(RETRY_DELAY)
                                else:
                                    logger.error(f"[Sampling Abort] Slot {slot_idx} all retries failed.")
                                    return "inventory"

                    future_to_slot[executor.submit(submit_with_retry)] = i

                for future in as_completed(future_to_slot):
                    slot_idx = future_to_slot[future]
                    try:
                        actions_dict[slot_idx] = future.result()
                    except Exception as e:
                        logger.error(f"[Sampling Fatal] Slot {slot_idx} raised unhandled exception: {e}")
                        actions_dict[slot_idx] = "inventory"

            steps_per_slot += np.ones(len(steps_per_slot))
            
            actions = ["look"] * current_bs
            for slot_idx, action in actions_dict.items():
                actions[slot_idx] = action

            valid_actions = []
            for i, act in enumerate(actions):
                if act is None:
                    logger.warning(f"[Sampling Warning] Slot {i} action is None, replaced with 'look'.")
                    valid_actions.append("look")
                elif not isinstance(act, str) or not act.strip():
                    logger.warning(f"[Sampling Warning] Slot {i} invalid action '{act}', replaced with 'look'.")
                    valid_actions.append("look")
                else:
                    valid_actions.append(act)
            actions = valid_actions

            step_results = mini_batch_env.step(actions)
            # --- Result processing and state update ---

            newly_finished_slots = []
            
            slots_to_check = active_slots

            for i in slots_to_check:
                result = step_results[i]
                current_observations[i] = result['obs']
                info = result.get("info", {}) or {}
                gamefile = info.get("extra.gamefile") or info.get("gamefile")
                if gamefile:
                    current_gamefiles[i] = gamefile

                if result['done']:
                    success = result.get('reward', 0) > 0
                    logger.info(f"Slot {i} finished a game. Success: {success}")
                    
                    completed_experiences.append({
                        "task_description": current_task_descs[i],
                        "trajectory": messages_per_slot[i], # The full conversation is the trajectory
                        "success": success,
                        "retrieved_queries": retrieved_queries_per_slot[i],
                        "retrieved_mems": retrieved_mems_per_slot[i],
                        "steps": steps_per_slot[i],
                        "gamefile": current_gamefiles[i],
                    })
                    
                    newly_finished_slots.append(i)
            if newly_finished_slots:
                active_slots = [s for s in active_slots if s not in newly_finished_slots]
        
        # Handle incomplete trajectories
        final_slots_to_check = active_slots
        for i in final_slots_to_check:
            # A trajectory is incomplete if its message history exists but the game isn't marked done
            if messages_per_slot[i] and not any(exp['task_description'] == current_task_descs[i] and exp['trajectory'] == messages_per_slot[i] for exp in completed_experiences):
                completed_experiences.append({
                    "task_description": current_task_descs[i],
                    "trajectory": messages_per_slot[i],
                    "success": False,
                    "retrieved_queries": retrieved_queries_per_slot[i],
                    "retrieved_mems": retrieved_mems_per_slot[i],
                    "steps": steps_per_slot[i],
                    "gamefile": current_gamefiles[i],
                })

        return completed_experiences

    def _evaluate(self, game_files: List[str], eval_type: str, after_section: int) -> float:
        """
        Runs the agent on a given dataset split for evaluation purposes only.
        No memory building or updating occurs.

        Args:
            game_files (list): list of game files
            eval_type (str): A string identifier for logging ('Validation' or 'Test').
            after_setcion (int): num of current section in the train loop
        Returns:
            float: The success rate on the evaluation set.
        """

        num_mini_batches = int(np.ceil(len(game_files) / self.batch_size))
        section_mini_batches = [
            game_files[i*self.batch_size : (i+1)*self.batch_size]
            for i in range(num_mini_batches) if game_files[i*self.batch_size : (i+1)*self.batch_size]
        ]

        if not section_mini_batches:
            logger.warning(f"No games to evaluate for {eval_type}.")
            return

        
        # --- Sample from each environment ---
        eval_trajectories = []
        for i, mini_batch_games in tqdm(enumerate(section_mini_batches), desc=f"Evaluating on {eval_type}"):
            mini_batch_env = None
            try:
                mini_batch_env = self.envs_built(mini_batch_games, task_type=eval_type)
                collected_trajs = self._sample_from_batch(mini_batch_env)
                eval_trajectories.extend(collected_trajs)
            finally:
                try:
                    if mini_batch_env is not None:
                        mini_batch_env.close()
                except Exception:
                    logger.debug("Failed to close eval mini_batch_env", exc_info=True)

        if not eval_trajectories:
            logger.warning(f"No trajectories were collected during {eval_type} evaluation.")
            self.writer.add_scalar(f"Evaluation/Success_Rate/{eval_type}", 0.0, after_section)
            self.writer.add_scalar(f"Evaluation/Avg_Steps/{eval_type}", 0.0, after_section)
            return

        # --- Calculate metrics and log the results ---
        successes = sum(1 for traj in eval_trajectories if traj["success"])
        success_rate = successes / len(eval_trajectories) if eval_trajectories else 0.0
        
        # Calculate average steps
        avg_steps = np.mean([traj["steps"] for traj in eval_trajectories])
        
        logger.info(f"--- Evaluation Complete on {eval_type} (after training Section {after_section}) ---")
        logger.info(f"Success Rate: {success_rate:.2%} ({successes}/{len(eval_trajectories)})")
        logger.info(f"Average Steps on Success: {avg_steps:.2f}") # Also print to console
        
        # --- [TENSORBOARD] Log both metrics ---
        self.writer.add_scalar(f"Evaluation/Success_Rate/{eval_type}", success_rate, after_section)
        self.writer.add_scalar(f"Evaluation/Avg_Steps/{eval_type}", avg_steps, after_section)
        
        # Log this result for the final text report
        self.results_log.append({
            "section": f"eval_s{after_section}",
            "after_section": after_section,
            "mode": eval_type,
            "success": success_rate,
            "steps": avg_steps # Store the new metric
        })

    def _evaluate_bon(self, game_files: List[str], eval_type: str) -> float:
        """
        Evaluate with Best-of-N strategy.
        Runs multiple evaluation passes (BoN) and logs success rates as a curve.
        """

        num_mini_batches = int(np.ceil(len(game_files) / self.batch_size))
        section_mini_batches = [
            game_files[i*self.batch_size : (i+1)*self.batch_size]
            for i in range(num_mini_batches)
            if game_files[i*self.batch_size : (i+1)*self.batch_size]
        ]

        if not section_mini_batches:
            logger.warning(f"No games to evaluate for {eval_type}.")
            return

        # --- BoN Config ---
        bon_samples = self.bon
        logger.info(f"Starting Best-of-{bon_samples} evaluation for {eval_type}...")

        all_eval_trajectories = [[] for _ in range(len(game_files))]

        # --- mutil-run ---
        for b in range(bon_samples):
            logger.info(f"[BoN] Sampling round {b+1}/{bon_samples}...")
            eval_trajectories = []
            for i, mini_batch_game in tqdm(enumerate(section_mini_batches), desc=f"Evaluating round {b+1}/{bon_samples}"):
                mini_batch_env = None
                try:
                    mini_batch_env = self.envs_built(mini_batch_game, task_type=eval_type)
                    collected_trajs = self._sample_from_batch(mini_batch_env)
                    eval_trajectories.extend(collected_trajs)
                finally:
                    try:
                        if mini_batch_env is not None:
                            mini_batch_env.close()
                    except Exception:
                        logger.debug("Failed to close BoN eval mini_batch_env", exc_info=True)

            for index, trajs in enumerate(eval_trajectories):
                all_eval_trajectories[index].append(trajs)

            # --- compute Best-of-(b+1) sr ---

            aggregated_results = []
            for index, trajs in enumerate(all_eval_trajectories):
                success = any(t["success"] for t in trajs)
                if success:
                    steps = min(t["steps"] for t in trajs if t["success"])
                else:
                    steps = min(t["steps"] for t in trajs)
                aggregated_results.append({"task_index": index, "success": success, "steps": steps})

            num_tasks = len(aggregated_results)
            success_rate = sum(t["success"] for t in aggregated_results) / num_tasks
            avg_steps = np.mean([t["steps"] for t in aggregated_results])

            logger.info(f"[{eval_type}] Best-of-{b+1} Success rate: {success_rate:.3f}, Avg steps: {avg_steps:.2f}")

            self.writer.add_scalar(f"Evaluation/BoN_Success_Rate/{eval_type}", success_rate, b+1)
            self.writer.add_scalar(f"Evaluation/BoN_Avg_Steps/{eval_type}", avg_steps, b+1)

        logger.info(f"--- Evaluation Complete on {eval_type} ---")
        logger.info(f"Final Best-of-{bon_samples} Success Rate: {success_rate:.2%}")
        logger.info(f"Final Best-of-{bon_samples} Avg Steps: {avg_steps:.2f}")

        self.results_log.append({
            "mode": eval_type,
            "success": success_rate,
            "steps": avg_steps,
            "bon": bon_samples
        })

    def run(self):
        """
        The main experiment execution flow, featuring section and batch loops.
        """
        start_section = self._resume_section_start
        skip_initial_eval = start_section > 1
        if self.baseline_mode in {"passk", "reflection"}:
            if self.baseline_mode == "passk":
                self._run_passk_baseline()
            else:
                self._run_reflection_baseline()
            self.writer.close()
            return
        if self.bon > 0:
            logger.info("eval spilt")
            self._evaluate_bon(
                game_files=self.valid_game_files,
                eval_type="eval_in_distribution"
            )
            # logger.info("test spilt")
            # self._evaluate_bon(
            #     game_files=self.test_game_files,
            #     eval_type="eval_out_of_distribution"
            # )
        else:
            if self.mode != 'test': 
                
                logger.info("eval spilt")
                if not skip_initial_eval:
                    self._evaluate(
                            game_files=self.valid_game_files,
                            eval_type="eval_in_distribution", 
                            after_section=0
                        )
                # logger.info("test spilt")
                # self._evaluate(
                #         game_files=self.test_game_files,
                #         eval_type="eval_out_of_distribution", 
                #         after_section=0
                #     )        

    # --- Loop: Iterate through Sections ---
        if self.bon == 0 : # do not test bon on val set.
            # 1. Prepare data splits
            train_sections_data = self.envs_spilt(self.train_game_files, 'train')

            for section_idx, section_data in enumerate(train_sections_data):
                section_num = section_idx + 1
                if section_num < start_section:
                    logger.info("Skipping section %d due to resume.", section_num)
                    continue

                logger.info("\n" + "#"*20 + f" STARTING SECTION {section_num}/{self.num_section}" + "#"*20)

                section_trajectories = []

                # --- Inner Loop: Iterate through mini-batches (environments) ---
                for i, mini_batch_games in tqdm(enumerate(section_data)):
                    logger.info(f"Processing mini-batch {i+1}/{len(section_data)} in section {section_num}...")

                    # Collect trajectories
                    mini_batch_env = None
                    try:
                        mini_batch_env = self.envs_built(mini_batch_games, 'train')
                        collected_trajs = self._sample_from_batch(mini_batch_env)
                    finally:
                        try:
                            if mini_batch_env is not None:
                                mini_batch_env.close()
                        except Exception:
                            logger.debug("Failed to close mini_batch_env", exc_info=True)

                    logger.info(f"Mini-batch {i+1} collected {len(collected_trajs)} trajectories.")
                    section_trajectories.extend(collected_trajs)

                    # --- Memory Processing for this mini-batch ---
                    task_descriptions = [traj["task_description"] for traj in collected_trajs]
                    trajectories = [traj['trajectory'] for traj in collected_trajs]
                    successes = [traj["success"] for traj in collected_trajs]

                    retrieved_ids_list = [
                        [
                            mem["memory_id"]
                            for mem_list in traj["retrieved_mems"].values()
                            for mem in mem_list
                            if "memory_id" in mem
                        ]
                        for traj in collected_trajs
                    ]

                    retrieved_queries = [traj["retrieved_queries"] for traj in collected_trajs]

                    # update q value for retrieved mems
                    updated_q_list = self.memory_service.update_values(successes, retrieved_ids_list)
                    logger.info(f"Updated Q-values for mini-batch {i+1}: {updated_q_list}")

                    metadatas_update = [
                        {
                            "source_benchmark": "alfworld_build",
                            "success": traj["success"],
                            "q_value": float(self.rl_config.q_init_pos) if traj['success'] else float(self.rl_config.q_init_neg),
                            "q_visits": 0,
                            "q_updated_at": datetime.now().isoformat(),
                            "last_used_at": datetime.now().isoformat(),
                            "reward_ma": 0.0,
                        }
                        for traj in collected_trajs
                    ]

                    self.memory_service.add_memories(
                        task_descriptions=task_descriptions,
                        trajectories=trajectories,
                        successes=successes,
                        retrieved_memory_queries=retrieved_queries,
                        retrieved_memory_ids_list=retrieved_ids_list,
                        metadatas=metadatas_update
                    )

                    logger.info(f"Mini-batch {i+1} memory update complete.")

                logger.info(f"Section {section_num} complete. Total {len(section_trajectories)} trajectories collected.")

                
                self._update_cum_success(section_trajectories)
                cum_acc = self._current_cum_acc()
                self._persist_cum_state()
                logger.info("Section %d Cumulative Acc: %.2f%%", section_num, cum_acc * 100)
                self.writer.add_scalar("Train/Cumulative_Success_Rate", cum_acc, section_num)
                self.results_log.append({
                    "section": section_num,
                    "mode": "train_cumulative",
                    "success": cum_acc,
                    "steps": None,
                })

                try:
                    ckpt_meta = self.memory_service.save_checkpoint_snapshot(
                        self.ck_dir, ckpt_id=section_num, local_cache_dir=self.local_cache_dir
                    )
                except TypeError:
                    ckpt_meta = self.memory_service.save_checkpoint_snapshot(
                        self.ck_dir, ckpt_id=section_num
                    )
                logger.info(f" Saved ckpt: {ckpt_meta}")
                snapshot_root = Path(self.ck_dir) / "snapshot" / str(section_num)
                self._persist_cum_state(snapshot_root / "local_cache" / "cum_state.json")
                # --- Log results for this section ---
                for traj_data in section_trajectories:
                    self.results_log.append({
                        "section": section_num,
                        "success": traj_data["success"],
                        "steps": traj_data["steps"],
                    })

                # --- [TENSORBOARD] Log training metrics for this section ---
                if section_trajectories:
                    section_success = sum(1 for traj in section_trajectories if traj["success"])
                    section_success_rate = section_success / len(section_trajectories)
                    section_avg_steps = np.mean([traj["steps"] for traj in section_trajectories if traj['trajectory']])
                    
                    self.writer.add_scalar("Train/Section_Success_Rate", section_success_rate, section_num)
                    self.writer.add_scalar("Train/Section_Avg_Steps", section_avg_steps, section_num)
                    logger.info(f"Section {section_num} Training Stats: Success Rate={section_success_rate:.2%}, Avg Steps={section_avg_steps:.2f}")

                if self.mode != 'test':
                    if self.valid_interval > 0 and section_num % self.valid_interval == 0:
                        self._evaluate(
                            game_files=self.valid_game_files,
                            eval_type="eval_in_distribution", 
                            after_section=section_num
                        )

                    # # Check if it's time to run evaluation on the test set
                    # if self.test_interval > 0 and section_num % self.test_interval == 0:
                    #     self._evaluate(
                    #         game_files=self.test_game_files,
                    #         eval_type="eval_out_of_distribution", 
                    #         after_section=section_num
                    #     )

            # Final analysis at the end of all sections
            self._analyze_and_report_results()
        # --- [TENSORBOARD] Close the writer ---
        self.writer.close()
