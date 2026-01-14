from alfworld.agents.environment import get_environment
import os
from .base import IEnv
import yaml
from typing import Optional

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


class AlfWorldEnv(IEnv):
    """
    ALFWorld wrapper environment for training memory agents with support for batch_size > 1.
    """

    def __init__(self, config_path: str, preconfigured_env: Optional[object] = None, 
                 task_type: str = "train", batch_size: int = 1):
        """
        Args:
            config_path (str): Path to the main ALFWorld config YAML.
            preconfigured_env (Optional[object]): If provided, the wrapper will use this
                                                   environment instance instead of creating its own.
            task_type (str): The dataset split to use if creating a default env.
            batch_size (int): The batch size to use if creating a default env.
        """
        self.config_path = config_path
        self.batch_size = batch_size
        self.current_trace_list = [[] for _ in range(batch_size)]
        
        if preconfigured_env:
            # If a specific environment is passed in, use it directly.
            self.env = preconfigured_env
        else:
            # Otherwise, create a default environment that samples from the whole split.
            config = load_config_from_path(config_path)
            env_type = config['env']['type']
            underlying_env_controller = get_environment(env_type)(config, train_eval=task_type)
            self.env = underlying_env_controller.init_env(batch_size=batch_size)

    def reset(self):
        """
        Reset environments and return first observations.
        
        Returns:
            list[dict]: [{"obs": str, "info": dict}, ...] (length=batch_size)
        """
        obs_list, info = self.env.reset()
        self.initial_obs = obs_list
        self.current_trace_list = [[] for _ in range(self.batch_size)]

        results = []
        for i, obs in enumerate(obs_list):
            per_env_info = {k: v[i] for k, v in info.items()}
            results.append({
                "obs": self._process_obs(obs),
                "info": per_env_info
            })
        return results



    def step(self, actions: list):
        """
        Execute one step in the environments.

        Args:
            actions (list[str]): textual actions for each environment

        Returns:
            list[dict]: [{"obs": str, "reward": float, "done": bool, "info": dict}, ...]
        """
        assert isinstance(actions, list), "When batch_size>1, actions must be a list"

        obs_list, reward_list, done_list, infos = self.env.step(actions)

        results = []
        for i, (obs, reward, done, action) in enumerate(zip(obs_list, reward_list, done_list, actions)):
            info_i = {k: v[i] for k, v in infos.items()}

            step_data = {
                "obs": self._process_obs(obs),
                "reward": reward,
                "done": done,
                "info": info_i
            }
            self.current_trace_list[i].append({"action": action, **step_data})
            results.append(step_data)
        return results


    def current_trace(self, env_id=None):
        """Return full trajectory for each env (or one env if env_id provided)."""
        if env_id is None:
            return self.current_trace_list
        return self.current_trace_list[env_id]

    def close(self):
        """Close ALFWorld env."""
        self.env.close()

    def _process_obs(self, obs):
        """
        Convert ALFWorld raw observation into a format suitable for memory agent.
        Default: return plain text.
        """
        if isinstance(obs, list):
            return " ".join(obs)
        return str(obs)