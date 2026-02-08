# FILE: memp/agent/memp_agent.py

import logging
from typing import List, Dict, Any
import copy
import ast

from .base import BaseAgent
from .history import EpisodeHistory
from . import prompts
from memrl.providers.llm import OpenAILLM

logger = logging.getLogger(__name__)

class MempAgent(BaseAgent):
    """
    A stateless agent that uses an LLM to make decisions.
    It receives all necessary context (history, retrieved memories) from an
    external controller (the Runner) at the moment of action.
    """
    def __init__(self, llm_provider: OpenAILLM, few_shot_examples: Dict[str, Any]):
        # The agent is now independent of the memory service.
        self.llm = llm_provider
        self.few_shot_examples = few_shot_examples
        self.prefixes = {
            'pick_and_place': 'put',
            'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine',
            'pick_two_obj': 'puttwo'
        }

    def reset(self, task_description: str) -> None:
        """Resets the agent for a new episode and retrieves relevant long-term memories."""
        self.task_description = task_description.strip()
        logger.info(f"Agent has been reset for new task: '{self.task_description}'")
        
    def _get_examples_for_task(self, task_type: str) -> str:
        """
        [NEW] Selects the relevant few-shot examples based on the task type.
        """
        for prefix, key in self.prefixes.items():
            if task_type.startswith(prefix):
                # This logic mirrors your example script: load two relevant examples
                for example in self.few_shot_examples:
                    if example['task'] == key:
                        return copy.deepcopy(example['example'])
        return "No specific examples found for this task type."

    def _format_retrieved_memory(self, raw_content: str) -> str:
        """
        [NEW HELPER METHOD]
        Parses the raw memory content to extract only the most useful parts
        (SCRIPT and the core Thought/Action/Observation sequence), removing
        redundant headers, system prompts, and old task descriptions.
        """
        try:
            header = ""
            trajectory_str = raw_content
            # 1. Split the content into the header (Task, Script) and the trajectory part
            if 'TRAJECTORY' in raw_content:
                header, trajectory_str = raw_content.split('\n\nTRAJECTORY:\n', 1)
            elif 'Failed approach' in raw_content:
                header, trajectory_str = raw_content.split('\n\nFailed approach:\n', 1) 

            clean_parts = []
            
            # 2. Extract the high-level SCRIPT or reflection if it exists
            if 'SCRIPT:' in header:
                script_part = header.split('SCRIPT:')[1].strip()
                clean_parts.append(f"Archived Script:\n{script_part}")
            if 'What went wrong:' in header:
                reflection_part = header.split('What went wrong:')[1].strip()
                clean_parts.append(f"Archived Script:\n{reflection_part}")                
            # 3. Parse the trajectory string into a Python list
            # ast.literal_eval is a safe way to evaluate a string containing a Python literal
            trajectory_list = ast.literal_eval(trajectory_str)
            
            # 4. Keep only the portion after the last "Now, it's your turn"
            turn_idx = -1
            for i, msg in enumerate(trajectory_list):
                if msg.get("role") == "user" and isinstance(msg.get("content", ""), str) and "Now, it's your turn" in msg["content"]:
                    turn_idx = i
            if turn_idx != -1:
                trajectory_list = trajectory_list[turn_idx:]

            clean_trajectory = []
            for message in trajectory_list:
                role = message.get("role")
                content = message.get("content", "")
                if role == "assistant":
                    clean_trajectory.append(f"> {content}")
                elif role == "user" and isinstance(content, str):
                    clean_trajectory.append(content)
            
            if clean_trajectory:
                clean_parts.append("Archived Trajectory:\n" + "\n".join(clean_trajectory))
            
            return "\n\n".join(clean_parts)
            
        except Exception as e:
            logger.warning(f"Could not parse retrieved memory content, using raw content. Error: {e}")
            idx = raw_content.find('[')
            if idx != -1:
                raw_content = raw_content[idx:]
            # Fallback to returning the raw content if parsing fails
            try:
                trajectory_list = ast.literal_eval(raw_content)
            except Exception:
                return raw_content

            # 4. Keep only the portion after the last "Now, it's your turn"
            turn_idx = -1
            for i, msg in enumerate(trajectory_list):
                if msg.get("role") == "user" and isinstance(msg.get("content", ""), str) and "Now, it's your turn" in msg["content"]:
                    turn_idx = i
            if turn_idx != -1:
                trajectory_list = trajectory_list[turn_idx:]

            clean_trajectory = []
            for message in trajectory_list:
                role = message.get("role")
                content = message.get("content", "")
                if role == "assistant":
                    clean_trajectory.append(f"> {content}")
                elif role == "user" and isinstance(content, str) and content.startswith("Observation:"):
                    clean_trajectory.append(content)
            
            if clean_trajectory:
                clean_parts.append("Archived Trajectory:\n" + "\n".join(clean_trajectory))
            return "\n\n".join(clean_parts)
        
    def _construct_messages(self, task_description: str, retrieved_memories: List[Dict], task_type: str) -> List[Dict[str, str]]:
        """
        [REFACTORED]
        Builds the message list in a conversational ReAct style.
        """
        # 1. Start with the system prompt
        messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]

        # 2. Add the selected few-shot example as a complete dialogue
        example_dialogue = self._get_examples_for_task(task_type)
        if example_dialogue:
            # Modify the first user message in the example to introduce it
            example_dialogue[0]['content'] = "Here is an example of how to solve the task:\n" + example_dialogue[0]['content']
            messages.extend(example_dialogue)

        # 3. Add retrieved memories as additional context for the agent
        if retrieved_memories:
            successful_mems = retrieved_memories.get('successed', [])
            failed_mems = retrieved_memories.get('failed', [])

            successful_mems_formatted = [
                self._format_retrieved_memory(mem['content']) for mem in successful_mems
            ] if successful_mems else []

            failed_mems_formatted = [
                self._format_retrieved_memory(mem['content']) for mem in failed_mems
            ] if failed_mems else []

            memory_parts = [
                "In addition to the example, you have the following memories from your own past experiences. "
                "Use them to help you if they are relevant:"
            ]

            if successful_mems_formatted:
                memory_parts.append(
                    "--- SUCCESSFUL MEMORIES (Examples to follow) ---\n" +
                    "\n".join(successful_mems_formatted)
                )

            if failed_mems_formatted:
                memory_parts.append(
                    "--- FAILED MEMORIES (Examples to avoid or learn from) ---\n" +
                    "\n".join(failed_mems_formatted)
                )

            if successful_mems_formatted or failed_mems_formatted:
                memory_context = "\n\n".join(memory_parts)
                messages.append({"role": "system", "content": memory_context})

        # 4. Add the current task description as the new user prompt
        # The history of the current task will be appended in the `act` method
        current_task_prompt = f"Now, it's your turn to solve a new task.\n{task_description}"
        messages.append({"role": "user", "content": current_task_prompt})
        # logger.info(f"\nPrompt {messages}")
        return messages

    def _parse_action(self, llm_response: str) -> str:
        """
        Extracts the 'Action:' part from the ReAct response.
        """
        if llm_response:
            if "Action:" in llm_response:
                return llm_response.split("Action:")[-1].strip()
            # Fallback if the model doesn't follow the format correctly
            logger.warning(f"Could not find 'Action:' in LLM response. Returning the full response: '{llm_response}'")
            return llm_response.strip()
        else:
            return 'look around'
    def act(self, observation: str, history_messages: List[Dict[str, str]], first_step: bool = False):
        """
        Agent performs one step of action generation.
        Ensures robustness: if LLM fails or returns invalid output, action=None is returned.
        """
        import json

        current_messages = copy.deepcopy(history_messages)
        if not first_step:
            current_messages.append({"role": "user", "content": f"Observation: {observation.strip()}"})

        filtered_messages = []
        for i, m in enumerate(current_messages):
            if m.get("content") is None:
                logger.warning(f"[Message Filter] Message {i} has None content, removed: {m}")
                continue
            if isinstance(m.get("content"), str) and not m["content"].strip():
                logger.warning(f"[Message Filter] Message {i} has empty content, removed: {m}")
                continue
            filtered_messages.append(m)
        current_messages = filtered_messages

        logger.debug("Querying LLM for the next action...")

        response = None
        try:
            response = self.llm.generate(current_messages)
        except Exception as e:
            logger.error("LLM generation failed: %s", str(e))
            logger.error("Messages before failure:\n%s", json.dumps(current_messages, indent=2, ensure_ascii=False))
            response = None  # fallback

        if not first_step:
            history_messages.append({"role": "user", "content": f"Observation: {observation.strip()}"})
        history_messages.append({"role": "assistant", "content": response if response is not None else "No response."})

        action = None
        if response:
            try:
                action = self._parse_action(response)
            except Exception as e:
                logger.warning(f"Action parsing failed for response='{response}': {e}")
                action = "inventory"

        return action



    def get_trajectory(self) -> List[Dict[str, str]]:
        """Returns the complete trajectory for the finished episode."""
        pass
