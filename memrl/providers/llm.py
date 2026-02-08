"""
OpenAI-compatible LLM provider implementation.

This module provides concrete implementations of the BaseLLM interface
for OpenAI and OpenAI-compatible services.
"""

import json
import re
import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except Exception:  # fallback if tenacity is unavailable
    def retry(*args, **kwargs):
        def deco(fn):
            return fn
        return deco
    def stop_after_attempt(*args, **kwargs):
        return None
    def wait_exponential(*args, **kwargs):
        return None
    def retry_if_exception_type(*args, **kwargs):
        return None
import logging

from .base import BaseLLM, LLMError


class OpenAILLM(BaseLLM):
    """
    OpenAI-compatible LLM provider.
    
    Supports both OpenAI's official API and any OpenAI-compatible services
    (like local models served via vLLM, ollama, etc.).
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        default_temperature: float = 0.7,
        default_max_tokens: Optional[int] = None,
        token_log_dir: Optional[str] = None,
        token_log_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize OpenAI LLM provider.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API (None for official OpenAI)
            model: Model name to use
            default_temperature: Default temperature for generation
            default_max_tokens: Default max tokens for generation
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)

        # Validate API key
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be empty")

        self.model = model
        self.base_url = base_url
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._token_log_lock = threading.Lock()
        self._token_log_path = self._resolve_token_log_path(token_log_path, token_log_dir)

        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            
        try:
            self.client = OpenAI(**client_kwargs)
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {e}")

    @staticmethod
    def _resolve_token_log_path(
        token_log_path: Optional[str],
        token_log_dir: Optional[str],
    ) -> Path:
        if token_log_path:
            path = Path(token_log_path)
        else:
            base_dir = token_log_dir or "local_cache"
            path = Path(base_dir) / "token_usage.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _usage_details_to_dict(details: Any) -> Dict[str, Any]:
        if details is None:
            return {}
        fields = [
            "reasoning_tokens",
            "audio_tokens",
            "text_tokens",
            "image_tokens",
            "cached_tokens",
            "accepted_prediction_tokens",
            "rejected_prediction_tokens",
        ]
        payload: Dict[str, Any] = {}
        for key in fields:
            val = getattr(details, key, None)
            if val is not None:
                payload[key] = val
        if payload:
            return payload
        try:
            if isinstance(details, dict):
                return details
            if hasattr(details, "model_dump"):
                return details.model_dump()
        except Exception:
            return {}
        return {}

    def _usage_to_dict(self, usage: Any) -> Dict[str, Any]:
        if usage is None:
            return {}
        payload: Dict[str, Any] = {}
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"]:
            val = getattr(usage, key, None)
            if val is not None:
                payload[key] = val
        completion_details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if completion_details is not None:
            payload["completion_tokens_details"] = self._usage_details_to_dict(completion_details)
        if prompt_details is not None:
            payload["prompt_tokens_details"] = self._usage_details_to_dict(prompt_details)
        if payload:
            return payload
        try:
            if isinstance(usage, dict):
                return usage
            if hasattr(usage, "model_dump"):
                return usage.model_dump()
        except Exception:
            return {}
        return {}

    @staticmethod
    def _summarize_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_chars = 0
        image_items = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        total_chars += len(item.get("text", "") or "")
                    elif item.get("type") == "image_url":
                        image_items += 1
        return {
            "messages_count": len(messages),
            "prompt_chars": total_chars,
            "image_items": image_items,
        }

    def _log_token_usage(self, payload: Dict[str, Any]) -> None:
        if not self._token_log_path:
            return
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}
        try:
            text = json.dumps(entry, ensure_ascii=False, default=str)
        except Exception:
            text = json.dumps({"ts": entry.get("ts"), "payload": str(payload)}, ensure_ascii=False)
        with self._token_log_lock:
            with open(self._token_log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Generate response using OpenAI Chat Completions API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response text
            
        Raises:
            LLMError: If generation fails after retries
        """
        # Merge default parameters with provided kwargs.
        #
        # IMPORTANT (LLB compatibility):
        # - LLB may not pass max_tokens per-call.
        # - In that case we must honor the configured default_max_tokens from YAML,
        #   otherwise the backend's implicit default can truncate generations.
        generation_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.default_temperature),
        }

        # Accept either OpenAI-style `max_tokens` or LLB-style `max_completion_tokens`.
        if "max_tokens" not in kwargs and "max_completion_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.get("max_completion_tokens")

        if "max_tokens" in kwargs:
            generation_kwargs["max_tokens"] = kwargs.get("max_tokens")
        elif self.default_max_tokens is not None:
            generation_kwargs["max_tokens"] = self.default_max_tokens
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in generation_kwargs:
                generation_kwargs[key] = value
        
        try:
            response = self.client.chat.completions.create(**generation_kwargs)
            # Inspect finish_reason and usage for diagnostics
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            if finish_reason in {"length", "content_filter"}:
                logging.warning(
                    "LLM generation stopped early (finish_reason=%s). Consider increasing max_tokens (now=%s).",
                    finish_reason,
                    generation_kwargs.get("max_tokens"),
                )
            # Optionally expose basic token usage in debug logs
            if usage is not None:
                logging.debug(
                    "LLM usage: prompt_tokens=%s, completion_tokens=%s, total_tokens=%s",
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "completion_tokens", None),
                    getattr(usage, "total_tokens", None),
                )
            try:
                usage_payload = self._usage_to_dict(usage)
                self._log_token_usage(
                    {
                        "provider": "llm",
                        "model": getattr(response, "model", self.model),
                        "base_url": self.base_url,
                        "request_params": {k: v for k, v in generation_kwargs.items() if k != "messages"},
                        "prompt_stats": self._summarize_messages(messages),
                        "usage": usage_payload,
                        "finish_reason": finish_reason,
                    }
                )
            except Exception:
                logging.debug("Failed to log token usage", exc_info=True)
            return content
        except Exception as e:
            status = None
            try:
                status = getattr(getattr(e, "response", None), "status_code", None)
            except Exception:
                status = None
            try:
                self._log_token_usage(
                    {
                        "provider": "llm",
                        "model": self.model,
                        "base_url": self.base_url,
                        "request_params": {k: v for k, v in generation_kwargs.items() if k != "messages"},
                        "prompt_stats": self._summarize_messages(messages),
                        "error": str(e),
                        "status": status,
                    }
                )
            except Exception:
                logging.debug("Failed to log token usage on error", exc_info=True)
            logging.error(
                "LLM request failed (model=%s, status=%s, error=%s): %s",
                self.model,
                status,
                e.__class__.__name__,
                e,
                exc_info=True,
            )
            raise LLMError(f"Failed to generate response: {e}") from e
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        """
        Extract keywords from text using LLM.
        
        This method uses the LLM to identify key concepts that can be used
        for the AveFact retrieval strategy.
        
        Args:
            text: Input text to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
            
        Raises:
            LLMError: If keyword extraction fails
        """
        prompt = f"""
        Extract up to {max_keywords} key concepts or keywords from the following text.
        Focus on the most important nouns, actions, and specific entities.
        Return only the keywords separated by commas, nothing else.
        
        Text: {text}
        
        Keywords:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.generate(messages, temperature=0, max_tokens=100)
            
            # Parse keywords from response
            keywords_text = response.strip()
            
            # Split by commas and clean up
            keywords = []
            for keyword in keywords_text.split(','):
                keyword = keyword.strip().lower()
                # Remove quotes and extra whitespace
                keyword = re.sub(r'^["\']|["\']$', '', keyword)
                keyword = re.sub(r'\s+', ' ', keyword)
                
                if keyword and len(keyword) > 1:  # Filter out single characters
                    keywords.append(keyword)
            
            return keywords[:max_keywords]
            
        except Exception as e:
            raise LLMError(f"Failed to extract keywords: {e}")
    
    def generate_script(self, trajectory: str) -> str:
        """
        Generate high-level script from trajectory.
        
        Args:
            trajectory: Detailed task trajectory
            
        Returns:
            High-level script representation
        """
        prompt = f"""
        Analyze the following detailed task trajectory and create a concise, 
        high-level script that captures the essential steps and decision points.
        
        The script should be:
        1. Generic enough to apply to similar tasks
        2. Specific enough to provide useful guidance
        3. 3-5 high-level steps maximum
        4. Focus on the strategy and key decisions, not detailed actions
        
        Trajectory:
        {trajectory}
        
        High-level script:"""
        
        messages = [{"role": "user", "content": prompt}]
        return self.generate(messages, temperature=self.default_temperature, max_tokens=self.default_max_tokens)


class MockLLM(BaseLLM):
    """
    Mock LLM provider for testing purposes.
    
    This provider returns predefined responses and is useful for
    unit testing without making actual API calls.
    """
    
    def __init__(self, responses: Optional[Dict[str, str]] = None, **kwargs: Any) -> None:
        """
        Initialize mock LLM provider.
        
        Args:
            responses: Dictionary mapping input patterns to responses
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.responses = responses or {}
        self.call_count = 0
    
    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generate mock response."""
        self.call_count += 1
        
        # Extract the user message content
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        
        # Check for predefined responses
        for pattern, response in self.responses.items():
            if pattern.lower() in user_content.lower():
                return response
        
        # Default response
        return f"Mock response {self.call_count} for: {user_content[:50]}..."
    
    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        """Extract mock keywords."""
        # Simple keyword extraction for testing
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3][:max_keywords]
        return keywords if keywords else ["test", "keyword"]
