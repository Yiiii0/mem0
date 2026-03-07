import os
from typing import Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.llms.base import LLMBase
from mem0.llms.openai import OpenAILLM


class ForgeLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, OpenAIConfig, Dict]] = None):
        if isinstance(config, dict):
            config = OpenAIConfig(**config)
        super().__init__(config)
        api_key = self.config.api_key or os.getenv("FORGE_API_KEY")
        base_url = (
            getattr(self.config, "openai_base_url", None)
            or os.getenv("FORGE_BASE_URL")
            or "https://api.forge.ai/v1"
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update({"model": self.config.model, "messages": messages})
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        response = self.client.chat.completions.create(**params)
        return OpenAILLM._parse_response(self, response, tools)
