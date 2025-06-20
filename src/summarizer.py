import json
import os
from typing import List, Dict

import yaml

from src.llm_client import LLMClient

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/system/summarizer.yaml")


class Summarizer:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.config = self._load_config(CONFIG_PATH)

    def update_summary(self, previous_summary: str, messages: List[Dict[str, str]]) -> str:
        prompt = self._build_prompt(previous_summary, messages)
        generated_response = self.llm_client.generate_response(
            system_prompt=prompt,
            messages=[],
            model=self.config["model"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            stream=self.config["stream"],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "summary_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                        },
                        "required": ["summary"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        generated_response = json.loads(generated_response)

        return generated_response["summary"]

    def _build_prompt(self, previous_summary: str, messages: List[Dict[str, str]]) -> str:
        latest_messages = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

        return self.config["system_prompt"].format(
            previous_summary=previous_summary or "None", latest_messages=latest_messages
        )

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load summarizer configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
