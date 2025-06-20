import json
import os
from enum import Enum
from typing import Dict

import yaml

from src.llm_client import LLMClient

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/interactive")
SELECTOR_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/system/agent_selector.yaml")


class AgentName(str, Enum):
    SENTINEL = "Sentinel"
    FINGUIDE = "FinGuide"
    EDUBOT = "EduBot"

    @classmethod
    def values(cls):
        return [agent.value for agent in cls]


class AgentManager:
    """Agent manager"""

    def __init__(self, llm_client: LLMClient, default_agent: AgentName):
        self.llm_client = llm_client
        self.default_agent = default_agent
        self.agent_configs = self._load_agent_configs(CONFIG_DIR)
        self.selector_config = self._load_selector_config(SELECTOR_CONFIG_PATH)

    def select_agent(self, query: str) -> str:
        """Select the best agent for a given query using LLM-based selection"""
        query_lower = query.lower()

        # Check for direct call (@AgentName)
        for agent_name in self.agent_configs:
            if f"@{agent_name}".lower() in query_lower:
                return agent_name

        # Use LLM to intelligently select the best agent
        return self._select_agent_with_llm(query)

    def _select_agent_with_llm(self, query: str) -> str:
        """Use LLM to select the most appropriate agent for the query"""
        # Create system prompt describing available agents
        agent_descriptions = []
        agent_names = list(self.agent_configs.keys())

        for agent_name, config in self.agent_configs.items():
            description = config["description"].strip()
            agent_descriptions.append(f"- {agent_name}: {description}")

        agents_info = "\n".join(agent_descriptions)

        # Use configuration for system prompt
        system_prompt = self.selector_config["system_prompt"].format(
            agents_info=agents_info, default_agent=self.default_agent
        )

        # JSON schema for structured response - only agent name
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "agent_selection",
                "schema": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "enum": agent_names,
                            "description": "The name of the selected agent",
                        }
                    },
                    "required": ["agent_name"],
                    "additionalProperties": False,
                },
            },
        }

        response = self.llm_client.generate_response(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": query}],
            model=self.selector_config["model"],
            temperature=self.selector_config["temperature"],
            max_tokens=self.selector_config["max_tokens"],
            stream=False,
            response_format=response_format,
        )

        result = json.loads(response)
        selected_agent = result.get("agent_name")

        # Validate the agent name
        if selected_agent and selected_agent in self.agent_configs:
            return selected_agent
        else:
            return self.default_agent

    def generate_response(self, agent_name: str, context: Dict) -> str:
        """Generate a response from a specific agent"""
        if agent_name not in self.agent_configs:
            return f"Error: Agent {agent_name} not found"

        agent_config = self.agent_configs[agent_name]

        system_prompt = agent_config["system_prompt"]
        if summary := context.get("summary"):
            system_prompt = f"{system_prompt}\nConversation summary: {summary}"
        agent_model = agent_config["model"]
        max_tokens = agent_config["max_tokens"]
        agent_temperature = agent_config["temperature"]

        return self.llm_client.generate_response(
            system_prompt=system_prompt,
            messages=context.get("messages", []),
            model=agent_model,
            temperature=agent_temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _load_agent_configs(config_dir: str) -> dict:
        """Load agent configuration files from a directory"""
        agents = {}
        for filename in os.listdir(config_dir):
            if filename.endswith(".yaml") and filename != "agent_selector.yaml":
                path = os.path.join(config_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    agents[config["name"]] = config
        return agents

    @staticmethod
    def _load_selector_config(config_path: str) -> dict:
        """Load agent selector configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
