import os
from enum import Enum
from typing import Dict, List

import yaml

from src.llm_client import LLMClient

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


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

    def select_agent(self, query: str) -> str:
        """Select the best agent for a given query"""
        query_lower = query.lower()

        # Check for direct call (@AgentName)
        for agent_name in self.agent_configs:
            if f"@{agent_name}".lower() in query_lower:
                return agent_name

        # Automatic selection based on keyword relevance
        selected_agent = self.default_agent
        best_score = 0.0

        for agent_name, config in self.agent_configs.items():
            score = self._calculate_relevance_score(query_lower, config["keywords"])
            if score > best_score:
                selected_agent = agent_name
                best_score = score

        return selected_agent

    def generate_response(self, agent_name: str, context: Dict) -> str:
        """Generate a response from a specific agent"""
        if agent_name not in self.agent_configs:
            return f"Error: Agent {agent_name} not found"

        agent_config = self.agent_configs[agent_name]
        system_prompt = agent_config["system_prompt"]
        agent_temperature = float(agent_config.get("temperature", 1))

        return self.llm_client.generate_response(system_prompt, context.get("messages", []), agent_temperature)

    @staticmethod
    def _calculate_relevance_score(query: str, keywords: List[str]) -> float:
        """Calculate agent relevance score for a given query"""
        matches = sum(1 for keyword in keywords if keyword in query)
        return matches / len(keywords) if keywords else 0.0

    @staticmethod
    def _load_agent_configs(config_dir: str) -> dict:
        """Load agent configuration files from a directory"""
        agents = {}
        for filename in os.listdir(config_dir):
            if filename.endswith(".yaml"):
                path = os.path.join(config_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    agents[config["name"]] = config
        return agents
