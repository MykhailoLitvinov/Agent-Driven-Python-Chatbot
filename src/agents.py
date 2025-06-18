from typing import Dict, List

from llm_client import LLMClient
from utils import load_agent_configs

CONFIG_DIR = "../config"


class AgentManager:
    """Agent manager"""

    def __init__(self, llm_client: LLMClient, default_agent: str):
        self.llm_client = llm_client
        self.default_agent = default_agent
        self.agent_configs = load_agent_configs(CONFIG_DIR)

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

    @staticmethod
    def _calculate_relevance_score(query: str, keywords: List[str]) -> float:
        """Calculate agent relevance score for a given query"""
        matches = sum(1 for keyword in keywords if keyword in query)
        return matches / len(keywords) if keywords else 0.0

    def generate_response(self, agent_name: str, context: Dict) -> str:
        """Generate a response from a specific agent"""
        if agent_name not in self.agent_configs:
            return f"Error: Agent {agent_name} not found"

        agent_config = self.agent_configs[agent_name]
        system_prompt = agent_config["system_prompt"]
        agent_temperature = float(agent_config.get("temperature", 1))

        return self.llm_client.generate_response(system_prompt, context.get("messages", []), agent_temperature)
