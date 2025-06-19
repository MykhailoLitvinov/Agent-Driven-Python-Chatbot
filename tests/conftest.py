import os
import tempfile
from unittest.mock import Mock

import pytest
import yaml

from src.llm_client import LLMClient


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing"""

    return Mock(spec=LLMClient)


@pytest.fixture
def sample_agent_configs():
    """Sample agent configurations for testing"""
    return {
        "Sentinel": {
            "name": "Sentinel",
            "system_prompt": "You are a security expert",
            "keywords": ["security", "hack", "password"],
            "model": "gpt-4o-mini",
            "temperature": 0.5,
            "max_tokens": 1000,
        },
        "FinGuide": {
            "name": "FinGuide",
            "system_prompt": "You are a financial advisor",
            "keywords": ["money", "budget", "finance"],
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 1500,
        },
        "EduBot": {
            "name": "EduBot",
            "system_prompt": "You are an educational assistant",
            "keywords": ["learn", "study", "education"],
            "model": "gpt-4o-mini",
            "temperature": 0.4,
            "max_tokens": 1200,
        },
    }


@pytest.fixture
def temp_config_dir(sample_agent_configs):
    """Create a temporary directory with agent config files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        for agent_name, config in sample_agent_configs.items():
            config_file = os.path.join(temp_dir, f"{agent_name.lower()}.yaml")
            with open(config_file, "w") as f:
                yaml.dump(config, f)
        yield temp_dir


@pytest.fixture
def sample_context():
    """Sample context for testing agent responses"""
    return {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How can you help me?"},
        ]
    }


@pytest.fixture
def sample_context_with_summary():
    """Sample context with conversation summary"""
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "summary": "Previous conversation about security best practices",
    }


@pytest.fixture
def empty_context():
    """Empty context for testing edge cases"""
    return {}


@pytest.fixture
def test_queries():
    """Common test queries for agent selection"""
    return {
        "direct_call_sentinel": "Please @Sentinel help me with security",
        "direct_call_finguide": "Can @FinGuide help with my budget?",
        "direct_call_case_insensitive": "@finguide please help",
        "security_keywords": "I need help with password security and hack prevention",
        "finance_keywords": "Help me with money and budget planning",
        "education_keywords": "I want to learn and study programming",
        "no_keywords": "Hello, how are you today?",
        "multiple_security_keywords": "security hack password firewall",
    }
