import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from src.agents import AgentName, AgentManager


def test_agent_name_values():
    """Test that all expected agent names are present"""
    assert AgentName.SENTINEL == "Sentinel"
    assert AgentName.FINGUIDE == "FinGuide"
    assert AgentName.EDUBOT == "EduBot"


def test_agent_name_values_class_method():
    """Test the values class method returns all agent names"""
    expected_values = ["Sentinel", "FinGuide", "EduBot"]
    assert AgentName.values() == expected_values


@pytest.fixture
def agent_manager(mock_llm_client, sample_agent_configs, sample_selector_config):
    """Create an AgentManager instance with mocked dependencies"""
    with patch.object(AgentManager, "_load_agent_configs", return_value=sample_agent_configs), patch.object(
        AgentManager, "_load_selector_config", return_value=sample_selector_config
    ):
        return AgentManager(mock_llm_client, AgentName.SENTINEL)


def test_agent_manager_init(mock_llm_client, sample_agent_configs, sample_selector_config):
    """Test AgentManager initialization"""
    with patch.object(
        AgentManager, "_load_agent_configs", return_value=sample_agent_configs
    ) as mock_load_agents, patch.object(
        AgentManager, "_load_selector_config", return_value=sample_selector_config
    ) as mock_load_selector:
        manager = AgentManager(mock_llm_client, AgentName.SENTINEL)

        assert manager.llm_client == mock_llm_client
        assert manager.default_agent == AgentName.SENTINEL
        assert manager.agent_configs == sample_agent_configs
        assert manager.selector_config == sample_selector_config
        mock_load_agents.assert_called_once()
        mock_load_selector.assert_called_once()


def test_select_agent_direct_call(agent_manager, test_queries):
    """Test agent selection with direct call (@AgentName)"""
    result = agent_manager.select_agent(test_queries["direct_call_finguide"])
    assert result == "FinGuide"


def test_generate_response_success(agent_manager, mock_llm_client, sample_context):
    """Test successful response generation"""
    mock_llm_client.generate_response.return_value = "Test response"

    result = agent_manager.generate_response("Sentinel", sample_context)

    assert result == "Test response"
    mock_llm_client.generate_response.assert_called_once_with(
        system_prompt="You are a security expert",
        messages=sample_context["messages"],
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=1000,
    )


def test_generate_response_with_summary(agent_manager, mock_llm_client, sample_context_with_summary):
    """Test response generation with conversation summary"""
    mock_llm_client.generate_response.return_value = "Test response"

    result = agent_manager.generate_response("Sentinel", sample_context_with_summary)

    expected_system_prompt = (
        "You are a security expert\nConversation summary: Previous conversation about security best practices"
    )
    mock_llm_client.generate_response.assert_called_once_with(
        system_prompt=expected_system_prompt,
        messages=sample_context_with_summary["messages"],
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=1000,
    )


def test_generate_response_agent_not_found(agent_manager, sample_context):
    """Test response generation for non-existent agent"""
    result = agent_manager.generate_response("NonExistent", sample_context)
    assert result == "Error: Agent NonExistent not found"


def test_generate_response_empty_context(agent_manager, mock_llm_client, empty_context):
    """Test response generation with empty context"""
    mock_llm_client.generate_response.return_value = "Test response"

    result = agent_manager.generate_response("Sentinel", empty_context)

    mock_llm_client.generate_response.assert_called_once_with(
        system_prompt="You are a security expert", messages=[], model="gpt-4o-mini", temperature=0.5, max_tokens=1000
    )


def test_load_agent_configs_success(temp_config_dir, sample_agent_configs):
    """Test successful loading of agent configurations"""
    result = AgentManager._load_agent_configs(temp_config_dir)

    assert len(result) == len(sample_agent_configs)
    for agent_name in sample_agent_configs:
        assert agent_name in result
        assert result[agent_name] == sample_agent_configs[agent_name]


def test_load_agent_configs_multiple_files():
    """Test loading multiple agent configuration files"""
    config1 = {"name": "Agent1", "system_prompt": "Prompt 1"}
    config2 = {"name": "Agent2", "system_prompt": "Prompt 2"}

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "agent1.yaml"), "w") as f:
            yaml.dump(config1, f)
        with open(os.path.join(temp_dir, "agent2.yaml"), "w") as f:
            yaml.dump(config2, f)
        with open(os.path.join(temp_dir, "readme.txt"), "w") as f:
            f.write("This should be ignored")

        result = AgentManager._load_agent_configs(temp_dir)

        assert len(result) == 2
        assert "Agent1" in result
        assert "Agent2" in result
        assert result["Agent1"] == config1
        assert result["Agent2"] == config2


def test_load_agent_configs_empty_directory():
    """Test loading from empty directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = AgentManager._load_agent_configs(temp_dir)
        assert result == {}


def test_load_selector_config_success(sample_selector_config):
    """Test successful loading of selector configuration"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "agent_selector.yaml")
        with open(config_path, "w") as f:
            yaml.dump(sample_selector_config, f)

        result = AgentManager._load_selector_config(config_path)
        assert result == sample_selector_config


@pytest.mark.parametrize(
    "query,expected_agent,mock_response",
    [
        ("@Sentinel help with security", "Sentinel", None),  # Direct call, no LLM needed
        ("@FinGuide budget advice", "FinGuide", None),  # Direct call, no LLM needed
        ("@EduBot teach me", "EduBot", None),  # Direct call, no LLM needed
        ("security password hack", "Sentinel", '{"agent_name": "Sentinel"}'),
        ("money budget finance", "FinGuide", '{"agent_name": "FinGuide"}'),
        ("learn study education", "EduBot", '{"agent_name": "EduBot"}'),
    ],
)
def test_select_agent_parametrized(agent_manager, mock_llm_client, query, expected_agent, mock_response):
    """Parametrized test for agent selection"""
    if mock_response:
        mock_llm_client.generate_response.return_value = mock_response

    result = agent_manager.select_agent(query)
    assert result == expected_agent
