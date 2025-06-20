from unittest.mock import Mock, patch

import pytest

from src.llm_client import LLMClient
from src.summarizer import Summarizer


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing"""
    return Mock(spec=LLMClient)


@pytest.fixture
def sample_summarizer_config():
    """Sample summarizer configuration for testing"""
    return {
        "name": "Summarizer",
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 300,
        "stream": False,
        "system_prompt": "You are expert in summarizing conversation.\nYour goal is to provide compact and clear summary based on latest messages and previous summary.\n\nPrevious summary:\n{previous_summary}\n\nLatest messages:\n{latest_messages}\n\nProvide an updated summary below:",
    }


@pytest.fixture
def summarizer(mock_llm_client, sample_summarizer_config):
    """Create Summarizer instance with mocked LLM client"""
    with patch.object(Summarizer, "_load_config", return_value=sample_summarizer_config):
        return Summarizer(mock_llm_client)


@pytest.fixture
def sample_messages():
    """Sample messages for testing"""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "It's sunny and warm today."},
    ]


def test_summarizer_init(mock_llm_client, sample_summarizer_config):
    """Test Summarizer initialization"""
    with patch.object(Summarizer, "_load_config", return_value=sample_summarizer_config):
        summarizer = Summarizer(mock_llm_client)

        assert summarizer.llm_client == mock_llm_client
        assert summarizer.config == sample_summarizer_config


def test_update_summary_success(summarizer, mock_llm_client, sample_messages):
    """Test successful summary update"""
    previous_summary = "Previous conversation about greetings"
    expected_summary = "Updated summary of conversation"

    # Mock JSON response from LLM
    mock_llm_client.generate_response.return_value = f'{{"summary": "{expected_summary}"}}'

    result = summarizer.update_summary(previous_summary, sample_messages)

    assert result == expected_summary

    # Verify LLM client was called with correct parameters
    mock_llm_client.generate_response.assert_called_once()
    call_args = mock_llm_client.generate_response.call_args

    assert call_args[1]["messages"] == []
    assert call_args[1]["model"] == "gpt-4o-mini"
    assert call_args[1]["temperature"] == 0.3
    assert call_args[1]["max_tokens"] == 300
    assert call_args[1]["stream"] is False

    # Verify system prompt was built correctly
    system_prompt = call_args[1]["system_prompt"]
    assert "Previous conversation about greetings" in system_prompt
    assert "user: Hello, how are you?" in system_prompt
    assert "assistant: I'm doing well, thank you!" in system_prompt


def test_update_summary_empty_previous_summary(summarizer, mock_llm_client, sample_messages):
    """Test update summary with empty previous summary"""
    expected_summary = "New summary of conversation"

    mock_llm_client.generate_response.return_value = f'{{"summary": "{expected_summary}"}}'

    result = summarizer.update_summary("", sample_messages)

    assert result == expected_summary

    # Verify system prompt handles empty summary
    call_args = mock_llm_client.generate_response.call_args
    system_prompt = call_args[1]["system_prompt"]
    assert "None" in system_prompt  # Empty summary should show as "None"


def test_update_summary_none_previous_summary(summarizer, mock_llm_client, sample_messages):
    """Test update summary with None previous summary"""
    expected_summary = "New summary of conversation"

    mock_llm_client.generate_response.return_value = f'{{"summary": "{expected_summary}"}}'

    result = summarizer.update_summary(None, sample_messages)

    assert result == expected_summary

    # Verify system prompt handles None summary
    call_args = mock_llm_client.generate_response.call_args
    system_prompt = call_args[1]["system_prompt"]
    assert "None" in system_prompt


def test_update_summary_empty_messages(summarizer, mock_llm_client):
    """Test update summary with empty messages list"""
    previous_summary = "Previous summary"
    expected_summary = "Summary remains the same"

    mock_llm_client.generate_response.return_value = f'{{"summary": "{expected_summary}"}}'

    result = summarizer.update_summary(previous_summary, [])

    assert result == expected_summary

    # Verify system prompt was built with empty messages
    call_args = mock_llm_client.generate_response.call_args
    system_prompt = call_args[1]["system_prompt"]
    assert "Previous summary" in system_prompt
    assert "Latest messages:" in system_prompt


def test_update_summary_single_message(summarizer, mock_llm_client):
    """Test update summary with single message"""
    previous_summary = "Previous summary"
    messages = [{"role": "user", "content": "Single message"}]
    expected_summary = "Updated summary"

    mock_llm_client.generate_response.return_value = f'{{"summary": "{expected_summary}"}}'

    result = summarizer.update_summary(previous_summary, messages)

    assert result == expected_summary

    # Verify system prompt includes the single message
    call_args = mock_llm_client.generate_response.call_args
    system_prompt = call_args[1]["system_prompt"]
    assert "user: Single message" in system_prompt


def test_build_prompt_with_summary_and_messages(summarizer):
    """Test _build_prompt with summary and messages"""
    summary = "Previous conversation summary"
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]

    prompt = summarizer._build_prompt(summary, messages)

    # Verify all components are present
    assert "You are expert in summarizing conversation" in prompt
    assert "Previous conversation summary" in prompt
    assert "user: Hello" in prompt
    assert "assistant: Hi there" in prompt
    assert "Provide an updated summary below:" in prompt


def test_build_prompt_empty_summary(summarizer):
    """Test _build_prompt with empty summary"""
    messages = [{"role": "user", "content": "Hello"}]

    prompt = summarizer._build_prompt("", messages)

    assert "None" in prompt  # Empty summary should show as "None"
    assert "user: Hello" in prompt


def test_build_prompt_none_summary(summarizer):
    """Test _build_prompt with None summary"""
    messages = [{"role": "user", "content": "Hello"}]

    prompt = summarizer._build_prompt(None, messages)

    assert "None" in prompt  # None summary should show as "None"
    assert "user: Hello" in prompt


def test_build_prompt_empty_messages(summarizer):
    """Test _build_prompt with empty messages"""
    summary = "Previous summary"

    prompt = summarizer._build_prompt(summary, [])

    assert "Previous summary" in prompt
    assert "Latest messages:" in prompt
    # Should have empty content after "Latest messages:"


def test_build_prompt_multiple_messages(summarizer):
    """Test _build_prompt with multiple messages"""
    summary = "Summary"
    messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Second message"},
        {"role": "assistant", "content": "Second response"},
    ]

    prompt = summarizer._build_prompt(summary, messages)

    # Verify all messages are included in order
    assert "user: First message" in prompt
    assert "assistant: First response" in prompt
    assert "user: Second message" in prompt
    assert "assistant: Second response" in prompt

    # Verify messages appear in the expected order
    user1_pos = prompt.find("user: First message")
    asst1_pos = prompt.find("assistant: First response")
    user2_pos = prompt.find("user: Second message")
    asst2_pos = prompt.find("assistant: Second response")

    assert user1_pos < asst1_pos < user2_pos < asst2_pos


def test_build_prompt_special_characters_in_content(summarizer):
    """Test _build_prompt with special characters in message content"""
    summary = "Previous summary"
    messages = [
        {"role": "user", "content": "Message with\nnewlines and\ttabs"},
        {"role": "assistant", "content": "Response with \"quotes\" and 'apostrophes'"},
    ]

    prompt = summarizer._build_prompt(summary, messages)

    # Verify special characters are preserved
    assert "Message with\nnewlines and\ttabs" in prompt
    assert "Response with \"quotes\" and 'apostrophes'" in prompt


def test_build_prompt_whitespace_handling(summarizer):
    """Test _build_prompt strips whitespace properly"""
    summary = "   Summary with spaces   "
    messages = [{"role": "user", "content": "   Message with spaces   "}]

    prompt = summarizer._build_prompt(summary, messages)

    # The method should preserve content spaces
    assert "   Summary with spaces   " in prompt  # Content spaces preserved
    assert "   Message with spaces   " in prompt  # Content spaces preserved


def test_build_prompt_different_roles(summarizer):
    """Test _build_prompt with different message roles"""
    summary = "Summary"
    messages = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "system", "content": "System message"},
    ]

    prompt = summarizer._build_prompt(summary, messages)

    assert "user: User message" in prompt
    assert "assistant: Assistant message" in prompt
    assert "system: System message" in prompt


def test_update_summary_llm_client_error_propagation(summarizer, mock_llm_client, sample_messages):
    """Test that errors from LLM client are propagated"""
    mock_llm_client.generate_response.side_effect = Exception("LLM error")

    with pytest.raises(Exception, match="LLM error"):
        summarizer.update_summary("Previous summary", sample_messages)


def test_summarizer_configuration_constants(sample_summarizer_config):
    """Test that summarizer uses expected configuration constants"""
    mock_llm_client = Mock(spec=LLMClient)

    with patch.object(Summarizer, "_load_config", return_value=sample_summarizer_config):
        summarizer = Summarizer(mock_llm_client)

        # Test configuration values
        assert summarizer.config["model"] == "gpt-4o-mini"
        assert summarizer.config["temperature"] == 0.3
        assert summarizer.config["max_tokens"] == 300
        assert summarizer.config["stream"] is False


@pytest.mark.parametrize(
    "previous_summary,expected_in_prompt",
    [
        ("Short summary", "Short summary"),
        ("", "None"),
        (None, "None"),
        ("Summary with\nmultiple\nlines", "Summary with\nmultiple\nlines"),
        ("Summary with special chars: !@#$%", "Summary with special chars: !@#$%"),
    ],
)
def test_build_prompt_summary_variations(summarizer, previous_summary, expected_in_prompt):
    """Test _build_prompt with various summary inputs"""
    messages = [{"role": "user", "content": "Test message"}]

    prompt = summarizer._build_prompt(previous_summary, messages)

    assert expected_in_prompt in prompt
    assert "user: Test message" in prompt


@pytest.mark.parametrize(
    "messages,expected_count",
    [
        ([], 0),
        ([{"role": "user", "content": "One"}], 1),
        ([{"role": "user", "content": "One"}, {"role": "assistant", "content": "Two"}], 2),
        ([{"role": "user", "content": f"Message {i}"} for i in range(5)], 5),
    ],
)
def test_build_prompt_message_variations(summarizer, messages, expected_count):
    """Test _build_prompt with various message counts"""
    summary = "Test summary"

    prompt = summarizer._build_prompt(summary, messages)

    # Count occurrences of role patterns
    role_count = prompt.count("user:") + prompt.count("assistant:") + prompt.count("system:")
    assert role_count == expected_count


def test_update_summary_return_value_passthrough(summarizer, mock_llm_client, sample_messages):
    """Test that update_summary returns exactly what LLM client returns"""
    import json

    test_summaries = [
        "Simple response",
        "",  # Empty response
        "Response with multiple lines",
        "Response with special chars: !@#$%^&*()",
    ]

    for expected_summary in test_summaries:
        # Mock JSON response from LLM - use json.dumps to properly escape
        mock_llm_client.generate_response.return_value = json.dumps({"summary": expected_summary})

        result = summarizer.update_summary("Previous", sample_messages)

        assert result == expected_summary


def test_load_config_success(sample_summarizer_config):
    """Test successful loading of summarizer configuration"""
    import tempfile
    import yaml
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "summarizer.yaml")
        with open(config_path, "w") as f:
            yaml.dump(sample_summarizer_config, f)

        result = Summarizer._load_config(config_path)
        assert result == sample_summarizer_config
