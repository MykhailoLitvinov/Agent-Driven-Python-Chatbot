from unittest.mock import Mock

import pytest

from src.llm_client import LLMClient
from src.summarizer import Summarizer


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing"""
    return Mock(spec=LLMClient)


@pytest.fixture
def summarizer(mock_llm_client):
    """Create Summarizer instance with mocked LLM client"""
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


def test_summarizer_init(mock_llm_client):
    """Test Summarizer initialization"""
    summarizer = Summarizer(mock_llm_client)

    assert summarizer.llm_client == mock_llm_client
    assert summarizer.model == "gpt-4o-mini"
    assert summarizer.temperature == 0.3
    assert summarizer.max_tokens == 300
    assert summarizer.stream is False


def test_update_summary_success(summarizer, mock_llm_client, sample_messages):
    """Test successful summary update"""
    previous_summary = "Previous conversation about greetings"
    expected_response = "Updated summary of conversation"

    mock_llm_client.generate_response.return_value = expected_response

    result = summarizer.update_summary(previous_summary, sample_messages)

    assert result == expected_response

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
    expected_response = "New summary of conversation"

    mock_llm_client.generate_response.return_value = expected_response

    result = summarizer.update_summary("", sample_messages)

    assert result == expected_response

    # Verify system prompt handles empty summary
    call_args = mock_llm_client.generate_response.call_args
    system_prompt = call_args[1]["system_prompt"]
    assert "None" in system_prompt  # Empty summary should show as "None"


def test_update_summary_none_previous_summary(summarizer, mock_llm_client, sample_messages):
    """Test update summary with None previous summary"""
    expected_response = "New summary of conversation"

    mock_llm_client.generate_response.return_value = expected_response

    result = summarizer.update_summary(None, sample_messages)

    assert result == expected_response

    # Verify system prompt handles None summary
    call_args = mock_llm_client.generate_response.call_args
    system_prompt = call_args[1]["system_prompt"]
    assert "None" in system_prompt


def test_update_summary_empty_messages(summarizer, mock_llm_client):
    """Test update summary with empty messages list"""
    previous_summary = "Previous summary"
    expected_response = "Summary remains the same"

    mock_llm_client.generate_response.return_value = expected_response

    result = summarizer.update_summary(previous_summary, [])

    assert result == expected_response

    # Verify system prompt was built with empty messages
    call_args = mock_llm_client.generate_response.call_args
    system_prompt = call_args[1]["system_prompt"]
    assert "Previous summary" in system_prompt
    assert "Latest messages:" in system_prompt


def test_update_summary_single_message(summarizer, mock_llm_client):
    """Test update summary with single message"""
    previous_summary = "Previous summary"
    messages = [{"role": "user", "content": "Single message"}]
    expected_response = "Updated summary"

    mock_llm_client.generate_response.return_value = expected_response

    result = summarizer.update_summary(previous_summary, messages)

    assert result == expected_response

    # Verify system prompt includes the single message
    call_args = mock_llm_client.generate_response.call_args
    system_prompt = call_args[1]["system_prompt"]
    assert "user: Single message" in system_prompt


def test_build_prompt_with_summary_and_messages():
    """Test _build_prompt with summary and messages"""
    summary = "Previous conversation summary"
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]

    prompt = Summarizer._build_prompt(summary, messages)

    # Verify all components are present
    assert "You are expert in summarizing conversation" in prompt
    assert "Previous conversation summary" in prompt
    assert "user: Hello" in prompt
    assert "assistant: Hi there" in prompt
    assert "Provide an updated summary below:" in prompt


def test_build_prompt_empty_summary():
    """Test _build_prompt with empty summary"""
    messages = [{"role": "user", "content": "Hello"}]

    prompt = Summarizer._build_prompt("", messages)

    assert "None" in prompt  # Empty summary should show as "None"
    assert "user: Hello" in prompt


def test_build_prompt_none_summary():
    """Test _build_prompt with None summary"""
    messages = [{"role": "user", "content": "Hello"}]

    prompt = Summarizer._build_prompt(None, messages)

    assert "None" in prompt  # None summary should show as "None"
    assert "user: Hello" in prompt


def test_build_prompt_empty_messages():
    """Test _build_prompt with empty messages"""
    summary = "Previous summary"

    prompt = Summarizer._build_prompt(summary, [])

    assert "Previous summary" in prompt
    assert "Latest messages:" in prompt
    # Should have empty content after "Latest messages:"


def test_build_prompt_multiple_messages():
    """Test _build_prompt with multiple messages"""
    summary = "Summary"
    messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Second message"},
        {"role": "assistant", "content": "Second response"},
    ]

    prompt = Summarizer._build_prompt(summary, messages)

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


def test_build_prompt_special_characters_in_content():
    """Test _build_prompt with special characters in message content"""
    summary = "Previous summary"
    messages = [
        {"role": "user", "content": "Message with\nnewlines and\ttabs"},
        {"role": "assistant", "content": "Response with \"quotes\" and 'apostrophes'"},
    ]

    prompt = Summarizer._build_prompt(summary, messages)

    # Verify special characters are preserved
    assert "Message with\nnewlines and\ttabs" in prompt
    assert "Response with \"quotes\" and 'apostrophes'" in prompt


def test_build_prompt_whitespace_handling():
    """Test _build_prompt strips whitespace properly"""
    summary = "   Summary with spaces   "
    messages = [{"role": "user", "content": "   Message with spaces   "}]

    prompt = Summarizer._build_prompt(summary, messages)

    # The method should strip the final prompt but preserve content
    assert not prompt.startswith(" ")
    assert not prompt.endswith(" ")
    assert "   Summary with spaces   " in prompt  # Content spaces preserved
    assert "   Message with spaces   " in prompt  # Content spaces preserved


def test_build_prompt_different_roles():
    """Test _build_prompt with different message roles"""
    summary = "Summary"
    messages = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "system", "content": "System message"},
    ]

    prompt = Summarizer._build_prompt(summary, messages)

    assert "user: User message" in prompt
    assert "assistant: Assistant message" in prompt
    assert "system: System message" in prompt


def test_update_summary_llm_client_error_propagation(summarizer, mock_llm_client, sample_messages):
    """Test that errors from LLM client are propagated"""
    mock_llm_client.generate_response.side_effect = Exception("LLM error")

    with pytest.raises(Exception, match="LLM error"):
        summarizer.update_summary("Previous summary", sample_messages)


def test_summarizer_configuration_constants():
    """Test that summarizer uses expected configuration constants"""
    mock_llm_client = Mock(spec=LLMClient)
    summarizer = Summarizer(mock_llm_client)

    # Test configuration values
    assert summarizer.model == "gpt-4o-mini"
    assert summarizer.temperature == 0.3
    assert summarizer.max_tokens == 300
    assert summarizer.stream is False


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
def test_build_prompt_summary_variations(previous_summary, expected_in_prompt):
    """Test _build_prompt with various summary inputs"""
    messages = [{"role": "user", "content": "Test message"}]

    prompt = Summarizer._build_prompt(previous_summary, messages)

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
def test_build_prompt_message_variations(messages, expected_count):
    """Test _build_prompt with various message counts"""
    summary = "Test summary"

    prompt = Summarizer._build_prompt(summary, messages)

    # Count occurrences of role patterns
    role_count = prompt.count("user:") + prompt.count("assistant:") + prompt.count("system:")
    assert role_count == expected_count


def test_update_summary_return_value_passthrough(summarizer, mock_llm_client, sample_messages):
    """Test that update_summary returns exactly what LLM client returns"""
    llm_responses = [
        "Simple response",
        "",  # Empty response
        "Response with\nmultiple\nlines",
        "Response with special chars: !@#$%^&*()",
    ]

    for expected_response in llm_responses:
        mock_llm_client.generate_response.return_value = expected_response

        result = summarizer.update_summary("Previous", sample_messages)

        assert result == expected_response
