from unittest.mock import Mock

import pytest

from src.memory import ConversationMemory
from src.summarizer import Summarizer


@pytest.fixture
def mock_summarizer():
    """Create a mock summarizer for testing"""
    return Mock(spec=Summarizer)


@pytest.fixture
def conversation_memory(mock_summarizer):
    """Create ConversationMemory instance with mocked summarizer"""
    return ConversationMemory(summarizer=mock_summarizer, max_raw_messages=10)


@pytest.fixture
def conversation_memory_small(mock_summarizer):
    """Create ConversationMemory instance with small message limit"""
    return ConversationMemory(summarizer=mock_summarizer, max_raw_messages=4)


def test_conversation_memory_init(mock_summarizer):
    """Test ConversationMemory initialization"""
    memory = ConversationMemory(summarizer=mock_summarizer, max_raw_messages=5)

    assert memory.max_raw_messages == 5
    assert memory.messages == []
    assert memory.summary == ""
    assert memory.summarizer == mock_summarizer


def test_conversation_memory_init_default_max_messages(mock_summarizer):
    """Test ConversationMemory initialization with default max_raw_messages"""
    memory = ConversationMemory(summarizer=mock_summarizer)

    assert memory.max_raw_messages == 10
    assert memory.messages == []
    assert memory.summary == ""
    assert memory.summarizer == mock_summarizer


def test_add_message_single(conversation_memory):
    """Test adding a single message"""
    conversation_memory.add_message("user", "Hello")

    assert len(conversation_memory.messages) == 1
    assert conversation_memory.messages[0] == {"role": "user", "content": "Hello"}


def test_add_message_multiple(conversation_memory):
    """Test adding multiple messages"""
    conversation_memory.add_message("user", "Hello")
    conversation_memory.add_message("assistant", "Hi there!")
    conversation_memory.add_message("user", "How are you?")

    assert len(conversation_memory.messages) == 3
    assert conversation_memory.messages[0] == {"role": "user", "content": "Hello"}
    assert conversation_memory.messages[1] == {"role": "assistant", "content": "Hi there!"}
    assert conversation_memory.messages[2] == {"role": "user", "content": "How are you?"}


def test_add_message_under_limit(conversation_memory, mock_summarizer):
    """Test adding messages under the limit doesn't trigger summarization"""
    for i in range(5):
        conversation_memory.add_message("user", f"Message {i}")

    assert len(conversation_memory.messages) == 5
    assert conversation_memory.summary == ""
    mock_summarizer.update_summary.assert_not_called()


def test_add_message_at_limit(conversation_memory, mock_summarizer):
    """Test adding messages at the limit doesn't trigger summarization"""
    for i in range(10):
        conversation_memory.add_message("user", f"Message {i}")

    assert len(conversation_memory.messages) == 10
    assert conversation_memory.summary == ""
    mock_summarizer.update_summary.assert_not_called()


def test_add_message_over_limit_triggers_summarization(conversation_memory_small, mock_summarizer):
    """Test adding messages over the limit triggers summarization"""
    mock_summarizer.update_summary.return_value = "Summary of conversation"

    # Add messages up to and beyond the limit (4)
    for i in range(5):
        conversation_memory_small.add_message("user", f"Message {i}")

    # Should have triggered summarization
    mock_summarizer.update_summary.assert_called_once()
    assert conversation_memory_small.summary == "Summary of conversation"

    # Should keep only the most recent messages (max_raw_messages // 2 = 2)
    assert len(conversation_memory_small.messages) == 2
    assert conversation_memory_small.messages[0] == {"role": "user", "content": "Message 3"}
    assert conversation_memory_small.messages[1] == {"role": "user", "content": "Message 4"}


def test_add_message_summarization_parameters(conversation_memory_small, mock_summarizer):
    """Test that summarization is called with correct parameters"""
    mock_summarizer.update_summary.return_value = "Updated summary"

    # Add initial summary
    conversation_memory_small.summary = "Existing summary"

    # Add messages to trigger summarization
    for i in range(5):
        conversation_memory_small.add_message("user", f"Message {i}")

    # Check that update_summary was called with existing summary and messages to summarize
    expected_messages_to_summarize = [
        {"role": "user", "content": "Message 0"},
        {"role": "user", "content": "Message 1"},
        {"role": "user", "content": "Message 2"},
    ]

    mock_summarizer.update_summary.assert_called_once_with("Existing summary", expected_messages_to_summarize)


def test_add_message_multiple_summarizations(conversation_memory_small, mock_summarizer):
    """Test multiple rounds of summarization"""
    mock_summarizer.update_summary.side_effect = ["First summary", "Second summary"]

    # First round - trigger first summarization
    for i in range(5):
        conversation_memory_small.add_message("user", f"Message {i}")

    assert conversation_memory_small.summary == "First summary"
    assert len(conversation_memory_small.messages) == 2

    # Second round - add more messages to trigger second summarization
    for i in range(5, 8):
        conversation_memory_small.add_message("user", f"Message {i}")

    assert conversation_memory_small.summary == "Second summary"
    assert len(conversation_memory_small.messages) == 2
    assert mock_summarizer.update_summary.call_count == 2


def test_get_context_empty(conversation_memory):
    """Test get_context with empty memory"""
    context = conversation_memory.get_context()

    assert context == {"messages": [], "summary": ""}


def test_get_context_with_messages(conversation_memory):
    """Test get_context with messages"""
    conversation_memory.add_message("user", "Hello")
    conversation_memory.add_message("assistant", "Hi there!")

    context = conversation_memory.get_context()

    expected_messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]

    assert context == {"messages": expected_messages, "summary": ""}


def test_get_context_with_summary(conversation_memory):
    """Test get_context with summary"""
    conversation_memory.summary = "Previous conversation summary"
    conversation_memory.add_message("user", "New message")

    context = conversation_memory.get_context()

    assert context == {
        "messages": [{"role": "user", "content": "New message"}],
        "summary": "Previous conversation summary",
    }


def test_get_context_limits_messages(conversation_memory):
    """Test that get_context limits messages to max_raw_messages"""
    # Add more messages than the limit
    for i in range(16):
        conversation_memory.add_message("user", f"Message {i}")

    context = conversation_memory.get_context()

    # Should only return the last max_raw_messages (10)
    assert len(context["messages"]) == 10
    assert context["messages"][0] == {"role": "user", "content": "Message 6"}
    assert context["messages"][-1] == {"role": "user", "content": "Message 15"}


def test_get_context_after_summarization(conversation_memory_small, mock_summarizer):
    """Test get_context after summarization has occurred"""
    mock_summarizer.update_summary.return_value = "Conversation summary"

    # Add messages to trigger summarization
    for i in range(5):
        conversation_memory_small.add_message("user", f"Message {i}")

    context = conversation_memory_small.get_context()

    assert context["summary"] == "Conversation summary"
    assert len(context["messages"]) == 2
    assert context["messages"] == [{"role": "user", "content": "Message 3"}, {"role": "user", "content": "Message 4"}]


def test_reset(conversation_memory, mock_summarizer):
    """Test resetting the memory"""
    # Add some messages and summary
    conversation_memory.add_message("user", "Hello")
    conversation_memory.add_message("assistant", "Hi there!")
    conversation_memory.summary = "Some summary"

    # Reset
    conversation_memory.reset()

    assert conversation_memory.messages == []
    assert conversation_memory.summary == ""


def test_reset_after_summarization(conversation_memory_small, mock_summarizer):
    """Test resetting after summarization has occurred"""
    mock_summarizer.update_summary.return_value = "Summary"

    # Add messages to trigger summarization
    for i in range(5):
        conversation_memory_small.add_message("user", f"Message {i}")

    # Verify summarization occurred
    assert conversation_memory_small.summary == "Summary"
    assert len(conversation_memory_small.messages) == 2

    # Reset
    conversation_memory_small.reset()

    assert conversation_memory_small.messages == []
    assert conversation_memory_small.summary == ""


@pytest.mark.parametrize(
    "max_messages,num_messages_to_add,expected_messages_kept",
    [
        (4, 5, 2),  # max_messages=4, add 5, keep 4//2=2
        (6, 7, 3),  # max_messages=6, add 7, keep 6//2=3
        (8, 9, 4),  # max_messages=8, add 9, keep 8//2=4
        (10, 11, 5),  # max_messages=10, add 11, keep 10//2=5
    ],
)
def test_add_message_summarization_message_counts(
    mock_summarizer, max_messages, num_messages_to_add, expected_messages_kept
):
    """Test that correct number of messages are kept after summarization"""
    mock_summarizer.update_summary.return_value = "Summary"
    memory = ConversationMemory(summarizer=mock_summarizer, max_raw_messages=max_messages)

    for i in range(num_messages_to_add):
        memory.add_message("user", f"Message {i}")

    assert len(memory.messages) == expected_messages_kept
    mock_summarizer.update_summary.assert_called_once()


def test_add_message_different_roles(conversation_memory):
    """Test adding messages with different roles"""
    conversation_memory.add_message("user", "User message")
    conversation_memory.add_message("assistant", "Assistant message")
    conversation_memory.add_message("system", "System message")

    assert len(conversation_memory.messages) == 3
    assert conversation_memory.messages[0]["role"] == "user"
    assert conversation_memory.messages[1]["role"] == "assistant"
    assert conversation_memory.messages[2]["role"] == "system"


def test_add_message_empty_content(conversation_memory):
    """Test adding messages with empty content"""
    conversation_memory.add_message("user", "")
    conversation_memory.add_message("assistant", "   ")

    assert len(conversation_memory.messages) == 2
    assert conversation_memory.messages[0]["content"] == ""
    assert conversation_memory.messages[1]["content"] == "   "


def test_conversation_memory_edge_case_max_messages_two(mock_summarizer):
    """Test edge case with max_raw_messages=2"""
    mock_summarizer.update_summary.return_value = "Summary"
    memory = ConversationMemory(summarizer=mock_summarizer, max_raw_messages=2)

    memory.add_message("user", "First message")
    memory.add_message("user", "Second message")
    assert len(memory.messages) == 2

    memory.add_message("user", "Third message")
    # Should trigger summarization, keeping 2//2 = 1 message
    assert len(memory.messages) == 1
    assert memory.messages[0]["content"] == "Third message"
    mock_summarizer.update_summary.assert_called_once()


def test_summarization_messages_calculation(conversation_memory_small, mock_summarizer):
    """Test that the correct messages are passed to summarization"""
    mock_summarizer.update_summary.return_value = "Summary"

    # Add exactly max_raw_messages + 1 to trigger summarization
    messages = []
    for i in range(5):  # max_raw_messages = 4, so this triggers summarization
        message_content = f"Message {i}"
        messages.append({"role": "user", "content": message_content})
        conversation_memory_small.add_message("user", message_content)

    # Should summarize first 3 messages (messages_to_summarize = messages[:-4//2] = messages[:-2])
    expected_summarized = messages[:-2]  # First 3 messages

    mock_summarizer.update_summary.assert_called_once_with("", expected_summarized)
