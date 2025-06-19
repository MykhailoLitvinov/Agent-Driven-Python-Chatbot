import sys
from datetime import datetime
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from src.agents import AgentName
from src.chatbot import Chatbot


@pytest.fixture
def mock_dependencies():
    """Mock all dependencies for Chatbot"""
    with patch("src.chatbot.LLMClient") as mock_llm_client, patch(
        "src.chatbot.ConversationMemory"
    ) as mock_memory, patch("src.chatbot.Summarizer") as mock_summarizer, patch(
        "src.chatbot.AgentManager"
    ) as mock_agent_manager, patch(
        "src.chatbot.setup_logging"
    ) as mock_logger:

        # Configure mocks
        mock_llm_instance = Mock()
        mock_llm_client.return_value = mock_llm_instance

        mock_memory_instance = Mock()
        mock_memory.return_value = mock_memory_instance

        mock_summarizer_instance = Mock()
        mock_summarizer.return_value = mock_summarizer_instance

        mock_agent_manager_instance = Mock()
        mock_agent_manager.return_value = mock_agent_manager_instance

        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        yield {
            "llm_client": mock_llm_client,
            "memory": mock_memory,
            "summarizer": mock_summarizer,
            "agent_manager": mock_agent_manager,
            "logger": mock_logger,
            "llm_instance": mock_llm_instance,
            "memory_instance": mock_memory_instance,
            "summarizer_instance": mock_summarizer_instance,
            "agent_manager_instance": mock_agent_manager_instance,
            "logger_instance": mock_logger_instance,
        }


@pytest.fixture
def chatbot(mock_dependencies):
    """Create a Chatbot instance with mocked dependencies"""
    return Chatbot()


def test_chatbot_init(mock_dependencies):
    """Test Chatbot initialization"""
    chatbot = Chatbot()

    assert chatbot.llm_client == mock_dependencies["llm_instance"]
    assert chatbot.memory == mock_dependencies["memory_instance"]
    assert chatbot.default_agent == AgentName.EDUBOT.value
    assert chatbot.current_agent == AgentName.EDUBOT.value
    assert chatbot.agent_manager == mock_dependencies["agent_manager_instance"]
    assert chatbot.logger == mock_dependencies["logger_instance"]

    # Verify initialization calls
    mock_dependencies["llm_client"].assert_called_once()
    mock_dependencies["memory"].assert_called_once_with(summarizer=mock_dependencies["summarizer_instance"])
    mock_dependencies["summarizer"].assert_called_once_with(mock_dependencies["llm_instance"])
    mock_dependencies["agent_manager"].assert_called_once_with(
        mock_dependencies["llm_instance"], AgentName.EDUBOT.value
    )
    mock_dependencies["logger"].assert_called_once()


def test_process_query_basic(chatbot, mock_dependencies):
    """Test basic query processing"""
    query = "Hello, how are you?"
    expected_response = "I'm doing well, thank you!"

    # Setup mocks
    mock_dependencies["agent_manager_instance"].select_agent.return_value = "EduBot"
    mock_dependencies["memory_instance"].get_context.return_value = {"messages": []}
    mock_dependencies["agent_manager_instance"].generate_response.return_value = expected_response

    result = chatbot.process_query(query)

    assert result == expected_response

    # Verify method calls
    mock_dependencies["agent_manager_instance"].select_agent.assert_called_once_with(query)
    mock_dependencies["memory_instance"].add_message.assert_any_call("user", query)
    mock_dependencies["memory_instance"].add_message.assert_any_call("assistant", expected_response)
    mock_dependencies["memory_instance"].get_context.assert_called_once()
    mock_dependencies["agent_manager_instance"].generate_response.assert_called_once_with("EduBot", {"messages": []})


def test_process_query_agent_switch(chatbot, mock_dependencies):
    """Test query processing with agent switch"""
    query = "@Sentinel help with security"
    expected_response = "I'll help you with security"

    # Setup mocks
    mock_dependencies["agent_manager_instance"].select_agent.return_value = "Sentinel"
    mock_dependencies["memory_instance"].get_context.return_value = {"messages": []}
    mock_dependencies["agent_manager_instance"].generate_response.return_value = expected_response

    # Capture stdout to check agent switch message
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        result = chatbot.process_query(query)
        output = captured_output.getvalue()

        assert result == expected_response
        assert "🔄 Switching to Sentinel" in output
        assert chatbot.current_agent == "Sentinel"
    finally:
        sys.stdout = old_stdout


def test_process_query_no_agent_switch(chatbot, mock_dependencies):
    """Test query processing without agent switch"""
    query = "Tell me about learning"
    expected_response = "Learning is important"

    # Setup mocks - same agent as current
    mock_dependencies["agent_manager_instance"].select_agent.return_value = "EduBot"
    mock_dependencies["memory_instance"].get_context.return_value = {"messages": []}
    mock_dependencies["agent_manager_instance"].generate_response.return_value = expected_response

    # Capture stdout to check no agent switch message
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        result = chatbot.process_query(query)
        output = captured_output.getvalue()

        assert result == expected_response
        assert "🔄 Switching to" not in output
        assert chatbot.current_agent == "EduBot"
    finally:
        sys.stdout = old_stdout


def test_process_query_logging(chatbot, mock_dependencies):
    """Test that query processing logs appropriately"""
    query = "Test query"
    expected_response = "Test response"

    # Setup mocks
    mock_dependencies["agent_manager_instance"].select_agent.return_value = "EduBot"
    mock_dependencies["memory_instance"].get_context.return_value = {"messages": []}
    mock_dependencies["agent_manager_instance"].generate_response.return_value = expected_response

    with patch("src.chatbot.datetime") as mock_datetime:
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 12, 0, 1)
        mock_datetime.now.side_effect = [start_time, end_time]

        chatbot.process_query(query)

        # Verify logging calls
        mock_dependencies["logger_instance"].info.assert_any_call(f"Retrieved user message: {query}")
        mock_dependencies["logger_instance"].info.assert_any_call(
            f"Retrieved response from EduBot: {expected_response}"
        )
        mock_dependencies["logger_instance"].info.assert_any_call("Query processed by EduBot in 1.00s")


def test_reset_conversation(chatbot, mock_dependencies):
    """Test conversation reset functionality"""
    # Change current agent first
    chatbot.current_agent = "Sentinel"

    # Capture stdout to check reset message
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        chatbot.reset_conversation()
        output = captured_output.getvalue()

        assert "🔄 Conversation reset!" in output
        assert chatbot.current_agent == chatbot.default_agent

        # Verify method calls
        mock_dependencies["memory_instance"].reset.assert_called_once()
        mock_dependencies["logger_instance"].info.assert_called_with("Conversation reset")
    finally:
        sys.stdout = old_stdout


def test_start_quit_command(chatbot):
    """Test start method with quit command"""
    with patch("builtins.input", side_effect=["quit"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            chatbot.start()
            output = captured_output.getvalue()
            assert "👋 Goodbye!" in output
        finally:
            sys.stdout = old_stdout


def test_start_exit_command(chatbot):
    """Test start method with exit command"""
    with patch("builtins.input", side_effect=["exit"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            chatbot.start()
            output = captured_output.getvalue()
            assert "👋 Goodbye!" in output
        finally:
            sys.stdout = old_stdout


def test_start_q_command(chatbot):
    """Test start method with q command"""
    with patch("builtins.input", side_effect=["q"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            chatbot.start()
            output = captured_output.getvalue()
            assert "👋 Goodbye!" in output
        finally:
            sys.stdout = old_stdout


def test_start_reset_command(chatbot, mock_dependencies):
    """Test start method with reset command"""
    with patch("builtins.input", side_effect=["reset", "quit"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            chatbot.start()
            output = captured_output.getvalue()
            assert "🔄 Conversation reset!" in output
            assert "👋 Goodbye!" in output
        finally:
            sys.stdout = old_stdout


def test_start_empty_input(chatbot):
    """Test start method with empty input"""
    with patch("builtins.input", side_effect=["", "  ", "quit"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            chatbot.start()
            output = captured_output.getvalue()
            assert "👋 Goodbye!" in output
        finally:
            sys.stdout = old_stdout


def test_start_keyboard_interrupt(chatbot):
    """Test start method with keyboard interrupt"""
    with patch("builtins.input", side_effect=KeyboardInterrupt()):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            chatbot.start()
            output = captured_output.getvalue()
            assert "👋 Goodbye!" in output
        finally:
            sys.stdout = old_stdout


def test_start_exception_handling(chatbot, mock_dependencies):
    """Test start method exception handling"""
    with patch("builtins.input", side_effect=["test query", "quit"]):
        with patch.object(chatbot, "process_query", side_effect=Exception("Test error")):
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            try:
                chatbot.start()
                output = captured_output.getvalue()
                assert "❌ Error: Test error" in output
                assert "👋 Goodbye!" in output

                # Verify error logging
                mock_dependencies["logger_instance"].error.assert_called_with("Error in main loop: Test error")
            finally:
                sys.stdout = old_stdout


def test_start_normal_query_processing(chatbot, mock_dependencies):
    """Test start method with normal query processing"""
    query = "Hello world"
    response = "Hello there!"

    # Setup mocks
    mock_dependencies["agent_manager_instance"].select_agent.return_value = "EduBot"
    mock_dependencies["memory_instance"].get_context.return_value = {"messages": []}
    mock_dependencies["agent_manager_instance"].generate_response.return_value = response

    with patch("builtins.input", side_effect=[query, "quit"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            chatbot.start()
            output = captured_output.getvalue()

            assert "👋 Goodbye!" in output
            # Verify process_query was called
            mock_dependencies["agent_manager_instance"].select_agent.assert_called_with(query)
        finally:
            sys.stdout = old_stdout


@pytest.mark.parametrize("command", ["quit", "exit", "q", "QUIT", "EXIT", "Q"])
def test_start_exit_commands_case_insensitive(chatbot, command):
    """Test that exit commands work case insensitively"""
    with patch("builtins.input", side_effect=[command]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            chatbot.start()
            output = captured_output.getvalue()
            assert "👋 Goodbye!" in output
        finally:
            sys.stdout = old_stdout


def test_start_reset_command_case_insensitive(chatbot):
    """Test that reset command works case insensitively"""
    with patch("builtins.input", side_effect=["RESET", "quit"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            chatbot.start()
            output = captured_output.getvalue()
            assert "🔄 Conversation reset!" in output
            assert "👋 Goodbye!" in output
        finally:
            sys.stdout = old_stdout
