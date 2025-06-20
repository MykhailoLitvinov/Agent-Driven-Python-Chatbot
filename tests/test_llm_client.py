import os
import sys
from io import StringIO
from unittest.mock import Mock, patch

import openai
import pytest

from src.llm_client import LLMClient


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    with patch("src.llm_client.openai.OpenAI") as mock_openai:
        mock_client_instance = Mock()
        mock_openai.return_value = mock_client_instance
        yield mock_client_instance


@pytest.fixture
def llm_client(mock_openai_client):
    """Create LLMClient instance with mocked OpenAI client"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
        return LLMClient()


@pytest.fixture
def sample_messages():
    """Sample message list for testing"""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]


@pytest.fixture
def mock_streaming_response():
    """Mock streaming response from OpenAI"""
    chunk1 = Mock()
    chunk1.choices = [Mock()]
    chunk1.choices[0].delta = Mock()
    chunk1.choices[0].delta.content = "Hello"

    chunk2 = Mock()
    chunk2.choices = [Mock()]
    chunk2.choices[0].delta = Mock()
    chunk2.choices[0].delta.content = " there!"

    chunk3 = Mock()
    chunk3.choices = [Mock()]
    chunk3.choices[0].delta = Mock()
    chunk3.choices[0].delta.content = None

    return [chunk1, chunk2, chunk3]


@pytest.fixture
def mock_non_streaming_response():
    """Mock non-streaming response from OpenAI"""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Hello there!"
    return response


def test_llm_client_init(mock_openai_client):
    """Test LLMClient initialization"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
        with patch("src.llm_client.openai.OpenAI") as mock_openai:
            mock_client_instance = Mock()
            mock_openai.return_value = mock_client_instance

            client = LLMClient()

            assert client.client == mock_client_instance
            mock_openai.assert_called_once_with(api_key="test-api-key")


def test_llm_client_init_no_api_key(mock_openai_client):
    """Test LLMClient initialization without API key"""
    with patch.dict(os.environ, {}, clear=True):
        with patch("src.llm_client.openai.OpenAI") as mock_openai:
            mock_client_instance = Mock()
            mock_openai.return_value = mock_client_instance

            client = LLMClient()

            mock_openai.assert_called_once_with(api_key=None)


def test_generate_response_streaming_success(llm_client, mock_openai_client, sample_messages, mock_streaming_response):
    """Test successful streaming response generation"""
    system_prompt = "You are a helpful assistant"
    model = "gpt-4o-mini"
    temperature = 0.5
    max_tokens = 1000

    mock_openai_client.chat.completions.create.return_value = mock_streaming_response

    # Capture stdout to verify streaming output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        result = llm_client.generate_response(
            system_prompt=system_prompt,
            messages=sample_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        output = captured_output.getvalue()

        assert result == "Hello there!"
        assert "🤖 Hello" in output
        assert " there!" in output

        # Verify API call
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            response_format=None,
        )
    finally:
        sys.stdout = old_stdout


def test_generate_response_non_streaming_success(
    llm_client, mock_openai_client, sample_messages, mock_non_streaming_response
):
    """Test successful non-streaming response generation"""
    system_prompt = "You are a helpful assistant"
    model = "gpt-4o-mini"
    temperature = 0.5
    max_tokens = 1000

    mock_openai_client.chat.completions.create.return_value = mock_non_streaming_response

    result = llm_client.generate_response(
        system_prompt=system_prompt,
        messages=sample_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    assert result == "Hello there!"

    # Verify API call
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        max_completion_tokens=max_tokens,
        temperature=temperature,
        stream=False,
        response_format=None,
    )


def test_generate_response_empty_messages(llm_client, mock_openai_client, mock_streaming_response):
    """Test response generation with empty messages list"""
    system_prompt = "You are a helpful assistant"
    model = "gpt-4o-mini"
    temperature = 0.5
    max_tokens = 1000

    mock_openai_client.chat.completions.create.return_value = mock_streaming_response

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        result = llm_client.generate_response(
            system_prompt=system_prompt,
            messages=[],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        assert result == "Hello there!"

        # Verify API call with only system message
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model=model,
            messages=[{"role": "system", "content": system_prompt}],
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            response_format=None,
        )
    finally:
        sys.stdout = old_stdout


def test_generate_response_openai_error(llm_client, mock_openai_client, sample_messages):
    """Test response generation with OpenAI error"""
    system_prompt = "You are a helpful assistant"
    model = "gpt-4o-mini"
    temperature = 0.5
    max_tokens = 1000

    error_message = "API quota exceeded"
    mock_openai_client.chat.completions.create.side_effect = openai.OpenAIError(error_message)

    result = llm_client.generate_response(
        system_prompt=system_prompt,
        messages=sample_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    assert f"I apologize, but I'm having trouble generating a response right now. Error: {error_message}" in result


def test_generate_response_generic_exception(llm_client, mock_openai_client, sample_messages):
    """Test response generation with generic exception"""
    system_prompt = "You are a helpful assistant"
    model = "gpt-4o-mini"
    temperature = 0.5
    max_tokens = 1000

    error_message = "Connection timeout"
    mock_openai_client.chat.completions.create.side_effect = Exception(error_message)

    result = llm_client.generate_response(
        system_prompt=system_prompt,
        messages=sample_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    assert f"Internal error: {error_message}" in result


def test_generate_response_streaming_no_content(llm_client, mock_openai_client, sample_messages):
    """Test streaming response with chunks that have no content"""
    system_prompt = "You are a helpful assistant"
    model = "gpt-4o-mini"
    temperature = 0.5
    max_tokens = 1000

    # Create chunks with no content
    chunk1 = Mock()
    chunk1.choices = [Mock()]
    chunk1.choices[0].delta = Mock()
    chunk1.choices[0].delta.content = None

    chunk2 = Mock()
    chunk2.choices = [Mock()]
    chunk2.choices[0].delta = Mock()
    chunk2.choices[0].delta.content = ""

    mock_streaming_response = [chunk1, chunk2]
    mock_openai_client.chat.completions.create.return_value = mock_streaming_response

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        result = llm_client.generate_response(
            system_prompt=system_prompt,
            messages=sample_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        output = captured_output.getvalue()

        assert result == ""
        assert "🤖" not in output  # No robot emoji should be printed
    finally:
        sys.stdout = old_stdout


def test_generate_response_streaming_single_token(llm_client, mock_openai_client, sample_messages):
    """Test streaming response with single token"""
    system_prompt = "You are a helpful assistant"
    model = "gpt-4o-mini"
    temperature = 0.5
    max_tokens = 1000

    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta = Mock()
    chunk.choices[0].delta.content = "Hello"

    mock_streaming_response = [chunk]
    mock_openai_client.chat.completions.create.return_value = mock_streaming_response

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        result = llm_client.generate_response(
            system_prompt=system_prompt,
            messages=sample_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        output = captured_output.getvalue()

        assert result == "Hello"
        assert "🤖 Hello" in output
    finally:
        sys.stdout = old_stdout


def test_generate_response_non_streaming_with_whitespace(llm_client, mock_openai_client, sample_messages):
    """Test non-streaming response strips whitespace"""
    system_prompt = "You are a helpful assistant"
    model = "gpt-4o-mini"
    temperature = 0.5
    max_tokens = 1000

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "  Hello there!  \n"

    mock_openai_client.chat.completions.create.return_value = response

    result = llm_client.generate_response(
        system_prompt=system_prompt,
        messages=sample_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    assert result == "Hello there!"


@pytest.mark.parametrize(
    "temperature,max_tokens,model",
    [
        (0.0, 100, "gpt-3.5-turbo"),
        (1.0, 2000, "gpt-4"),
        (0.7, 1500, "gpt-4o-mini"),
    ],
)
def test_generate_response_different_parameters(
    llm_client, mock_openai_client, sample_messages, mock_streaming_response, temperature, max_tokens, model
):
    """Test response generation with different parameter values"""
    system_prompt = "You are a helpful assistant"

    mock_openai_client.chat.completions.create.return_value = mock_streaming_response

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        llm_client.generate_response(
            system_prompt=system_prompt,
            messages=sample_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        mock_openai_client.chat.completions.create.assert_called_once_with(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            response_format=None,
        )
    finally:
        sys.stdout = old_stdout


def test_generate_response_message_formatting(llm_client, mock_openai_client, mock_streaming_response):
    """Test that messages are properly formatted for API call"""
    system_prompt = "Custom system prompt"
    messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "Assistant response"},
        {"role": "user", "content": "Second message"},
    ]

    mock_openai_client.chat.completions.create.return_value = mock_streaming_response

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        llm_client.generate_response(
            system_prompt=system_prompt,
            messages=messages,
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=1000,
            stream=True,
        )

        expected_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "user", "content": "Second message"},
        ]

        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["messages"] == expected_messages
    finally:
        sys.stdout = old_stdout


def test_generate_response_with_json_schema_format(llm_client, mock_openai_client, sample_messages):
    """Test response generation with JSON schema response format"""
    system_prompt = "You are a helpful assistant"
    model = "gpt-4o-mini"
    temperature = 0.5
    max_tokens = 1000

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "test_response",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = '{"answer": "Hello there!"}'

    mock_openai_client.chat.completions.create.return_value = response

    result = llm_client.generate_response(
        system_prompt=system_prompt,
        messages=sample_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        response_format=response_format,
    )

    assert result == '{"answer": "Hello there!"}'

    # Verify API call includes response_format
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        max_completion_tokens=max_tokens,
        temperature=temperature,
        stream=False,
        response_format=response_format,
    )
