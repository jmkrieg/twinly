"""
Unit tests for the base LLM provider and concrete implementations.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from app.models.chat import ChatCompletionRequest, Message
from app.services.llms.azure_openai_provider import AzureOpenAIProvider
from app.services.llms.base_llm_provider import BaseLLMProvider
from app.services.llms.claude_provider import ClaudeProvider
from app.services.llms.echo_provider import EchoProvider


class DummyProvider(BaseLLMProvider):
    """Dummy implementation for contract testing."""

    def is_available(self) -> bool:
        return True

    async def generate_response(self, req: ChatCompletionRequest):
        return {"dummy": "response"}

    async def generate_streaming_response(self, req: ChatCompletionRequest):
        yield "dummy chunk"


class BrokenProvider(BaseLLMProvider):
    """Broken implementation that violates the contract."""
    pass  # Intentionally doesn't implement abstract methods


class TestBaseLLMProvider:
    """Test the abstract base LLM provider contract."""

    def test_base_provider_cannot_be_instantiated_directly(self):
        """Test that BaseLLMProvider cannot be instantiated directly due to abstract methods."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseLLMProvider()

    def test_broken_provider_cannot_be_instantiated(self):
        """Test that providers must implement all abstract methods."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BrokenProvider()

    def test_dummy_provider_fulfills_contract(self):
        """Test that a proper implementation can be instantiated."""
        provider = DummyProvider()
        assert provider is not None
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_dummy_provider_implements_all_methods(self):
        """Test that dummy provider implements all required methods."""
        provider = DummyProvider()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Test message")]
        )

        # Test generate_response
        response = await provider.generate_response(request)
        assert response == {"dummy": "response"}

        # Test generate_streaming_response
        chunks = []
        async for chunk in provider.generate_streaming_response(request):
            chunks.append(chunk)
        assert chunks == ["dummy chunk"]


class TestEchoProvider:
    """Test the Echo provider implementation."""

    def test_echo_provider_is_always_available(self):
        """Test that Echo provider is always available."""
        provider = EchoProvider()
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_echo_provider_generate_response(self):
        """Test echo provider response generation."""
        provider = EchoProvider()
        request = ChatCompletionRequest(
            model="echo-model",
            messages=[Message(role="user", content="Hello, world!")]
        )

        response = await provider.generate_response(request)

        assert response["object"] == "chat.completion"
        assert response["id"] == "chatcmpl-echo"
        assert len(response["choices"]) == 1
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert "You said: Hello, world!" in response["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_echo_provider_generate_streaming_response(self):
        """Test echo provider streaming response generation."""
        provider = EchoProvider()
        request = ChatCompletionRequest(
            model="echo-model",
            messages=[Message(role="user", content="Test")]
        )

        chunks = []
        async for chunk in provider.generate_streaming_response(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1] == "data: [DONE]\n\n"

        # Check that we have role and content chunks
        role_chunk_found = False
        content_chunk_found = False
        all_content = ""

        for chunk in chunks[:-1]:  # Exclude [DONE] chunk
            if '"role": "assistant"' in chunk:
                role_chunk_found = True
            if '"content"' in chunk:
                # Extract content from JSON
                try:
                    import json
                    # Parse the chunk to extract content
                    chunk_data = chunk.replace("data: ", "").strip()
                    if chunk_data:
                        chunk_json = json.loads(chunk_data)
                        delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            all_content += delta["content"]
                except Exception:
                    pass

        # Check if we received the expected content
        if "You said:" in all_content and "Test" in all_content:
            content_chunk_found = True

        assert role_chunk_found
        assert content_chunk_found

    @pytest.mark.asyncio
    async def test_echo_provider_handles_no_user_messages(self):
        """Test echo provider handles case with no user messages."""
        provider = EchoProvider()
        request = ChatCompletionRequest(
            model="echo-model",
            messages=[Message(role="system", content="System message")]
        )

        response = await provider.generate_response(request)

        assert "No user message provided" in response["choices"][0]["message"]["content"]


class TestAzureOpenAIProvider:
    """Test the Azure OpenAI provider implementation."""

    @patch.dict('os.environ', {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
        'AZURE_OPENAI_API_KEY': 'test-key'
    })
    def test_azure_provider_initialization_with_env_vars(self):
        """Test Azure provider initialization with environment variables."""
        provider = AzureOpenAIProvider()
        assert provider.is_available() is True
        assert provider.endpoint == 'https://test.openai.azure.com'
        assert provider.api_key == 'test-key'

    @patch.dict('os.environ', {}, clear=True)
    def test_azure_provider_initialization_without_env_vars(self):
        """Test Azure provider initialization without environment variables."""
        provider = AzureOpenAIProvider()
        assert provider.is_available() is False
        assert provider.client is None

    @patch.dict('os.environ', {}, clear=True)
    @pytest.mark.asyncio
    async def test_azure_provider_raises_exception_when_not_configured(self):
        """Test that Azure provider raises exception when not configured."""
        provider = AzureOpenAIProvider()
        request = ChatCompletionRequest(
            model="gpt4.1-chat",
            messages=[Message(role="user", content="Test")]
        )

        with pytest.raises(HTTPException) as exc_info:
            await provider.generate_response(request)

        assert exc_info.value.status_code == 500
        assert "not configured" in str(exc_info.value.detail)

    @patch('app.services.llms.azure_openai_provider.AsyncAzureOpenAI')
    @patch.dict('os.environ', {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
        'AZURE_OPENAI_API_KEY': 'test-key'
    })
    @pytest.mark.asyncio
    async def test_azure_provider_generate_response_success(self, mock_client_class):
        """Test successful response generation with mocked Azure OpenAI."""
        # Mock the response
        mock_response = Mock()
        mock_response.id = "chatcmpl-test"
        mock_response.choices = [Mock()]
        mock_response.choices[0].index = 0
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"

        # Mock the client
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        provider = AzureOpenAIProvider()
        request = ChatCompletionRequest(
            model="gpt4.1-chat",
            messages=[Message(role="user", content="Test message")]
        )

        response = await provider.generate_response(request)

        assert response["id"] == "chatcmpl-test"
        assert response["object"] == "chat.completion"
        assert len(response["choices"]) == 1
        assert response["choices"][0]["message"]["content"] == "Test response"

        # Verify the SDK was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        # Note: deployment name comes from env var or defaults to "gpt-4"
        assert call_args[1]["model"] in ["gpt-4", "gpt-4.1"]  # Allow both values
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["content"] == "Test message"

    @patch('app.services.llms.azure_openai_provider.AsyncAzureOpenAI')
    @patch.dict('os.environ', {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
        'AZURE_OPENAI_API_KEY': 'test-key'
    })
    @pytest.mark.asyncio
    async def test_azure_provider_generate_streaming_response_success(self, mock_client_class):
        """Test successful streaming response generation with mocked Azure OpenAI."""
        # Mock streaming chunks
        mock_chunk1 = Mock()
        mock_chunk1.id = "chatcmpl-stream"
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].index = 0
        mock_chunk1.choices[0].delta = Mock()
        mock_chunk1.choices[0].delta.role = "assistant"
        mock_chunk1.choices[0].delta.content = None
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = Mock()
        mock_chunk2.id = "chatcmpl-stream"
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].index = 0
        mock_chunk2.choices[0].delta = Mock()
        mock_chunk2.choices[0].delta.role = None
        mock_chunk2.choices[0].delta.content = "Hello"
        mock_chunk2.choices[0].finish_reason = None

        mock_chunk3 = Mock()
        mock_chunk3.id = "chatcmpl-stream"
        mock_chunk3.choices = [Mock()]
        mock_chunk3.choices[0].index = 0
        mock_chunk3.choices[0].delta = Mock()
        mock_chunk3.choices[0].delta.role = None
        mock_chunk3.choices[0].delta.content = None
        mock_chunk3.choices[0].finish_reason = "stop"

        # Mock the async iterator
        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2
            yield mock_chunk3

        # Mock the client
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_client_class.return_value = mock_client

        provider = AzureOpenAIProvider()
        request = ChatCompletionRequest(
            model="gpt4.1-chat",
            messages=[Message(role="user", content="Test message")]
        )

        chunks = []
        async for chunk in provider.generate_streaming_response(request):
            chunks.append(chunk)

        assert len(chunks) >= 2  # At least some content chunks + [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"

        # Check for proper SSE format
        for chunk in chunks[:-1]:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")

    @patch('app.services.llms.azure_openai_provider.AsyncAzureOpenAI')
    @patch.dict('os.environ', {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
        'AZURE_OPENAI_API_KEY': 'test-key'
    })
    @pytest.mark.asyncio
    async def test_azure_provider_handles_api_error(self, mock_client_class):
        """Test that Azure provider handles API errors gracefully."""
        # Mock the client to raise an exception
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        mock_client_class.return_value = mock_client

        provider = AzureOpenAIProvider()
        request = ChatCompletionRequest(
            model="gpt4.1-chat",
            messages=[Message(role="user", content="Test message")]
        )

        with pytest.raises(HTTPException) as exc_info:
            await provider.generate_response(request)

        assert exc_info.value.status_code == 500
        assert "Azure OpenAI error" in str(exc_info.value.detail)


class TestClaudeProvider:
    """Test the Claude provider implementation."""

    @patch.dict('os.environ', {'CLAUDE_API_KEY': 'test-key'})
    def test_claude_provider_initialization_with_env_vars(self):
        """Test Claude provider initialization with environment variables."""
        with patch('anthropic.Anthropic'):
            provider = ClaudeProvider()
            assert provider.is_available() is True
            assert provider.api_key == 'test-key'

    @patch.dict('os.environ', {}, clear=True)
    def test_claude_provider_initialization_without_env_vars(self):
        """Test Claude provider initialization without environment variables."""
        provider = ClaudeProvider()
        assert provider.is_available() is False
        assert provider.client is None

    def test_claude_model_name_mapping(self):
        """Test Claude model name mapping."""
        provider = ClaudeProvider()

        # Test known mapping
        assert provider._get_claude_model_name("claude-4-sonnet") == "claude-sonnet-4-20250514"

        # Test default fallback
        assert provider._get_claude_model_name("unknown-model") == "claude-sonnet-4-20250514"

    def test_claude_message_preparation(self):
        """Test Claude message preparation."""
        provider = ClaudeProvider()

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="User message"),
            Message(role="assistant", content="Assistant response"),
            Message(role="system", content="Another system message")
        ]

        system_prompt, chat_messages = provider._prepare_messages(messages)

        assert system_prompt == "System prompt Another system message"
        assert len(chat_messages) == 2
        assert chat_messages[0]["role"] == "user"
        assert chat_messages[0]["content"] == "User message"
        assert chat_messages[1]["role"] == "assistant"
        assert chat_messages[1]["content"] == "Assistant response"

    @patch.dict('os.environ', {}, clear=True)
    @pytest.mark.asyncio
    async def test_claude_provider_raises_exception_when_not_configured(self):
        """Test that Claude provider raises exception when not configured."""
        provider = ClaudeProvider()
        request = ChatCompletionRequest(
            model="claude-4-sonnet",
            messages=[Message(role="user", content="Test")]
        )

        with pytest.raises(HTTPException) as exc_info:
            await provider.generate_response(request)

        assert exc_info.value.status_code == 500
        assert "not configured" in str(exc_info.value.detail)

    @patch.dict('os.environ', {'CLAUDE_API_KEY': 'test-key'})
    @patch('anthropic.Anthropic')
    @pytest.mark.asyncio
    async def test_claude_provider_generate_response_success(self, mock_client_class):
        """Test successful response generation with mocked Claude API."""
        # Mock the response
        mock_response = Mock()
        mock_response.id = "msg-claude-test"
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Claude response"

        # Mock the client
        mock_client = Mock()
        mock_client.messages.create = Mock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        provider = ClaudeProvider()
        request = ChatCompletionRequest(
            model="claude-4-sonnet",
            messages=[
                Message(role="system", content="System prompt"),
                Message(role="user", content="Test message")
            ]
        )

        response = await provider.generate_response(request)

        assert response["id"] == "msg-claude-test"
        assert response["object"] == "chat.completion"
        assert len(response["choices"]) == 1
        assert response["choices"][0]["message"]["content"] == "Claude response"
        assert response["choices"][0]["message"]["role"] == "assistant"

        # Verify the SDK was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["system"] == "System prompt"
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["content"] == "Test message"

    @patch.dict('os.environ', {'CLAUDE_API_KEY': 'test-key'})
    @patch('anthropic.Anthropic')
    @pytest.mark.asyncio
    async def test_claude_provider_generate_streaming_response_success(self, mock_client_class):
        """Test successful streaming response generation with mocked Claude API."""
        # Mock streaming events
        mock_event1 = Mock()
        mock_event1.type = "content_block_delta"
        mock_event1.delta = Mock()
        mock_event1.delta.text = "Hello"

        mock_event2 = Mock()
        mock_event2.type = "content_block_delta"
        mock_event2.delta = Mock()
        mock_event2.delta.text = " world"

        mock_event3 = Mock()
        mock_event3.type = "message_stop"

        # Mock the streaming context manager
        mock_stream = Mock()
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=None)
        mock_stream.__iter__ = Mock(return_value=iter([mock_event1, mock_event2, mock_event3]))

        # Mock the client
        mock_client = Mock()
        mock_client.messages.stream = Mock(return_value=mock_stream)
        mock_client_class.return_value = mock_client

        provider = ClaudeProvider()
        request = ChatCompletionRequest(
            model="claude-4-sonnet",
            messages=[Message(role="user", content="Test message")]
        )

        chunks = []
        async for chunk in provider.generate_streaming_response(request):
            chunks.append(chunk)

        assert len(chunks) >= 3  # Role chunk + content chunks + [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"

        # Check for proper SSE format and content
        content_found = False
        for chunk in chunks[:-1]:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")
            if "Hello" in chunk or "world" in chunk:
                content_found = True

        assert content_found

    @patch.dict('os.environ', {'CLAUDE_API_KEY': 'test-key'})
    @patch('anthropic.Anthropic')
    @pytest.mark.asyncio
    async def test_claude_provider_handles_api_error(self, mock_client_class):
        """Test that Claude provider handles API errors gracefully."""
        # Mock the client to raise an exception
        mock_client = Mock()
        mock_client.messages.create = Mock(side_effect=Exception("Claude API Error"))
        mock_client_class.return_value = mock_client

        provider = ClaudeProvider()
        request = ChatCompletionRequest(
            model="claude-4-sonnet",
            messages=[Message(role="user", content="Test message")]
        )

        with pytest.raises(HTTPException) as exc_info:
            await provider.generate_response(request)

        assert exc_info.value.status_code == 500
        assert "Claude API error" in str(exc_info.value.detail)

    @patch.dict('os.environ', {'CLAUDE_API_KEY': 'test-key'})
    @patch('anthropic.Anthropic')
    @pytest.mark.asyncio
    async def test_claude_provider_handles_different_content_formats(self, mock_client_class):
        """Test that Claude provider handles different response content formats."""
        # Test with string content
        mock_response = Mock()
        mock_response.id = "msg-claude-string"
        mock_response.content = "Direct string content"

        mock_client = Mock()
        mock_client.messages.create = Mock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        provider = ClaudeProvider()
        request = ChatCompletionRequest(
            model="claude-4-sonnet",
            messages=[Message(role="user", content="Test message")]
        )

        response = await provider.generate_response(request)
        assert response["choices"][0]["message"]["content"] == "Direct string content"


class TestProviderInstances:
    """Test the global provider instances."""

    def test_global_provider_instances_exist(self):
        """Test that global provider instances are created."""
        from app.services.llms import (
            azure_openai_provider,
            claude_provider,
            echo_provider,
        )

        assert azure_openai_provider is not None
        assert isinstance(azure_openai_provider, AzureOpenAIProvider)

        assert claude_provider is not None
        assert isinstance(claude_provider, ClaudeProvider)

        assert echo_provider is not None
        assert isinstance(echo_provider, EchoProvider)

        # Echo provider should always be available
        assert echo_provider.is_available() is True


class TestLegacyCompatibility:
    """Test backward compatibility with legacy methods."""

    @pytest.mark.asyncio
    async def test_echo_provider_legacy_methods(self):
        """Test that legacy methods work on Echo provider."""
        provider = EchoProvider()
        request = ChatCompletionRequest(
            model="echo-model",
            messages=[Message(role="user", content="Test")]
        )

        # Test legacy sync method
        response = provider.create_chat_completion(request)
        assert response["object"] == "chat.completion"

        # Test legacy async streaming method
        chunks = []
        async for chunk in provider.create_chat_completion_stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_echo_provider_static_legacy_method(self):
        """Test that static legacy method works."""
        request = ChatCompletionRequest(
            model="echo-model",
            messages=[Message(role="user", content="Test")]
        )

        response = EchoProvider.generate_response_static(request)
        assert response["object"] == "chat.completion"


# Test markers and fixtures
@pytest.fixture
def sample_chat_request():
    """Fixture providing a sample chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello, how are you?")
        ],
        temperature=0.7,
        max_tokens=150
    )


@pytest.fixture
def streaming_chat_request():
    """Fixture providing a sample streaming chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[Message(role="user", content="Tell me a short story.")],
        temperature=0.7,
        max_tokens=100,
        stream=True
    )
