"""
Tests for the embedding providers.
"""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.services.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.services.embeddings.azure_openai import AzureEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, available: bool = True, embedding_dims: int = 1536):
        self.available = available
        self.embedding_dims = embedding_dims
    
    def is_available(self) -> bool:
        return self.available
    
    async def create_embedding(self, text: str, memory_action=None):
        if not self.available:
            raise Exception("Mock provider not available")
        # Return a mock embedding vector
        return [0.1] * self.embedding_dims


class TestBaseEmbeddingProvider:
    """Test the base embedding provider interface."""
    
    def test_base_provider_is_abstract(self):
        """Test that BaseEmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbeddingProvider()
    
    async def test_mock_provider_works(self):
        """Test that the mock provider works correctly."""
        provider = MockEmbeddingProvider()
        
        assert provider.is_available() is True
        embedding = await provider.create_embedding("test text")
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
    
    async def test_mock_provider_unavailable(self):
        """Test mock provider when unavailable."""
        provider = MockEmbeddingProvider(available=False)
        
        assert provider.is_available() is False
        with pytest.raises(Exception):
            await provider.create_embedding("test text")


class TestAzureEmbeddingProvider:
    """Test Azure embedding provider."""
    
    @pytest.fixture
    def mock_azure_client(self):
        """Mock Azure OpenAI client."""
        client = AsyncMock()
        client.embeddings.create = AsyncMock()
        return client
    
    @pytest.fixture
    def azure_provider_with_client(self, mock_azure_client):
        """Azure embedding provider with mocked client."""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small"
        }):
            provider = AzureEmbeddingProvider()
            provider.client = mock_azure_client
            return provider
    
    def test_provider_initialization_with_env_vars(self):
        """Test provider initialization with environment variables."""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-large"
        }):
            provider = AzureEmbeddingProvider()
            assert provider.endpoint == "https://test.openai.azure.com/"
            assert provider.api_key == "test-key"
            assert provider.embedding_deployment == "text-embedding-3-large"
            assert provider.is_available() is True
    
    def test_provider_initialization_without_env_vars(self):
        """Test provider initialization without required environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            provider = AzureEmbeddingProvider()
            assert provider.is_available() is False
    
    def test_provider_uses_legacy_env_var(self):
        """Test that provider supports legacy environment variable."""
        env_vars = {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
        }
        # Remove the new env var to ensure only legacy is set
        current_env = dict(os.environ)
        current_env.update(env_vars)
        current_env.pop("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", None)
        current_env["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"] = "text-embedding-ada-002"
        
        with patch.dict(os.environ, current_env, clear=True):
            provider = AzureEmbeddingProvider()
            assert provider.embedding_deployment == "text-embedding-ada-002"
    
    def test_provider_uses_safe_default(self):
        """Test that provider uses safe default deployment."""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key"
        }):
            provider = AzureEmbeddingProvider()
            assert provider.embedding_deployment == "text-embedding-3-small"
    
    async def test_create_embedding_success(self, azure_provider_with_client, mock_azure_client):
        """Test successful embedding creation."""
        # Mock the embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4]
        mock_azure_client.embeddings.create.return_value = mock_response
        
        result = await azure_provider_with_client.create_embedding("test text")
        
        mock_azure_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test text"
        )
        assert result == [0.1, 0.2, 0.3, 0.4]
    
    async def test_create_embedding_with_memory_action(self, azure_provider_with_client, mock_azure_client):
        """Test embedding creation with memory action."""
        # Mock the embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.5, 0.6, 0.7]
        mock_azure_client.embeddings.create.return_value = mock_response
        
        result = await azure_provider_with_client.create_embedding("search query", "search")
        
        mock_azure_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="search query"
        )
        assert result == [0.5, 0.6, 0.7]
    
    async def test_create_embedding_cleans_text(self, azure_provider_with_client, mock_azure_client):
        """Test that text is cleaned before embedding."""
        # Mock the embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2]
        mock_azure_client.embeddings.create.return_value = mock_response
        
        await azure_provider_with_client.create_embedding("text with\n\nnewlines  ")
        
        mock_azure_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="text with  newlines"
        )
    
    async def test_create_embedding_without_client_raises_error(self):
        """Test that embedding raises error when client is not configured."""
        provider = AzureEmbeddingProvider()
        provider.client = None
        
        with pytest.raises(HTTPException) as exc_info:
            await provider.create_embedding("test text")
        
        assert "Azure OpenAI embedding service is not configured" in str(exc_info.value.detail)
    
    async def test_create_embedding_api_error(self, azure_provider_with_client, mock_azure_client):
        """Test handling of API errors."""
        mock_azure_client.embeddings.create.side_effect = Exception("API Error")
        
        with pytest.raises(HTTPException) as exc_info:
            await azure_provider_with_client.create_embedding("test text")
        
        assert "Azure OpenAI embedding error" in str(exc_info.value.detail)
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimensions for different models."""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-large"
        }):
            provider = AzureEmbeddingProvider()
            assert provider.get_embedding_dimension() == 3072
        
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002"
        }):
            provider = AzureEmbeddingProvider()
            assert provider.get_embedding_dimension() == 1536
        
        # Test unknown model defaults to 1536
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "unknown-model"
        }):
            provider = AzureEmbeddingProvider()
            assert provider.get_embedding_dimension() == 1536


class TestEmbeddingProviderContract:
    """Contract tests for embedding providers."""
    
    @pytest.mark.parametrize("provider_factory", [
        lambda: MockEmbeddingProvider(available=True),
    ])
    async def test_provider_contract(self, provider_factory):
        """Test that providers follow the contract."""
        provider = provider_factory()
        
        # Test is_available returns boolean
        assert isinstance(provider.is_available(), bool)
        
        if provider.is_available():
            # Test create_embedding with text
            embedding = await provider.create_embedding("test text")
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)
            
            # Test create_embedding with memory action
            embedding_with_action = await provider.create_embedding("test text", "search")
            assert isinstance(embedding_with_action, list)
            assert len(embedding_with_action) > 0
