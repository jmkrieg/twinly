"""
Tests for memory service integration with embedding providers.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.memory_qdrant import ConversationMemory
from app.services.embeddings.base_embedding_provider import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, available: bool = True, embedding_dims: int = 1536):
        self.available = available
        self.embedding_dims = embedding_dims
        self.call_log = []
    
    def is_available(self) -> bool:
        return self.available
    
    async def create_embedding(self, text: str, memory_action=None):
        self.call_log.append({
            'text': text,
            'memory_action': memory_action
        })
        if not self.available:
            raise Exception("Mock provider not available")
        # Return a mock embedding vector
        return [0.1] * self.embedding_dims


class TestMemoryServiceEmbeddingIntegration:
    """Test memory service integration with embedding providers."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client."""
        client = MagicMock()
        client.collection_exists.return_value = True
        client.upsert = MagicMock()
        client.query_points = MagicMock()
        return client
    
    @pytest.fixture
    def mock_embedding_provider(self):
        """Mock embedding provider."""
        return MockEmbeddingProvider()
    
    @pytest.fixture
    def memory_service_with_embedding_provider(self, mock_qdrant_client, mock_embedding_provider):
        """Memory service with mock embedding provider."""
        with patch('app.services.memory_qdrant.QdrantClient') as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client
            memory = ConversationMemory(
                qdrant_url="http://localhost:6333",
                collection_name="test_collection",
                embedding_provider=mock_embedding_provider
            )
            return memory, mock_embedding_provider
    
    async def test_memory_service_uses_embedding_provider(self, memory_service_with_embedding_provider):
        """Test that memory service uses the provided embedding provider."""
        memory, provider = memory_service_with_embedding_provider
        
        # Test creating embedding directly
        embedding = await memory._create_embedding("test text", "search")
        
        assert len(embedding) == 1536
        assert len(provider.call_log) == 1
        assert provider.call_log[0]['text'] == "test text"
        assert provider.call_log[0]['memory_action'] == "search"
    
    async def test_store_fact_uses_add_action(self, memory_service_with_embedding_provider):
        """Test that storing facts uses 'add' memory action."""
        memory, provider = memory_service_with_embedding_provider
        
        # Mock the search for duplicates to return empty list
        provider.call_log.clear()
        mock_points = MagicMock()
        mock_points.points = []
        memory.client.query_points.return_value = mock_points
        
        await memory.store_fact(
            user_id="test_user",
            fact_text="User likes pizza",
            category="preference"
        )
        
        # Should have two calls: one for searching duplicates, one for storing
        assert len(provider.call_log) == 2
        assert provider.call_log[0]['memory_action'] == "search"  # duplicate check
        assert provider.call_log[1]['memory_action'] == "add"     # storing fact
    
    async def test_search_memory_uses_search_action(self, memory_service_with_embedding_provider):
        """Test that searching memory uses 'search' memory action."""
        memory, provider = memory_service_with_embedding_provider
        
        # Mock search results
        mock_points = MagicMock()
        mock_points.points = []
        memory.client.query_points.return_value = mock_points
        
        provider.call_log.clear()
        await memory.search_memory(
            user_id="test_user",
            query_text="pizza preferences"
        )
        
        assert len(provider.call_log) == 1
        assert provider.call_log[0]['memory_action'] == "search"
    
    async def test_store_conversation_uses_add_action(self, memory_service_with_embedding_provider):
        """Test that storing conversations uses 'add' memory action."""
        memory, provider = memory_service_with_embedding_provider
        
        provider.call_log.clear()
        await memory.store_conversation_turn(
            user_id="test_user",
            user_message="Hello",
            assistant_response="Hi there!"
        )
        
        # Should have two calls: one for user message, one for assistant response
        assert len(provider.call_log) == 2
        assert all(call['memory_action'] == "add" for call in provider.call_log)
        assert provider.call_log[0]['text'] == "Hello"
        assert provider.call_log[1]['text'] == "Hi there!"
    
    async def test_fallback_when_provider_unavailable(self, mock_qdrant_client):
        """Test fallback behavior when embedding provider is unavailable."""
        unavailable_provider = MockEmbeddingProvider(available=False)
        
        with patch('app.services.memory_qdrant.QdrantClient') as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client
            memory = ConversationMemory(
                qdrant_url="http://localhost:6333",
                collection_name="test_collection",
                embedding_provider=unavailable_provider
            )
        
        # Should generate fallback embedding
        embedding = await memory._create_embedding("test text")
        
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        # Should be non-zero values (fallback uses hash-based embedding)
        assert any(x > 0 for x in embedding)
    
    def test_memory_service_initialization_with_embedding_provider(self, mock_embedding_provider):
        """Test memory service initialization with embedding provider."""
        with patch('app.services.memory_qdrant.QdrantClient'):
            memory = ConversationMemory(
                embedding_provider=mock_embedding_provider
            )
        
        assert memory.embedding_provider is mock_embedding_provider
        assert hasattr(memory, 'llm_provider')  # Should have LLM provider for chat
    
    def test_memory_service_initialization_with_defaults(self):
        """Test memory service initialization with default providers."""
        with patch('app.services.memory_qdrant.QdrantClient'):
            memory = ConversationMemory()
        
        # Should use the global azure embedding provider by default
        assert memory.embedding_provider is not None
        assert hasattr(memory, 'llm_provider')


class TestGlobalMemoryFunctions:
    """Test global memory functions with embedding providers."""
    
    def test_get_conversation_memory_with_embedding_provider(self):
        """Test getting conversation memory with embedding provider."""
        mock_provider = MockEmbeddingProvider()
        
        from app.services.memory_qdrant import get_conversation_memory
        
        with patch('app.services.memory_qdrant.QdrantClient'):
            memory = get_conversation_memory(
                collection_name="test_collection",
                embedding_provider=mock_provider
            )
        
        assert memory.embedding_provider is mock_provider
    
    def test_get_agent_conversation_memory_with_embedding_provider(self):
        """Test getting agent conversation memory with embedding provider."""
        mock_provider = MockEmbeddingProvider()
        
        from app.services.memory_qdrant import get_agent_conversation_memory
        
        with patch('app.services.memory_qdrant.QdrantClient'):
            memory = get_agent_conversation_memory(
                agent_name="test-agent",
                embedding_provider=mock_provider
            )
        
        assert memory.embedding_provider is mock_provider
