# Verschoben aus dem Hauptverzeichnis
"""
Test duplicate fact detection functionality.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.memory_qdrant import ConversationMemory


class TestDuplicateFactDetection:
    """Test duplicate fact detection in memory storage."""
    
    @pytest.fixture
    def mock_azure_service(self):
        """Mock Azure OpenAI service."""
        service = AsyncMock()
        service.is_available.return_value = True
        service.create_embedding.return_value = [0.1] * 1536  # Mock embedding
        return service
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client."""
        client = MagicMock()
        client.collection_exists.return_value = True
        return client
    
    @pytest.fixture
    def memory_service(self, mock_azure_service, mock_qdrant_client):
        """Create memory service with mocked dependencies."""
        with patch('app.services.memory_qdrant.QdrantClient') as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client
            service = ConversationMemory()
        return service
    
    @pytest.mark.asyncio
    async def test_duplicate_fact_detection_prevents_storage(self, memory_service, mock_qdrant_client):
        """Test that similar facts are detected and prevent duplicate storage."""
        # Mock search results showing a similar fact exists
        mock_result = MagicMock()
        mock_result.payload = {
            "text": "John likes coffee in the morning",
            "category": "preference",
            "session_id": "test_session"
        }
        mock_result.score = 0.92  # High similarity
        mock_result.id = "existing_fact_id"
        
        mock_qdrant_client.query_points.return_value.points = [mock_result]
        
        # Try to store a similar fact
        result = await memory_service.store_fact(
            user_id="test_user",
            fact_text="John prefers coffee in the morning",
            category="preference",
            check_duplicates=True,
            similarity_threshold=0.85
        )
        
        # Should return False because duplicate was detected
        assert result is False
        
        # Should have searched for similar facts
        mock_qdrant_client.query_points.assert_called_once()
        
        # Should NOT have called upsert to store the duplicate
        mock_qdrant_client.upsert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_unique_fact_storage_proceeds(self, memory_service, mock_qdrant_client):
        """Test that unique facts are stored when no duplicates exist."""
        # Mock search results showing no similar facts
        mock_qdrant_client.query_points.return_value.points = []
        
        # Try to store a unique fact
        result = await memory_service.store_fact(
            user_id="test_user",
            fact_text="Sarah likes tea in the evening",
            category="preference",
            check_duplicates=True,
            similarity_threshold=0.85
        )
        
        # Should return True because no duplicates were found
        assert result is True
        
        # Should have searched for similar facts
        mock_qdrant_client.query_points.assert_called_once()
        
        # Should have called upsert to store the unique fact
        mock_qdrant_client.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_duplicate_checking_can_be_disabled(self, memory_service, mock_qdrant_client):
        """Test that duplicate checking can be disabled."""
        # Try to store a fact with duplicate checking disabled
        result = await memory_service.store_fact(
            user_id="test_user",
            fact_text="Any fact content",
            category="general",
            check_duplicates=False
        )
        
        # Should return True regardless of duplicates
        assert result is True
        
        # Should NOT have searched for similar facts
        mock_qdrant_client.query_points.assert_not_called()
        
        # Should have called upsert to store the fact
        mock_qdrant_client.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_facts_with_filters(self, memory_service, mock_qdrant_client):
        """Test that similar fact search applies correct filters."""
        # Mock some search results
        mock_result = MagicMock()
        mock_result.payload = {
            "text": "Test fact",
            "category": "test_category",
            "session_id": "test_session"
        }
        mock_result.score = 0.9
        mock_result.id = "test_id"
        
        mock_qdrant_client.query_points.return_value.points = [mock_result]
        
        # Search for similar facts
        similar_facts = await memory_service.search_similar_facts(
            user_id="test_user",
            fact_text="Similar test fact",
            category="test_category",
            similarity_threshold=0.8
        )
        
        # Should return the mocked results
        assert len(similar_facts) == 1
        assert similar_facts[0]["text"] == "Test fact"
        assert similar_facts[0]["score"] == 0.9
        
        # Should have called query_points with proper filters
        mock_qdrant_client.query_points.assert_called_once()
        call_args = mock_qdrant_client.query_points.call_args
        
        # Verify the query filter was applied
        assert call_args.kwargs["query_filter"] is not None
        assert call_args.kwargs["score_threshold"] == 0.8
        assert call_args.kwargs["limit"] == 3  # Default limit
    
    @pytest.mark.asyncio
    async def test_similarity_threshold_configuration(self, memory_service, mock_qdrant_client):
        """Test that similarity threshold can be configured."""
        # Mock search results with different similarity scores
        high_similarity_result = MagicMock()
        high_similarity_result.payload = {"text": "High similarity fact", "category": "test"}
        high_similarity_result.score = 0.95
        high_similarity_result.id = "high_sim_id"
        
        # Test with high threshold - should detect duplicate
        mock_qdrant_client.query_points.return_value.points = [high_similarity_result]
        
        result_high_threshold = await memory_service.store_fact(
            user_id="test_user",
            fact_text="Test fact",
            category="test",
            check_duplicates=True,
            similarity_threshold=0.90  # High threshold - should find the 0.95 similarity result
        )
        
        # Should detect duplicate with high threshold
        assert result_high_threshold is False
        
        # Reset mock to return no results for very high threshold
        mock_qdrant_client.reset_mock()
        mock_qdrant_client.collection_exists.return_value = True
        mock_qdrant_client.query_points.return_value.points = []  # No results above very high threshold
        
        result_very_high_threshold = await memory_service.store_fact(
            user_id="test_user",
            fact_text="Test fact",
            category="test", 
            check_duplicates=True,
            similarity_threshold=0.98  # Very high threshold - no results returned
        )
        
        # Should not detect duplicate with very high threshold
        assert result_very_high_threshold is True


if __name__ == "__main__":
    pytest.main([__file__])

