# Verschoben aus dem Hauptverzeichnis
"""
Test module to verify that only user-provided facts are extracted and stored.
This ensures we don't store assistant-generated information.
"""

import pytest
import sys
import os
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.memory_qdrant import ConversationMemory
from app.services.embeddings.base_embedding_provider import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def is_available(self) -> bool:
        return True
    
    async def create_embedding(self, text: str, memory_action=None):
        # Return a mock embedding vector
        return [0.1] * 1536


@pytest.fixture
async def memory_service():
    """Create a memory service instance for testing."""
    embedding_provider = MockEmbeddingProvider()
    
    with patch('app.services.memory_qdrant.QdrantClient'):
        return ConversationMemory(
            embedding_provider=embedding_provider
        )

@pytest.mark.asyncio
async def test_user_personal_info_extraction(memory_service):
    """Test that user personal information is extracted correctly."""
    user_prompt = "Mein Name ist John Doe und ich esse gerne Steaks. Ich arbeite mit Python und FastAPI."
    assistant_response = "Schön Sie kennenzulernen, John! Steaks sind eine großartige Wahl. Sie könnten versuchen, ein FastAPI Backend für ein Restaurant-System zu entwickeln."
    goal = "Personal introduction and preferences"
    
    extracted_facts = await memory_service.extract_facts_from_conversation(
        prompt=user_prompt,
        response=assistant_response, 
        goal=goal
    )
    
    # Should capture user information
    assert "John Doe" in extracted_facts
    assert ("Steaks" in extracted_facts or "steaks" in extracted_facts)
    assert "Python" in extracted_facts
    assert "FastAPI" in extracted_facts
    
    # Should NOT capture assistant suggestions
    assert "Restaurant-System" not in extracted_facts
    assert "entwickeln" not in extracted_facts

@pytest.mark.asyncio
async def test_no_facts_from_simple_questions(memory_service):
    """Test that simple questions don't generate facts to store."""
    user_prompt = "Wie kann ich meine FastAPI Anwendung testen?"
    assistant_response = "Sie können pytest verwenden. Hier ist ein Beispiel mit TestClient..."
    goal = "Technical help request"
    
    extracted_facts = await memory_service.extract_facts_from_conversation(
        prompt=user_prompt,
        response=assistant_response,
        goal=goal
    )
    
    # Should indicate no significant facts to store
    assert ("No significant" in extracted_facts or len(extracted_facts.strip()) < 50)

@pytest.mark.asyncio
async def test_user_preference_corrections(memory_service):
    """Test that user corrections and confirmations are captured."""
    user_prompt = "Ja, das stimmt. Aber ich bevorzuge eigentlich TypeScript über JavaScript."
    assistant_response = "Verstanden! TypeScript bietet bessere Typsicherheit. Ich werde das notieren."
    goal = "Preference correction"
    
    extracted_facts = await memory_service.extract_facts_from_conversation(
        prompt=user_prompt,
        response=assistant_response,
        goal=goal
    )
    
    # Should capture user's preference
    assert "TypeScript" in extracted_facts
    assert "JavaScript" in extracted_facts
    
    # Should NOT capture assistant's explanation
    assert "Typsicherheit" not in extracted_facts
    assert "notieren" not in extracted_facts

@pytest.mark.asyncio
async def test_fallback_behavior_when_service_unavailable():
    """Test fallback behavior when Azure OpenAI service is not available."""
    # Create a mock service that's not available
    mock_azure_service = MagicMock()
    mock_azure_service.is_available.return_value = False
    
    with patch('app.services.memory_qdrant.QdrantClient'), patch.object(
        ConversationMemory, '_generate_hash_based_embedding', return_value="hash_embedding"
    ) as mock_hash_embedding:
        memory_service = ConversationMemory()
    
    user_prompt = "Mein Name ist Test User."
    assistant_response = "Hallo Test User!"
    goal = "Introduction"
    
    extracted_facts = await memory_service.extract_facts_from_conversation(
        prompt=user_prompt,
        response=assistant_response,
        goal=goal
    )
    
    # Should use fallback and only include user statement
    assert "Test User" in extracted_facts
    assert goal in extracted_facts
    assert "Hallo" not in extracted_facts  # Should not include assistant response

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

