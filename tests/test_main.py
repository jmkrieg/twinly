"""
Tests for the Twinly API with new structure.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app

client = TestClient(app)



def test_models_endpoint():
    """Test the models endpoint returns available models."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    
    # Check that basic models are always available
    model_ids = [model["id"] for model in data["data"]]
    assert "echo-model" in model_ids
    assert "agent-mode" in model_ids


def test_chat_completion_echo_model():
    """Test chat completion with echo model."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "echo-model",
            "messages": [
                {"role": "user", "content": "Hello, world!"}
            ],
            "stream": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "You said: Hello, world!" in data["choices"][0]["message"]["content"]


def test_chat_completion_streaming_echo():
    """Test streaming chat completion with echo model."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "echo-model",
            "messages": [
                {"role": "user", "content": "Test streaming"}
            ],
            "stream": True
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


@patch('app.logic.agent_controller.agent_controller.run_agent')
@pytest.mark.asyncio
async def test_chat_completion_agent_mode(mock_run_agent):
    """Test chat completion with agent mode."""
    # Mock the agent response
    mock_run_agent.return_value = "This is an agent response based on plan-reason-respond paradigm."
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "agent-mode",
            "messages": [
                {"role": "user", "content": "How do I learn Python?"}
            ],
            "stream": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "agent response" in data["choices"][0]["message"]["content"]
    
    # Verify the agent was called
    mock_run_agent.assert_called_once()


def test_chat_completion_agent_mode_streaming_not_supported():
    """Test that agent mode doesn't support streaming."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "agent-mode",
            "messages": [
                {"role": "user", "content": "Test"}
            ],
            "stream": True
        }
    )
    assert response.status_code == 400
    assert "Streaming is not supported for agent-mode" in response.json()["detail"]


def test_chat_completion_invalid_model():
    """Test chat completion with non-existent model falls back to echo."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "non-existent-model",
            "messages": [
                {"role": "user", "content": "Test"}
            ],
            "stream": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "You said: Test" in data["choices"][0]["message"]["content"]


def test_chat_completion_missing_messages():
    """Test chat completion with missing messages."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "echo-model",
            "messages": [],
            "stream": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "No user message provided" in data["choices"][0]["message"]["content"]


def test_chat_completion_system_and_user_messages():
    """Test chat completion with both system and user messages."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "echo-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "stream": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "You said: What is 2+2?" in data["choices"][0]["message"]["content"]


# def test_demo_endpoint():
#     """Test the demo endpoint."""
#     response = client.get("/demo")
#     assert response.status_code == 200
#     assert response.headers["content-type"] == "text/html; charset=utf-8"


@patch('app.api.chat.azure_openai_provider.is_available')
def test_models_with_azure_available(mock_azure_available):
    """Test models endpoint when Azure OpenAI is available."""
    mock_azure_available.return_value = True
    
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    
    model_ids = [model["id"] for model in data["data"]]
    assert "gpt4.1-chat" in model_ids


@patch('app.services.claude.claude_service.is_available')
def test_models_with_claude_available(mock_claude_available):
    """Test models endpoint when Claude is available."""
    mock_claude_available.return_value = True
    
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    
    model_ids = [model["id"] for model in data["data"]]
    assert "claude-4-sonnet" in model_ids
