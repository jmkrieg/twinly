"""
Tests for the agent controller functionality.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.logic.agent_controller import AgentController
from app.models.chat import Message


@pytest.fixture
def agent():
    """Create an agent controller instance for testing."""
    return AgentController()


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="How do I bake a chocolate cake?")
    ]


@pytest.mark.asyncio
async def test_agent_with_available_service(agent, sample_messages):
    """Test agent execution when Azure OpenAI service is available."""
    # Mock the Azure service to be available
    agent.azure_service.is_available = MagicMock(return_value=True)
    
    # Mock the service responses
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "Test response"
                }
            }
        ]
    }
    agent.azure_service.create_chat_completion = AsyncMock(return_value=mock_response)
    
    # Run the agent
    result = await agent.run_agent(sample_messages)
    
    # Verify the agent was called correctly
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Verify the service was called multiple times (for planning, reasoning, and response)
    assert agent.azure_service.create_chat_completion.call_count == 3


@pytest.mark.asyncio
async def test_agent_with_unavailable_service(agent, sample_messages):
    """Test agent execution when Azure OpenAI service is not available."""
    # Mock the Azure service to be unavailable
    agent.azure_service.is_available = MagicMock(return_value=False)
    
    # Run the agent
    result = await agent.run_agent(sample_messages)
    
    # Verify fallback behavior
    assert isinstance(result, str)
    assert "limited mode" in result.lower() or "not configured" in result.lower()


@pytest.mark.asyncio
async def test_agent_with_service_error(agent, sample_messages):
    """Test agent execution when service throws an error."""
    # Mock the Azure service to be available but throw an error
    agent.azure_service.is_available = MagicMock(return_value=True)
    agent.azure_service.create_chat_completion = AsyncMock(side_effect=Exception("Service error"))
    
    # Run the agent
    result = await agent.run_agent(sample_messages)
    
    # Verify error handling
    assert isinstance(result, str)
    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_simple_planner_fallback(agent, sample_messages):
    """Test the simple planner fallback when service is unavailable."""
    # Mock the Azure service to be unavailable
    agent.azure_service.is_available = MagicMock(return_value=False)
    
    # Test the planner
    goal = await agent._simple_planner(sample_messages)
    
    # Verify fallback behavior
    assert isinstance(goal, str)
    assert "chocolate cake" in goal.lower()


@pytest.mark.asyncio
async def test_think_step_by_step_fallback(agent, sample_messages):
    """Test the reasoning step fallback when service is unavailable."""
    # Mock the Azure service to be unavailable
    agent.azure_service.is_available = MagicMock(return_value=False)
    
    # Test the reasoning step
    goal = "Provide step-by-step instructions for baking a chocolate cake"
    reasoning = await agent._think_step_by_step(sample_messages, goal)
    
    # Verify fallback behavior
    assert isinstance(reasoning, str)
    assert goal in reasoning


@pytest.mark.asyncio
async def test_generate_final_response_fallback(agent, sample_messages):
    """Test the final response generation fallback when service is unavailable."""
    # Mock the Azure service to be unavailable
    agent.azure_service.is_available = MagicMock(return_value=False)
    
    # Test the final response generation
    goal = "Provide step-by-step instructions for baking a chocolate cake"
    reasoning = "Need to provide clear instructions"
    response = await agent._generate_final_response(sample_messages, goal, reasoning)
    
    # Verify fallback behavior
    assert isinstance(response, str)
    assert "limited mode" in response.lower()


@pytest.mark.asyncio
async def test_empty_messages_handling(agent):
    """Test agent behavior with empty messages."""
    empty_messages = []
    
    # Mock the Azure service to be unavailable for consistent testing
    agent.azure_service.is_available = MagicMock(return_value=False)
    
    # Run the agent
    result = await agent.run_agent(empty_messages)
    
    # Verify handling of empty messages
    assert isinstance(result, str)
    assert len(result) > 0
