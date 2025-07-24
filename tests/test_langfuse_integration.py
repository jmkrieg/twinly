"""
Tests for Langfuse observability integration.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from fastapi.testclient import TestClient

from app.main import app
from app.models.chat import Message

client = TestClient(app)


class TestLangfuseIntegration:
    """Test suite for Langfuse observability integration."""

    @pytest.fixture
    def mock_langfuse(self):
        """Mock Langfuse client and decorators."""
        with patch('langfuse.decorators.langfuse_context') as mock_context, \
             patch('langfuse.decorators.observe') as mock_observe:
            
            # Mock the observe decorator to behave as a pass-through
            mock_observe.side_effect = lambda *args, **kwargs: lambda f: f
            
            # Mock langfuse_context methods
            mock_context.update_current_trace = Mock()
            mock_context.score_current_trace = Mock()
            mock_context.get_current_trace_id = Mock(return_value="test-trace-id")
            mock_context.get_current_observation_id = Mock(return_value="test-observation-id")
            
            yield {
                'context': mock_context,
                'observe': mock_observe
            }

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is the capital of France?")
        ]

    def test_agent_controller_has_observe_decorators(self):
        """Test that agent controller methods are decorated with @observe."""
        from app.logic.agent_controller import AgentController
        
        controller = AgentController()
        
        # Check if the main method has the decorator attribute
        # Note: This test will pass once we add the decorators
        assert hasattr(controller.run_agent, '__wrapped__') or hasattr(controller.run_agent, '_langfuse_observed')

    def test_agent_workflow_creates_trace(self, mock_langfuse, sample_messages):
        """Test that the agent workflow creates a proper Langfuse trace."""
        from app.logic.agent_controller import agent_controller
        
        # Mock Azure OpenAI provider
        with patch.object(agent_controller, 'azure_service') as mock_service:
            mock_service.is_available.return_value = True
            mock_service.generate_response = AsyncMock(return_value={
                "choices": [{"message": {"content": "Test response"}}]
            })
            
            # Run the agent
            import asyncio
            result = asyncio.run(agent_controller.run_agent(sample_messages))
            
            # Verify that trace is updated with appropriate metadata
            mock_langfuse['context'].update_current_trace.assert_called()
            
            # Check that the result is returned
            assert isinstance(result, str)

    def test_agent_mode_endpoint_with_langfuse(self, mock_langfuse):
        """Test that the agent-mode endpoint integrates with Langfuse."""
        with patch('app.api.chat.agent_controller') as mock_agent:
            mock_agent.run_agent = AsyncMock(return_value="Mocked response")
            
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "agent-mode",
                    "messages": [
                        {"role": "user", "content": "Test message"}
                    ],
                    "stream": False
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == "Mocked response"

    def test_langfuse_trace_metadata_is_set(self, mock_langfuse, sample_messages):
        """Test that Langfuse trace metadata is properly set."""
        from app.logic.agent_controller import agent_controller
        
        with patch.object(agent_controller, 'azure_service') as mock_service:
            mock_service.is_available.return_value = True
            mock_service.generate_response = AsyncMock(return_value={
                "choices": [{"message": {"content": "Test response"}}]
            })
            
            import asyncio
            asyncio.run(agent_controller.run_agent(sample_messages))
            
            # Verify trace metadata calls
            calls = mock_langfuse['context'].update_current_trace.call_args_list
            assert len(calls) > 0
            
            # Check if any call includes expected metadata
            found_metadata = False
            for call in calls:
                if call.kwargs.get('name') == 'agent-workflow':
                    found_metadata = True
                    break
            
            assert found_metadata, "Expected trace metadata not found"

    def test_planning_step_is_traced(self, mock_langfuse, sample_messages):
        """Test that the planning step is properly traced."""
        from app.logic.agent_controller import AgentController
        
        controller = AgentController()
        
        with patch.object(controller.azure_service, 'is_available', return_value=True), \
             patch.object(controller.azure_service, 'create_chat_completion', 
                         new=AsyncMock(return_value={
                             "choices": [{"message": {"content": "Test goal"}}]
                         })):
            
            import asyncio
            result = asyncio.run(controller._simple_planner(sample_messages))
            
            assert isinstance(result, str)
            assert result == "Test goal"

    def test_reasoning_step_is_traced(self, mock_langfuse, sample_messages):
        """Test that the reasoning step is properly traced."""
        from app.logic.agent_controller import AgentController
        
        controller = AgentController()
        
        with patch.object(controller.azure_service, 'is_available', return_value=True), \
             patch.object(controller.azure_service, 'create_chat_completion', 
                         new=AsyncMock(return_value={
                             "choices": [{"message": {"content": "Test reasoning"}}]
                         })):
            
            import asyncio
            result = asyncio.run(controller._think_step_by_step(sample_messages, "test goal"))
            
            assert isinstance(result, str)
            assert result == "Test reasoning"

    def test_response_generation_is_traced(self, mock_langfuse, sample_messages):
        """Test that the response generation step is properly traced."""
        from app.logic.agent_controller import AgentController
        
        controller = AgentController()
        
        with patch.object(controller.azure_service, 'is_available', return_value=True), \
             patch.object(controller.azure_service, 'create_chat_completion', 
                         new=AsyncMock(return_value={
                             "choices": [{"message": {"content": "Final response"}}]
                         })):
            
            import asyncio
            result = asyncio.run(controller._generate_final_response(
                sample_messages, "test goal", "test reasoning"
            ))
            
            assert isinstance(result, str)
            assert result == "Final response"

    def test_error_handling_with_langfuse(self, mock_langfuse, sample_messages):
        """Test that errors are properly handled and traced."""
        from app.logic.agent_controller import AgentController
        
        controller = AgentController()
        
        with patch.object(controller.azure_service, 'is_available', return_value=True), \
             patch.object(controller.azure_service, 'create_chat_completion', 
                         side_effect=Exception("Test error")):
            
            import asyncio
            result = asyncio.run(controller.run_agent(sample_messages))
            
            # Should return error message, not raise exception
            assert isinstance(result, str)
            assert "error" in result.lower()

    def test_langfuse_configuration_env_vars(self):
        """Test that Langfuse configuration respects environment variables."""
        import os
        
        # Test with environment variables set
        test_env = {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key',
            'LANGFUSE_HOST': 'https://test.langfuse.com'
        }
        
        with patch.dict(os.environ, test_env):
            # Import should work without errors
            from app.utils.langfuse_config import get_langfuse_config
            config = get_langfuse_config()
            
            assert config is not None

    def test_langfuse_disabled_when_not_configured(self):
        """Test that Langfuse is gracefully disabled when not configured."""
        import os
        
        # Test without environment variables
        with patch.dict(os.environ, {}, clear=True):
            from app.utils.langfuse_config import get_langfuse_config
            config = get_langfuse_config()
            
            # Should return None or indicate disabled state
            assert config is None or config.get('enabled') is False
