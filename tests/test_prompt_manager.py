import pytest
from pathlib import Path
from jinja2 import TemplateNotFound
from app.prompts.prompt_manager import PromptManager


class TestPromptManager:
    """Test cases for PromptManager functionality."""
    
    def test_get_prompt_with_variables(self):
        """Test rendering a template with variables."""
        result = PromptManager.get_prompt(
            "system_prompt",
            assistant_name="Test Assistant",
            current_date="2025-06-09",
            specialized_knowledge="Python programming"
        )
        
        assert "Test Assistant" in result
        assert "2025-06-09" in result
        assert "Python programming" in result
    
    def test_get_prompt_with_defaults(self):
        """Test rendering a template with default values."""
        result = PromptManager.get_prompt(
            "system_prompt",
            current_date="2025-06-09"
        )
        
        assert "Twinly" in result  # Default value
        assert "2025-06-09" in result
    
    def test_get_prompt_greeting(self):
        """Test rendering the greeting template."""
        result = PromptManager.get_prompt(
            "greeting",
            user_name="John",
            time_of_day="morning",
            task_type="coding"
        )
        
        assert "Hello John!" in result
        assert "Good morning!" in result
        assert "coding" in result
    
    def test_get_prompt_nonexistent_template(self):
        """Test error handling for nonexistent template."""
        with pytest.raises(TemplateNotFound, match="Template nonexistent_template.j2 not found"):
            PromptManager.get_prompt("nonexistent_template")
    
    def test_get_template_info(self):
        """Test getting template information."""
        info = PromptManager.get_template_info("system_prompt")
        
        assert info["name"] == "system_prompt"
        assert info["description"] == "System prompt for the AI assistant"
        assert info["author"] == "Marinho Krieg"
        assert "assistant_name" in info["variables"]
        assert "current_date" in info["variables"]
        assert "specialized_knowledge" in info["variables"]
    
    def test_list_templates(self):
        """Test listing available templates."""
        templates = PromptManager.list_templates()
        
        assert "system_prompt" in templates
        assert "greeting" in templates
        assert isinstance(templates, list)
        assert len(templates) >= 2
    
    def test_template_with_conditional_content(self):
        """Test template with conditional content."""
        # With previous conversation
        result_with_prev = PromptManager.get_prompt(
            "greeting",
            user_name="Alice",
            previous_conversation=True,
            previous_topic="machine learning"
        )
        
        assert "machine learning" in result_with_prev
        
        # Without previous conversation
        result_without_prev = PromptManager.get_prompt(
            "greeting",
            user_name="Alice"
        )
        
        assert "machine learning" not in result_without_prev
    
    def test_template_directory_creation(self):
        """Test that templates directory is created if it doesn't exist."""
        templates_path = Path(__file__).parent.parent / "app" / "prompts" / "templates"
        assert templates_path.exists()
        assert templates_path.is_dir()


if __name__ == "__main__":
    pytest.main([__file__])
