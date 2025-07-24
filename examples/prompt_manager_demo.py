#!/usr/bin/env python3
"""
Example usage of the PromptManager class.

This script demonstrates how to use the PromptManager to render templates
with various parameters and metadata.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path to import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.prompts.prompt_manager import PromptManager


def main():
    """Demonstrate PromptManager functionality."""
    print("=== PromptManager Demo ===\n")
    
    # List available templates
    print("Available templates:")
    templates = PromptManager.list_templates()
    for template in templates:
        print(f"  - {template}")
    print()
    
    # Example 1: System prompt with all parameters
    print("1. System Prompt (full parameters):")
    system_prompt = PromptManager.get_prompt(
        "system_prompt",
        assistant_name="Demo Assistant",
        current_date=datetime.now().strftime("%Y-%m-%d"),
        specialized_knowledge="Python, FastAPI, and AI development"
    )
    print(system_prompt)
    print("-" * 50)
    
    # Example 2: System prompt with minimal parameters (using defaults)
    print("2. System Prompt (minimal parameters):")
    minimal_system_prompt = PromptManager.get_prompt(
        "system_prompt",
        current_date=datetime.now().strftime("%Y-%m-%d")
    )
    print(minimal_system_prompt)
    print("-" * 50)
    
    # Example 3: Greeting template
    print("3. Greeting Template:")
    greeting = PromptManager.get_prompt(
        "greeting",
        user_name="Alice",
        time_of_day="morning",
        task_type="Python programming",
        previous_conversation=True,
        previous_topic="FastAPI development"
    )
    print(greeting)
    print("-" * 50)
    
    # Example 4: Template info
    print("4. Template Information:")
    info = PromptManager.get_template_info("system_prompt")
    print(f"Name: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Author: {info['author']}")
    print(f"Variables: {', '.join(info['variables'])}")
    print(f"Metadata: {info['frontmatter']}")
    print("-" * 50)
    
    # Example 5: Error handling
    print("5. Error Handling Demo:")
    try:
        PromptManager.get_prompt("nonexistent_template")
    except Exception as e:
        print(f"Expected error: {e}")
    print()


if __name__ == "__main__":
    main()
