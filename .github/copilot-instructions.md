# GitHub Copilot Instructions

## Goal of the Repository
This repository aims to provide a simple FastAPI application that serves as a container for the development of LLM Powered Chatbots and agents.

The application will be consumed by a Open WebUI Frontend and maintains OpenAI API compatibility.

## Technology Stack
- **Core Framework**: FastAPI with uvicorn
- **Testing**: pytest with httpx for async testing
- **Environment Management**: python-dotenv for configuration
- **AI/LLM Integration**: OpenAI, Anthropic (Claude)
- **Observability**: Langfuse for tracing and monitoring
- **Containerization**: Docker and Docker Compose

## Environment Setup
- **CRITICAL**: Always activate the conda environment first: `conda activate twinly`
- Never install packages globally or in the local Python environment
- Use `pip install -r requirements.txt` within the activated environment

## Project Structure
- `app/main.py`: FastAPI application entry point
- `app/api/`: API route handlers
- `app/models/`: Pydantic models for request/response validation
- `app/services/`: External service integrations
- `app/services/llms/`: LLM provider implementations with unified BaseLLMProvider interface
- `app/services/embeddings/`: Embedding provider implementations with unified BaseEmbeddingProvider interface
- `app/logic/`: Business logic and agent controllers
- `app/utils/`: Utility functions (logging, config, decorators)
- `tests/`: Comprehensive test suite

## Conventions
- Use only `FastAPI` and `uvicorn` as the core framework
- DO NOT INSTALL INTO LOCAL PYTHON ENVIRONMENT. ALWAYS USE conda activate twinly first to activate the environment
- Keep everything as minimal as possible
- Do not use unnecessary dependencies
- Write clear, well-commented code like a professional developer
- API responses are always in JSON format
- Write tests for everything before implementing new features
- Use type hints for all function signatures
- Use `pydantic` models for request and response validation
- Do not hallucinate. If you are unsure about something, ask for clarification
- Use descriptive variable and function names
- Follow Python's PEP 8 style guide for code formatting
- Use `pytest` for testing

## API Design Principles
- Follow OpenAI API format for `/v1/chat/completions` and `/v1/models` endpoints
- Maintain backward compatibility with OpenAI client libraries
- Support both streaming and non-streaming responses
- Use consistent error response formats
- Handle external API failures gracefully with fallback responses

## Testing Standards
- Write tests for all endpoints using FastAPI's TestClient
- Mock external API calls (OpenAI, Claude) to avoid costs and ensure reliability
- Test both success and error scenarios
- Include integration tests for the full request/response cycle
- Test Langfuse integration separately with mocked dependencies

## Configuration Management
- Use environment variables for all external service configurations
- Store sensitive information (API keys) in environment variables only
- Use .env files for local development (never commit to version control)
- Document all required environment variables in README.md

## Error Handling
- Use FastAPI's HTTPException for API errors
- Implement proper error responses that match OpenAI API format
- Log errors appropriately using the structured logging system
- Handle external API failures gracefully with fallback responses

## Observability and Monitoring
- Use Langfuse decorators for tracing LLM calls
- Implement structured logging throughout the application
- Include performance monitoring for API response times
- Track usage statistics for different models and agents

## Avoid
- Complex authentication systems (keep it simple for now)
- Database dependencies (use in-memory storage if needed)
- Unnecessary middleware or complex routing
- Direct file system operations (use proper service layers)
- Hardcoded API keys or configuration values
- Breaking changes to OpenAI API compatibility
- Installing packages globally or in the local Python environment

## Deployment Considerations
- Use Docker for containerization
- Support both development and production configurations
- Ensure proper health checks are implemented
- Document environment-specific configurations

## Development Process
1. **Start with a clear understanding of the feature or bug**: Read the request, issue or feature request carefully.
2. **Write tests first**: Before implementing any new feature, write tests that define the expected behavior.
3. **Plan the implementation**: Break down the feature into smaller tasks and outline how you will implement it.
4. **Check if you need additional information**: If you are unsure about any aspect of the tools use context7 mcp server to get more information.
5. **Ask for clarification**: If you have any doubts or need more information, ask for clarification before proceeding.
6. **Implement the feature**: Write the code to implement the feature, following the conventions and guidelines.
7. **Run tests**: After implementing the feature, run the tests to ensure everything works as expected.
8. **Review and refactor**: Review your code for any improvements or optimizations, and refactor if necessary.
9. **Document the code**: Ensure that your code is well-documented, including comments and docstrings.
10. **Provide a summary**: At the end of your implementation, provide a summary of what you did and any important notes.

## How to add a new LLM provider

All LLM providers now use a unified architecture based on the abstract `BaseLLMProvider` class:

### Steps:
1. **Create provider class**: Create a new file in `app/services/llms/my_provider.py`
2. **Inherit from BaseLLMProvider**: Implement all abstract methods:
   - `is_available() -> bool`: Checks if the provider is available (API key etc.)
   - `async generate_response(req: ChatCompletionRequest) -> Dict[str, Any]`: Non-streaming response
   - `async generate_streaming_response(req: ChatCompletionRequest) -> AsyncGenerator[str, None]`: Streaming response
3. **Global instance**: Create a global instance at the end of the file: `my_provider = MyProvider()`
4. **Add export**: Add the provider to `app/services/llms/__init__.py`
5. **Model mapping**: Update `MODEL_MAPPING` in `app/api/chat.py`
6. **Models endpoint**: Add the model to the `/v1/models` endpoint (with `is_available()` check)
7. **Write tests**: Create contract and mock tests in `tests/test_llm_providers.py`
8. **Linting & type checks**: Ensure `ruff check`, `black` and `mypy` pass
9. **Documentation**: Document required environment variables

### Advantages of the new architecture:
- **Unified API**: All providers implement the same methods
- **Interchangeability**: Providers can be easily swapped via MODEL_MAPPING
- **Testability**: Each provider can be tested in isolation with mocks
- **Extensibility**: New providers are easy to add without changing the chat API
- **Factory pattern**: MODEL_MAPPING enables dynamic provider selection

## How to add a new Embedding provider

All embedding providers now use a unified architecture based on the abstract `BaseEmbeddingProvider` class:

### Steps:
1. **Create provider class**: Create a new file in `app/services/embeddings/my_provider.py`
2. **Inherit from BaseEmbeddingProvider**: Implement all abstract methods:
   - `is_available() -> bool`: Checks if the provider is available (API key etc.)
   - `async create_embedding(text: str, action: str = "add") -> List[float]`: Creates embedding for given text
   - `get_embedding_dimension() -> int`: Returns the dimension of the embeddings
3. **Global instance**: Create a global instance at the end of the file: `my_embedding_provider = MyEmbeddingProvider()`
4. **Add export**: Add the provider to `app/services/embeddings/__init__.py`
5. **Memory service integration**: The provider can optionally be passed to `ConversationMemory`
6. **Write tests**: Create contract and mock tests in `tests/test_embedding_providers.py`
7. **Linting & type checks**: Ensure `ruff check`, `black` and `mypy` pass
8. **Documentation**: Document required environment variables

### Advantages of the new architecture:
- **Unified API**: All providers implement the same methods
- **Dependency injection**: Memory service can work with different embedding providers
- **Testability**: Each provider can be tested in isolation with mocks
- **Extensibility**: New providers are easy to add without changing the memory service
- **Fallback logic**: Graceful degradation when providers are not available
- **Action support**: Embeddings can be optimized for different actions (add, search)

### Environment Variables for Embedding Providers:
- Use specific env variables for each provider (e.g. `AZURE_EMBEDDING_ENDPOINT`)
- Implement fallback logic for legacy variables
- Document all required variables in README.md

## Suggestions
- If the copilot-Instructions.md needs to be updated, please let me know and suggest changes. DO NOT MODIFY IT DIRECTLY.
- If you have any questions or need clarification, feel free to ask.