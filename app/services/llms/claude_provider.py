"""
Claude service provider implementation.

This module contains the Claude service provider that inherits from BaseLLMProvider.
"""

import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import anthropic
from fastapi import HTTPException

from app.models.chat import ChatCompletionRequest, Message

from .base_llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseLLMProvider):
    """Claude/Anthropic service provider implementation."""

    def __init__(self):
        """Initialize Claude service with environment configuration."""
        self.api_key = os.getenv("CLAUDE_API_KEY")

        self.client = None
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("Claude provider initialized successfully")
        else:
            logger.warning("Claude provider not configured - missing API key")

    def is_available(self) -> bool:
        """Check if Claude service is available."""
        return self.client is not None

    def _get_claude_model_name(self, model_id: str) -> str:
        """
        Map public model names to Claude API model names.

        Args:
            model_id: The public model identifier

        Returns:
            The actual Claude API model name
        """
        model_mapping = {"claude-4-sonnet": "claude-sonnet-4-20250514"}

        return model_mapping.get(
            model_id, "claude-sonnet-4-20250514"
        )  # Default to Sonnet

    def _prepare_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """
        Prepare messages for Claude API by separating system messages.

        Args:
            messages: List of chat messages

        Returns:
            Tuple of (system_prompt, chat_messages)
        """
        system_messages = []
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_messages.append(msg.content)
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        # Combine system messages into a single system prompt
        system_prompt = " ".join(system_messages) if system_messages else None

        return system_prompt, chat_messages

    async def generate_response(self, req: ChatCompletionRequest) -> dict[str, Any]:
        """
        Generate a chat completion response using Claude API.

        Args:
            req: Chat completion request

        Returns:
            Chat completion response in OpenAI format

        Raises:
            HTTPException: If service is not configured or API call fails
        """
        if not self.client:
            raise HTTPException(
                status_code=500,
                detail="Claude is not configured. Please set CLAUDE_API_KEY environment variable.",
            )

        try:
            logger.info(
                f"Creating chat completion with Claude using model: {req.model}"
            )

            # Prepare messages for Claude API
            system_prompt, chat_messages = self._prepare_messages(req.messages)

            # Prepare the request parameters
            request_params = {
                "model": self._get_claude_model_name(req.model),
                "max_tokens": req.max_tokens or 1000,
                "temperature": req.temperature or 0.7,
                "messages": chat_messages,
            }

            # Add system prompt if available
            if system_prompt:
                request_params["system"] = system_prompt

            # Call Claude API
            response = self.client.messages.create(**request_params)

            # Extract text content from response
            content = ""
            if hasattr(response, "content") and response.content:
                # Handle list of content blocks
                if isinstance(response.content, list) and len(response.content) > 0:
                    content = response.content[0].text
                else:
                    content = str(response.content)

            # Convert response to OpenAI format
            result = {
                "id": response.id,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }

            logger.info("Claude chat completion successful")
            return result

        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Claude API error: {str(e)}"
            ) from e

    async def generate_streaming_response(
        self, req: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming chat completion response using Claude API.

        Args:
            req: Chat completion request

        Yields:
            Server-sent events in OpenAI streaming format

        Raises:
            HTTPException: If service is not configured or API call fails
        """
        if not self.client:
            raise HTTPException(
                status_code=500,
                detail="Claude is not configured. Please set CLAUDE_API_KEY environment variable.",
            )

        try:
            logger.info(
                f"Creating streaming chat completion with Claude using model: {req.model}"
            )

            # Prepare messages for Claude API
            system_prompt, chat_messages = self._prepare_messages(req.messages)

            # Prepare the request parameters
            request_params = {
                "model": self._get_claude_model_name(req.model),
                "max_tokens": req.max_tokens or 1000,
                "temperature": req.temperature or 0.7,
                "messages": chat_messages,
            }

            # Add system prompt if available
            if system_prompt:
                request_params["system"] = system_prompt

            # Use Claude's streaming API
            with self.client.messages.stream(**request_params) as stream:
                # Send initial chunk with role
                chunk = {
                    "id": f"msg-claude-stream-{hash(str(chat_messages))}",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                # Process stream events
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event, "delta"):
                                # Try to extract text content from delta
                                content = ""
                                if hasattr(event.delta, "text"):
                                    content = getattr(event.delta, "text", "")
                                elif hasattr(event.delta, "content"):
                                    content = getattr(event.delta, "content", "")

                                if content:
                                    chunk = {
                                        "id": f"msg-claude-stream-{hash(str(chat_messages))}",
                                        "object": "chat.completion.chunk",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": str(content)},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                        elif event.type == "message_stop":
                            # Send final chunk
                            chunk = {
                                "id": f"msg-claude-stream-{hash(str(chat_messages))}",
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            break

            # Send done signal
            yield "data: [DONE]\n\n"
            logger.info("Claude streaming completion successful")

        except Exception as e:
            logger.error(f"Claude streaming error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Claude streaming error: {str(e)}"
            ) from e


# Global provider instance
claude_provider = ClaudeProvider()
