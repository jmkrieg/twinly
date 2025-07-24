"""
Memory service using Qdrant vector database for conversation memory.
"""

import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Literal
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from app.services.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.services.embeddings.azure_openai import azure_embedding_provider
from app.services.llms.azure_openai_provider import azure_openai_provider
from app.models.chat import ChatCompletionRequest, Message
from app.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Conversation memory service using Qdrant vector database.
    
    This service handles storing and retrieving conversation context using
    semantic similarity search with OpenAI embeddings.
    """
    
    def __init__(self, qdrant_url: Optional[str] = None, collection_name: str = "conversation_memory", 
                 embedding_provider: Optional[BaseEmbeddingProvider] = None):
        """
        Initialize the conversation memory service.
        
        Args:
            qdrant_url: URL to Qdrant instance (defaults to in-memory for development)
            collection_name: Name of the Qdrant collection
            embedding_provider: Embedding provider instance
        """
        self.collection_name = collection_name
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536  # Dimension for text-embedding-3-small
        
        # Initialize Qdrant client
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        else:
            # Use in-memory database for development
            self.client = QdrantClient(url="http://localhost:6333")
        
        # Initialize embedding provider with dependency injection
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
            logger.info("Using provided embedding provider")
        else:
            # Default to global azure embedding provider
            self.embedding_provider = azure_embedding_provider
            logger.info("Using default Azure embedding provider")
        
        # Initialize LLM provider for chat completions
        self.llm_provider = azure_openai_provider
        
        self._ensure_collection_exists()
        
    def _ensure_collection_exists(self):
        """Ensure the conversation memory collection exists."""
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Creating collection '{self.collection_name}'")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
    
    async def _create_embedding(self, text: str, memory_action: Optional[str] = None) -> List[float]:
        """
        Create embedding for the given text using the configured embedding provider.
        
        Args:
            text: Text to embed
            memory_action: Optional action context for embedding optimization
                         ("add" for storing new memories, "search" for queries, 
                          "update" for modifying existing memories)
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Check if embedding provider is available
            if not self.embedding_provider.is_available():
                logger.warning("Embedding provider not available. Generating hash-based embedding.")
                return self._generate_fallback_embedding(text)
            
            # Convert memory_action to proper Literal type if provided
            action: Optional[Literal["add", "search", "update"]] = None
            if memory_action in ["add", "search", "update"]:
                action = memory_action  # type: ignore
            
            return await self.embedding_provider.create_embedding(text, action)
                
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            # Generate hash-based fallback instead of zero vector
            return self._generate_fallback_embedding(text)
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """
        Generate a hash-based fallback embedding when the embedding service is not available.
        
        Args:
            text: Text to create fallback embedding for
            
        Returns:
            Hash-based embedding vector
        """
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to numeric values and normalize
        embedding = []
        for i in range(0, min(len(text_hash), self.embedding_dimension * 2), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(value)
        
        # Pad or truncate to match embedding dimension
        while len(embedding) < self.embedding_dimension:
            embedding.append(0.1)  # Small non-zero value
        
        return embedding[:self.embedding_dimension]
    
    async def store_conversation_turn(
        self, 
        user_id: str, 
        user_message: str, 
        assistant_response: str, 
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a conversation turn (user message + assistant response) in memory.
        
        Args:
            user_id: Unique identifier for the user
            user_message: The user's message
            assistant_response: The assistant's response
            session_id: Session identifier (optional)
            metadata: Additional metadata to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Store user message
            user_point_id = str(uuid.uuid4())
            user_embedding = await self._create_embedding(user_message, "add")
            
            user_payload = {
                "user_id": user_id,
                "text": user_message,
                "role": "user",
                "session_id": session_id or "default",
                "type": "conversation_turn"
            }
            
            if metadata:
                user_payload.update(metadata)
            
            # Store assistant response  
            assistant_point_id = str(uuid.uuid4())
            assistant_embedding = await self._create_embedding(assistant_response, "add")
            
            assistant_payload = {
                "user_id": user_id,
                "text": assistant_response,
                "role": "assistant", 
                "session_id": session_id or "default",
                "type": "conversation_turn"
            }
            
            if metadata:
                assistant_payload.update(metadata)
            
            # Upsert both points
            points = [
                PointStruct(
                    id=user_point_id,
                    vector=user_embedding,
                    payload=user_payload
                ),
                PointStruct(
                    id=assistant_point_id,
                    vector=assistant_embedding,
                    payload=assistant_payload
                )
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Stored conversation turn for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing conversation turn: {str(e)}")
            return False
    
    async def search_similar_facts(
        self,
        user_id: str,
        fact_text: str,
        category: str = "general",
        similarity_threshold: float = 0.85,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar existing facts to avoid storing duplicates.
        
        Args:
            user_id: Unique identifier for the user
            fact_text: The fact text to check for similarity
            category: Category of the fact to check within
            similarity_threshold: Minimum similarity score to consider as duplicate (0.0 to 1.0)
            limit: Maximum number of similar facts to return
            
        Returns:
            List of similar existing facts with their similarity scores
        """
        try:
            query_embedding = await self._create_embedding(fact_text, "search")
            logger.info(f"Searching for similar facts for user {user_id} in category '{category}'")
            
            # Search for similar vectors with filters for user and fact type
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                score_threshold=similarity_threshold,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="type", match=MatchValue(value="fact")),
                        FieldCondition(key="category", match=MatchValue(value=category))
                    ]
                )
            ).points
            
            logger.info(f"Found {len(search_results)} similar facts with threshold {similarity_threshold}")

            # Format results
            similar_facts = []
            for result in search_results:
                if result.payload:
                    fact = {
                        "text": result.payload.get("text", ""),
                        "category": result.payload.get("category", ""),
                        "session_id": result.payload.get("session_id", ""),
                        "score": result.score,
                        "id": result.id,
                        "metadata": {k: v for k, v in result.payload.items() 
                                  if k not in ["text", "category", "session_id", "user_id", "role", "type"]}
                    }
                    similar_facts.append(fact)
            
            return similar_facts
            
        except Exception as e:
            logger.error(f"Error searching for similar facts: {str(e)}")
            return []

    async def store_fact(
        self,
        user_id: str,
        fact_text: str,
        category: str = "general",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        check_duplicates: bool = True,
        similarity_threshold: float = 0.85
    ) -> bool:
        """
        Store a specific fact or piece of information about the user.
        
        Args:
            user_id: Unique identifier for the user
            fact_text: The fact or information to store
            category: Category of the fact (e.g., "preference", "personal", "context")
            session_id: Session identifier (optional)
            metadata: Additional metadata to store
            check_duplicates: Whether to check for similar existing facts before storing
            similarity_threshold: Similarity threshold for considering duplicates
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            if check_duplicates:
                similar_facts = await self.search_similar_facts(
                    user_id=user_id,
                    fact_text=fact_text,
                    category=category,
                    similarity_threshold=similarity_threshold
                )
                if similar_facts:
                    most_similar = similar_facts[0]
                    logger.info(f"Found {len(similar_facts)} similar facts for user {user_id} in category '{category}'. "
                              f"Most similar fact: '{most_similar['text'][:100]}...' "
                              f"(similarity: {most_similar['score']:.3f}). Skipping storage to avoid duplicate.")
                    return False
            
            point_id = str(uuid.uuid4())
            embedding = await self._create_embedding(fact_text, "add")
            
            payload = {
                "user_id": user_id,
                "text": fact_text,
                "role": "fact",
                "category": category,
                "session_id": session_id or "default",
                "type": "fact"
            }
            
            if metadata:
                payload.update(metadata)
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Stored new fact for user {user_id} in category '{category}': {fact_text[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error storing fact: {str(e)}")
            return False
    
    async def search_memory(
        self,
        user_id: str,
        query_text: str,
        limit: int = 5,
        threshold: float = 0.7,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories based on query text only, ignoring user_id and session_id filters.
        
        Args:
            user_id: Unique identifier for the user (kept for API compatibility but not used in search)
            query_text: Text to search for similar memories
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)
            session_id: Optional session filter (kept for API compatibility but not used in search)
            
        Returns:
            List of relevant memory entries with metadata
        """
        try:
            query_embedding = await self._create_embedding(query_text, "search")
            logger.info(f"Creating embedding with query: {query_text} (searching across all users/sessions)")
            
            # Search for similar vectors without any filters - search across all memories
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                score_threshold=threshold
            ).points
            logger.info(f"Search results in collection {self.collection_name} without filters: {len(search_results)} found")

            # Format results
            memories = []
            for result in search_results:
                if result.payload:
                    memory = {
                        "text": result.payload.get("text", ""),
                        "role": result.payload.get("role", ""),
                        "type": result.payload.get("type", ""),
                        "category": result.payload.get("category", ""),
                        "session_id": result.payload.get("session_id", ""),
                        "user_id": result.payload.get("user_id", ""),  # Include user_id in results for reference
                        "score": result.score,
                        "metadata": {k: v for k, v in result.payload.items() 
                                  if k not in ["text", "role", "type", "category", "session_id", "user_id"]}
                    }
                    memories.append(memory)
            
            logger.info(f"Found {len(memories)} relevant memories across all users/sessions")
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}")
            return []
    
    async def search_memory_with_compact_query(
        self,
        user_id: str,
        user_prompt: str,
        limit: int = 5,
        threshold: float = 0.7,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using a compact, optimized search query.
        
        Args:
            user_id: Unique identifier for the user (kept for API compatibility)
            user_prompt: Original user prompt to create search query from
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)
            session_id: Optional session filter (kept for API compatibility)
            
        Returns:
            List of relevant memory entries with metadata
        """
        try:
            # Create compact search query
            compact_query = await self.create_compact_search_query(user_prompt)
            logger.info(f"Using compact query: '{compact_query}' for memory search")
            
            # Use the existing search_memory method with the compact query
            return await self.search_memory(
                user_id=user_id,
                query_text=compact_query,
                limit=limit,
                threshold=threshold,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"Error in compact query memory search: {str(e)}")
            # Fallback to regular search with original prompt
            return await self.search_memory(
                user_id=user_id,
                query_text=user_prompt,
                limit=limit,
                threshold=threshold,
                session_id=session_id
            )

    def evaluate_memory_relevance(self, text: str, memories: List[Dict[str, Any]]) -> float:
        """
        Evaluate how relevant the memories are for the given text.
        
        Args:
            text: The text to evaluate against
            memories: List of memory entries
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not memories:
            return 0.0
            
        # Simple relevance based on average similarity scores
        total_score = sum(memory.get("score", 0.0) for memory in memories)
        avg_score = total_score / len(memories)
        
        # Normalize to 0-1 range (assuming cosine similarity)
        return min(1.0, max(0.0, avg_score))
    
    def is_available(self) -> bool:
        """Check if the memory service is available and working."""
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as e:
            logger.error(f"Memory service availability check failed: {str(e)}")
            return False

    async def extract_facts_from_conversation(self, prompt: str, response: str, goal: str) -> str:
        """
        Extract and summarize facts from USER'S message ONLY.
        Uses LLM to identify key information that the USER explicitly provided or confirmed.
        
        Args:
            prompt: User's message (primary source for fact extraction)
            response: Assistant response (used only for context, not fact extraction)
            goal: The determined goal
            
        Returns:
            Compact fact summary containing ONLY user-provided information
        """
        if not self.llm_provider.is_available():
            # Simple fallback - only store user's explicit statement
            return f"Goal: {goal}. User statement: {prompt[:100]}..."

        fact_extraction_messages = [
            Message(
                role="system",
                content=PromptManager.get_prompt("fact_extractor")
            ),
            Message(
                role="user",
                content=f"Goal: {goal}\n\nUser message to analyze: {prompt}\n\nContext (Assistant response - DO NOT extract facts from this): {response}"
            )
        ]

        try:
            request = ChatCompletionRequest(
                model="gpt4.1-chat",
                messages=fact_extraction_messages,
                temperature=0.2,
                max_tokens=200
            )

            response_data = await self.llm_provider.generate_response(request)
            fact_summary = response_data["choices"][0]["message"]["content"].strip()

            logger.info(f"Successfully extracted facts from conversation")
            return fact_summary

        except Exception as e:
            logger.warning(f"Fact extraction failed: {str(e)}")
            # Fallback - only use user's input, not assistant response
            return f"Goal: {goal}. User statement: {prompt[:100]}..."

    async def create_compact_search_query(self, user_prompt: str) -> str:
        """
        Create a compact, focused search query from the user's prompt.
        Extracts key terms and concepts to improve memory search precision.
        
        Args:
            user_prompt: The original user prompt
            
        Returns:
            Compact search query optimized for memory retrieval
        """
        if not self.llm_provider.is_available():
            # Simple fallback - extract key words and remove common words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "how", "what", "when", "where", "why", "who", "can", "could", "would", "should", "will", "is", "are", "was", "were", "do", "does", "did", "have", "has", "had"}
            words = user_prompt.lower().split()
            key_words = [word.strip(".,!?;:") for word in words if word.lower() not in stop_words and len(word) > 2]
            return " ".join(key_words[:8])  # Limit to 8 key words
        
        try:
            query_messages = [
                Message(
                    role="system",
                    content="""Extract the most important search terms from the user's prompt for memory retrieval. 
                    Focus on:
                    - Key topics and subjects
                    - Specific terms and concepts
                    - Action words and objectives
                    - Important context clues
                    
                    Return only the essential search terms as a short phrase (max 10 words).
                    Remove filler words, articles, and unnecessary details.
                    
                    Examples:
                    "How do I configure SSL certificates for my web server?" → "configure SSL certificates web server"
                    "What was that Python library we discussed for data analysis?" → "Python library data analysis"
                    "Can you help me debug this JavaScript async function issue?" → "debug JavaScript async function"
                    """
                ),
                Message(
                    role="user", 
                    content=user_prompt
                )
            ]
            
            request = ChatCompletionRequest(
                model="gpt4.1-chat",
                messages=query_messages,
                temperature=0.1,
                max_tokens=50  # Keep it very short
            )
            
            response = await self.llm_provider.generate_response(request)
            compact_query = response["choices"][0]["message"]["content"].strip()
            
            logger.info(f"Created compact search query: '{compact_query}' from prompt: '{user_prompt[:50]}...'")
            return compact_query
            
        except Exception as e:
            logger.warning(f"Compact query creation failed: {str(e)}")
            # Fallback to simple keyword extraction
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "how", "what", "when", "where", "why", "who", "can", "could", "would", "should", "will", "is", "are", "was", "were", "do", "does", "did", "have", "has", "had"}
            words = user_prompt.lower().split()
            key_words = [word.strip(".,!?;:") for word in words if word.lower() not in stop_words and len(word) > 2]
            return " ".join(key_words[:8])


def get_conversation_memory(collection_name: Optional[str] = None, embedding_provider: Optional[BaseEmbeddingProvider] = None) -> ConversationMemory:
    """
    Get or create a conversation memory instance for the specified collection.
    This function ensures lazy initialization to avoid importing issues.
    
    Args:
        collection_name: Optional collection name override
        embedding_provider: Optional embedding provider to use
    """
    global _conversation_memory_instances
    
    # Initialize global instances dict if not exists
    if '_conversation_memory_instances' not in globals() or _conversation_memory_instances is None:
        _conversation_memory_instances = {}
    
    # Determine collection name
    final_collection_name = (
        collection_name or 
        os.getenv("QDRANT_COLLECTION_NAME", "conversation_memory")
    )
    
    # Create instance if not exists for this collection
    if final_collection_name not in _conversation_memory_instances:
        _conversation_memory_instances[final_collection_name] = ConversationMemory(
            qdrant_url=os.getenv("QDRANT_URL") or None,
            collection_name=final_collection_name,
            embedding_provider=embedding_provider
        )
    
    return _conversation_memory_instances[final_collection_name]


def get_agent_conversation_memory(agent_name: str, embedding_provider: Optional[BaseEmbeddingProvider] = None) -> ConversationMemory:
    """
    Get conversation memory instance for a specific agent.
    
    Args:
        agent_name: Name of the agent (e.g., "agent-mode")
        embedding_provider: Optional embedding provider to use
        
    Returns:
        ConversationMemory instance configured for the specific agent
    """
    # Use agent name directly as collection name with sanitization
    sanitized_agent_name = agent_name.replace("-", "_").replace(" ", "_").lower()
    collection_name = f"memory_{sanitized_agent_name}"
    
    # Check for environment variable override first
    agent_env_map = {
        "agent-mode": "AGENT_MODE"
    }
    
    if agent_name in agent_env_map:
        env_suffix = agent_env_map[agent_name]
        env_collection_name = os.getenv(f"QDRANT_{env_suffix}_COLLECTION")
        if env_collection_name:
            collection_name = env_collection_name
    
    return get_conversation_memory(collection_name, embedding_provider)


# Global variable to hold multiple memory instances
_conversation_memory_instances: Optional[Dict[str, ConversationMemory]] = None
