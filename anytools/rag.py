import os
from typing import Any, Dict, List, Literal, Optional, AsyncGenerator, Union
from uuid import uuid4
import asyncio
from functools import lru_cache

import httpx
from mistralai import Mistral
from pydantic import Field, field_validator
from typing_extensions import NotRequired, Required, TypedDict, Unpack

from anytools.proxy import LazyProxy
from .tool import Tool
from .utils import get_logger

logger = get_logger(__name__)

# TypedDict definitions
class SparseValues(TypedDict, total=False):
    indices: Required[List[int]]
    values: Required[List[float]]

class Metadata(TypedDict, total=False):
    content: Required[str | list[str]]
    namespace: Required[str]

class Vector(TypedDict, total=False):
    id: Required[str]
    values: Required[List[float]]
    sparseValues: NotRequired[SparseValues]
    metadata: Required[Metadata]

class QueryMatch(TypedDict, total=False):
    id: Required[str]
    score: Required[float]
    metadata: Required[Metadata]
    values: NotRequired[List[float]]

class UpsertRequest(TypedDict, total=False):
    vectors: Required[List[Vector]]
    namespace: Required[str]

class UpsertResponse(TypedDict, total=False):
    upsertedCount: Required[int]

class QueryRequest(TypedDict, total=False):
    vector: Required[List[float]]
    namespace: Required[str]
    filter: NotRequired[Dict[str, Any]]
    topK: Required[int]
    includeMetadata: Required[bool]
    includeValues: NotRequired[bool]

class Usage(TypedDict):
    readUnits: Required[int]

class QueryResponse(TypedDict):
    matches: Required[List[QueryMatch]]
    namespace: Required[str]
    usage: Required[Usage]


class RagTool(Tool, LazyProxy[httpx.AsyncClient]):
    """
    [MUST USE]
    Provides Retrieval Augmented Generation capabilities to the model for semantic memory.
    Stores and retrieves content using vector embeddings for semantic similarity search.
    Use it to continuously store and retrieve relevant information from the conversation.
    """

    content: str = Field(
        ...,
        description="Text to be embedded and stored/retrieved during RAG.",
    )
    namespace: str = Field(
        ..., description="Namespace identifier for content organization."
    )
    topK: Optional[int] = Field(
        default=None,
        description="Number of top results to retrieve when querying (1-20).",
    )
    action: Literal["query", "upsert"] = Field(
        default="upsert",
        description="Action to perform: 'query' to search or 'upsert' to store.",
    )
    
    # Optional configurations with defaults
    similarity_threshold: float = Field(
        default=0.75,
        description="Minimum similarity score threshold for query results (0.0-1.0).",
        ge=0.0,
        le=1.0
    )
    
    # Cache for clients
    _mistral_client = None
    _httpx_client = None
    
    @field_validator("topK")
    @classmethod
    def validate_top_k(cls, v:Union[int, None], info:Any):
        if info.data.get("action") == "query" and (v is None or v < 1 or v > 20):
            raise ValueError("topK must be between 1 and 20 for query actions")
        return v
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v:str):
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()

    def __load__(self) -> httpx.AsyncClient:
        """Create or reuse cached httpx client with appropriate configuration"""
        if RagTool._httpx_client is None:
            RagTool._httpx_client = httpx.AsyncClient(
                base_url=os.environ.get("PINECONE_BASE_URL", "").rstrip("/"),
                headers={
                    "Content-Type": "application/json",
                    "Api-Key": os.environ.get("PINECONE_API_KEY", ""),
                },
                timeout=30.0,  # Increased timeout
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                http2=True,  # Use HTTP/2 for better performance
            )
        return RagTool._httpx_client
    
    @lru_cache(maxsize=1)
    def _get_mistral_client(self) -> Mistral:
        """Get or create cached Mistral client"""
        if RagTool._mistral_client is None:
            RagTool._mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", ""))
        return RagTool._mistral_client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text with error handling and retries"""
        client = self._get_mistral_client()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = await client.embeddings.create_async(
                    model="mistral-embed", 
                    inputs=text
                )
                embedding = response.data[0].embedding
                assert isinstance(embedding, list)
                return embedding
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate embedding after {max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Embedding attempt {attempt+1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
        return []
    
    async def upsert(self, **kwargs: Unpack[UpsertRequest]) -> UpsertResponse:
        """Upsert vectors with improved error handling"""
        client = self.__load__()
        try:
            response = await client.post("/vector/upsert", json=kwargs)
            response.raise_for_status()
            return UpsertResponse(**response.json())
        except httpx.HTTPStatusError as e:
            logger.error(f"Upsert HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Upsert request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Upsert unexpected error: {str(e)}")
            raise

    async def query(self, **kwargs: Unpack[QueryRequest]) -> QueryResponse:
        """Query vectors with improved error handling"""
        client = self.__load__()
        try:
            response = await client.post("/query", json=kwargs)
            response.raise_for_status()
            return QueryResponse(**response.json())
        except httpx.HTTPStatusError as e:
            logger.error(f"Query HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Query request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Query unexpected error: {str(e)}")
            raise

    async def run(self) -> AsyncGenerator[str, None]:
        """Execute RAG operation with better output formatting and error handling"""
        try:
            if self.action == "upsert":
                # Generate embedding and create vector
                embedding = await self.embed(self.content)
                
                upsert_request = UpsertRequest(
                    vectors=[
                        {
                            "id": str(uuid4()),
                            "values": embedding,
                            "metadata": {
                                "content": self.content,
                                "namespace": self.namespace,
                            },
                        }
                    ],
                    namespace=self.namespace,
                )
                
                response = await self.upsert(**upsert_request)
                yield f"✅ Successfully stored {response['upsertedCount']} embedding in '{self.namespace}'.\n"
                
            elif self.action == "query":
                # Validate topK before proceeding
                if not isinstance(self.topK, int) or self.topK < 1:
                    raise ValueError("topK must be a positive integer for query operations")
                
                embedding = await self.embed(self.content)
                
                query_request = QueryRequest(
                    vector=embedding,
                    namespace=self.namespace,
                    topK=self.topK,
                    includeMetadata=True,
                    includeValues=False,  # Save bandwidth by not returning vectors
                )
                
                response = await self.query(**query_request)
                
                # Filter by similarity threshold
                relevant_matches = [m for m in response["matches"] if m["score"] >= self.similarity_threshold]
                
                if not relevant_matches:
                    yield f"No relevant matches found in '{self.namespace}' above threshold {self.similarity_threshold}.\n"
                else:
                    yield f"Found {len(relevant_matches)} relevant matches in '{self.namespace}':\n\n"
                    
                    for i, match in enumerate(relevant_matches, 1):
                        content = match["metadata"]["content"]
                        # Format as a readable numbered list with scores
                        yield f"{i}. [Score: {match['score']:.4f}] {content}\n"
            else:
                raise ValueError(f"Invalid action '{self.action}'. Please choose 'query' or 'upsert'.")
                
        except Exception as e:
            error_msg = f"Error during {self.action} operation: {str(e)}"
            logger.error(error_msg)
            yield f"❌ {error_msg}"
            
    @classmethod
    async def close_clients(cls):
        """Properly close clients when shutting down"""
        if cls._httpx_client is not None:
            await cls._httpx_client.aclose()
            cls._httpx_client = None