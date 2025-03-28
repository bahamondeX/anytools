import os
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import httpx
from typing_extensions import NotRequired, Required, TypedDict, Unpack

from .tool import Tool
from pydantic import Field
from mistralai import Mistral


class SparseValues(TypedDict, total=False):
    indices: Required[List[int]]
    values: Required[List[float]]


class Metadata(TypedDict, total=False):
    # Assuming Record is a dictionary with string keys and any type of values
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


class Usage(TypedDict):
    readUnits: Required[int]


class QueryResponse(TypedDict):
    matches: Required[List[QueryMatch]]
    namespace: Required[str]
    usage: Required[Usage]


class RagTool(Tool[httpx.AsyncClient]):
    """
    [MUST USE]
    Provides Retrieval Augmented Generation Capabilities to the model for infinite semantic memory. It upserts and queries with embeddings a namespaced vector database for semanticlly compatible content via cosine similarity. Use it to store the content from the user's conversation constinouslly.
    """
    content:str = Field(...,description="Text embedding to be `upserted`/`queried` during the Retrieval Augmented Generation process.")
    namespace:str = Field(...,description="Namespace of the tenant in which the content is allocated.")
    topK:Optional[int] = Field(default=None, description="Top K results to be retrieved in case the action is `query`, otherwise it's set to `None`.")
    action:Literal["query","upsert"] = Field(default="upsert", description="The action to perform can be either `query` or `upsert`.")    

    def __load__(self):
        return httpx.AsyncClient(base_url=os.environ["PINECONE_BASE_URL"], headers={
            "Content-Type": "application/json",
            "Api-Key": os.environ["PINECONE_API_KEY"],
        })


    async def embed(self):
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = await client.embeddings.create_async(
            model="mistral-embed", inputs=self.content
        )
        embedding = response.data[0].embedding
        assert isinstance(embedding,list)
        return embedding
    
    async def upsert(self, **kwargs: Unpack[UpsertRequest]) -> UpsertResponse:
        client = self.__load__()
        response = await client.post(
            f"{self.base_url}/vectors/upsert", json=kwargs
        )
        response.raise_for_status()
        return UpsertResponse(**response.json())

    async def query(self, **kwargs: Unpack[QueryRequest]) -> QueryResponse:
        client = self.__load__()
        response = await client.post(
            f"{self.base_url}/query", json=kwargs
        )
        response.raise_for_status()
        return QueryResponse(**response.json())


    async def run(self):
        if self.action == "upsert":
            upsert_request = UpsertRequest(vectors=[{"id": str(uuid4()), "values":await self.embed(), "metadata": {"content": self.content, "namespace": self.namespace}}], namespace=self.namespace)
            response = await self.upsert(**upsert_request)
            yield f"Upserted {response['upsertedCount']} embedding.\n\n"
        elif self.action == "query":
            assert isinstance(self.topK, int)
            query_request = QueryRequest(vector=await self.embed(), namespace=self.namespace, topK=self.topK, includeMetadata=True)
            response = await self.query(**query_request)
            for match in response["matches"]:
                yield f"Score: {match['score']}, Content: {match['metadata']['content']}\n\n"
        else:
            raise ValueError("Invalid action. Please choose 'query' or 'upsert'.")