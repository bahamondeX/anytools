import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import httpx
from typing_extensions import NotRequired, Required, TypedDict, Unpack

from ._proxy import LazyProxy


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


@dataclass
class Pinecone(LazyProxy[httpx.AsyncClient]):
    headers: dict[str, str] = field(
        default_factory=lambda: {
            "Content-Type": "application/json",
            "Api-Key": os.environ.get("PINECONE_API_KEY"),
        }
    )  # type: ignore
    base_url: str = field(default=os.environ.get("PINECONE_BASE_URL"))  # type: ignore

    def __load__(self):
        return httpx.AsyncClient(base_url=self.base_url, headers=self.headers)

    async def upsert(self, **kwargs: Unpack[UpsertRequest]) -> UpsertResponse:
        client = self.__load__()
        response = await client.post(
            f"{self.base_url}/vector/upsert", json=kwargs, headers=self.headers
        )
        response.raise_for_status()
        return UpsertResponse(**response.json())

    async def query(self, **kwargs: Unpack[QueryRequest]) -> QueryResponse:
        client = self.__load__()
        response = await client.post(
            f"{self.base_url}/vector/query", json=kwargs, headers=self.headers
        )
        response.raise_for_status()
        return QueryResponse(**response.json())
