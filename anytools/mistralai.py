import sys

try:
    import mistralai  # type: ignore
except ImportError:
    print(
        "Please install the MistralAI library first to use it as tool orchestrator: `pip install mistralai`"
    )
    sys.exit(0)
################################################################
import os
import uuid
from abc import ABC, abstractmethod
from functools import cached_property
from typing import AsyncGenerator, Literal

from mistralai import MessagesTypedDict, Mistral, ToolTypedDict
from pydantic import Field
from typing_extensions import TypeAlias

from .rag import Pinecone
from .tool import Tool

MistralModels: TypeAlias = Literal[
    "mistral-large-latest", "pixtral-large-latest", "codestral-latest"
]


class MistralTool(Tool[Mistral], ABC):
    def __load__(self):
        return Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    @abstractmethod
    def run(self) -> AsyncGenerator[str, None]:
        raise NotImplementedError


class MistralAgent(MistralTool):
    model: MistralModels = Field(default="mistral-large-latest")
    messages: list[MessagesTypedDict] = Field(default_factory=list)
    tools: list[ToolTypedDict] = Field(default_factory=list)
    response_format: Literal["text", "json_object"] = Field(default="text")
    max_tokens: int = Field(default=32378)
    namespace: str = Field(...)

    @cached_property
    def db(self) -> Pinecone:
        return Pinecone()

    async def embed(self, inputs: str, client: Mistral):
        response = await client.embeddings.create_async(
            model="mistral-embed", inputs=inputs
        )
        data = response.data[0].embedding
        assert isinstance(data, list)
        return data

    async def run(self):
        last_content = self.messages[-1]["content"]  # type: ignore
        assert isinstance(last_content, str)
        client = self.__load__()
        vector = await self.embed(last_content, client)
        query_results = await self.db.query(
            vector=vector,
            namespace=self.namespace,
            topK=5,
            includeMetadata=True,
        )
        context = "# Results from knowledge base:" + "\n".join(
            f"Score: {q['score']}\nContent: {q['metadata']['content']}"
            for q in query_results["matches"]
        )
        messages: list[MessagesTypedDict] = [{"role": "system", "content": context}]
        messages.extend(self.messages)
        response = await client.chat.stream_async(
            model=self.model,
            messages=messages,
            tools=self.tools,
            response_format={"type": self.response_format},
            max_tokens=self.max_tokens,
        )
        string = ""
        async for chunk in response:
            content = chunk.data.choices[0].delta.content
            if isinstance(content, str):
                yield content
                string += content
        vector = await self.embed(string, client)
        await self.db.upsert(
            vectors=[
                {
                    "values": vector,
                    "id": str(uuid.uuid4()),
                    "metadata": {"content": string, "namespace": self.namespace},
                }
            ],
            namespace=self.namespace,
        )
