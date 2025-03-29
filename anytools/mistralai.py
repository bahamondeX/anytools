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
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Literal

from mistralai import MessagesTypedDict, Mistral, ToolTypedDict
from pydantic import Field
from typing_extensions import TypeAlias

from .proxy import LazyProxy
from .tool import Tool

MistralModels: TypeAlias = Literal[
    "mistral-large-latest", "pixtral-large-latest", "codestral-latest"
]


class MistralTool(Tool, LazyProxy[Mistral], ABC):
    def __load__(self):
        return Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    @abstractmethod
    def run(self) -> AsyncGenerator[str, None]:
        raise NotImplementedError


class MistralAgent(MistralTool):
    model: MistralModels = Field(default="mistral-large-latest")
    messages: list[MessagesTypedDict] = Field(default_factory=list)
    tools: list[ToolTypedDict] = Field(default_factory=lambda: [t.tool_definition() for t in Tool.__subclasses__()])  # type: ignore
    response_format: Literal["text", "json_object"] = Field(default="text")
    max_tokens: int = Field(default=32378)
    namespace: str = Field(...)

    async def run(self):
        client = self.__load__()
        response = await client.chat.stream_async(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            response_format={"type": self.response_format},
            max_tokens=self.max_tokens,
        )
        async for chunk in response:
            content = chunk.data.choices[0].delta.content
            if isinstance(content, str):
                yield content
