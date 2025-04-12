import typing as tp
from abc import ABC, abstractmethod

import typing_extensions as tpe
from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextDelta,
    ToolParam,
    ToolUseBlock,
    ToolChoiceParam,
)
from pydantic import Field

from .tool import Tool
from .utils import get_logger, get_random_int
from .rag import HttpTool

logger = get_logger(__name__)

AnthropicModels: tpe.TypeAlias = tp.Literal[
    "claude-3-7-sonnet-20250219",  # $3 • $15 / 8192-64000
    "claude-3-5-haiku-20241022",  # $0.8 • $4 / 8192
    "claude-3-haiku-20240307",  # $0.25 • $1.25 / 4096
]

logger = get_logger()


class AnthropicTool(Tool[AsyncAnthropic], ABC):
    """
    An abstract base class representing a tool that can be used in chat completions.

    This class combines functionality from Pydantic's BaseModel, LazyProxy, and ABC to create
    a flexible and extensible tool structure for use with groq's chat completion API.
    """

    def __load__(self):
        return AsyncAnthropic()

    @abstractmethod
    def run(self) -> tp.AsyncGenerator[str, None]: ...


class AnthropicAgent(AnthropicTool):
    model: AnthropicModels = Field(default="claude-3-7-sonnet-20250219")
    messages: list[MessageParam] = Field(default_factory=list)
    tools: list[ToolParam] = Field(default_factory=list)
    max_tokens: int = Field(default_factory=get_random_int)
    tool_choice: ToolChoiceParam = Field(
        default={"type": "any", "disable_parallel_tool_use": False}
    )

    async def run(self) -> tp.AsyncGenerator[str, None]:
        client = self.__load__()
        async with client.messages.stream(
            model=self.model,
            tools=self.tools,
            messages=self.messages,
            max_tokens=self.max_tokens,
        ) as response_stream:
            tool_classes = {
                t.__name__: t
                for t in AnthropicTool.__subclasses__() + HttpTool.__subclasses__()
            }
            async for raw_content_block in response_stream:
                if raw_content_block.type == "content_block_stop":
                    if isinstance(raw_content_block.content_block, ToolUseBlock):
                        logger.info(
                            "Executing tool %s", raw_content_block.content_block.name
                        )
                        content = ""
                        async for chunk in (
                            tool_classes[raw_content_block.content_block.name]
                            .model_validate(raw_content_block.content_block.input)
                            .run()
                        ):
                            yield chunk
                            content += chunk
                        self.messages.append({"role": "user", "content": content})
                        self.tool_choice["type"] = "none"  # type: ignore
                        async for chunk in self.run():
                            yield chunk
                        self.tool_choice["type"] = "any"  # type: ignore
                elif raw_content_block.type == "content_block_delta":
                    if isinstance(raw_content_block.delta, TextDelta):
                        if raw_content_block.delta.text:
                            yield raw_content_block.delta.text
                else:
                    continue
