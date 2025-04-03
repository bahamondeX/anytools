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

logger = get_logger(__name__)

AnthropicModels: tpe.TypeAlias = tp.Literal[
    "claude-3-7-sonnet-20250219",  # $75 • $150
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
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

        # Use iterative approach instead of recursive
        processing_stack = [True]  # Start with processing the main request
        # Keep a copy of the original tools
        tool_classes = {cls.__name__: cls for cls in AnthropicTool.__subclasses__()}
        while processing_stack:
            processing_stack.pop()  # Process the current item

            async with client.messages.stream(
                model=self.model,
                tools=self.tools,
                tool_choice=self.tool_choice,
                messages=self.messages,
                max_tokens=self.max_tokens,
            ) as response_stream:

                async for raw_content_block in response_stream:
                    if raw_content_block.type == "content_block_stop":
                        if isinstance(raw_content_block.content_block, ToolUseBlock):
                            logger.info(
                                "Executing tool %s",
                                raw_content_block.content_block.name,
                            )
                            # Use list to collect chunks for efficiency
                            content_chunks: list[str] = []
                            tool_name = raw_content_block.content_block.name
                            tool_input = raw_content_block.content_block.input

                            async for chunk in (
                                tool_classes[tool_name].model_validate(tool_input).run()
                            ):
                                yield chunk
                                content_chunks.append(chunk)

                            content = "".join(content_chunks)
                            self.messages.append({"role": "user", "content": content})
                            self.tool_choice = {"type": "none"}
                            # Add to processing stack instead of recursion
                            processing_stack.append(True)
                            break  # Break from current stream to handle the new item in stack

                    elif raw_content_block.type == "content_block_delta":
                        if (
                            isinstance(raw_content_block.delta, TextDelta)
                            and raw_content_block.delta.text
                        ):
                            yield raw_content_block.delta.text
