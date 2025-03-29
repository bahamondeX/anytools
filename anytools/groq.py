import sys

try:
    import groq  # type: ignore
except ImportError:
    print(
        "Please install the Groq library first to use it as tool orchestrator: `pip install groq`"
    )
    sys.exit(0)
################################################################

import typing as tp
from abc import ABC

import typing_extensions as tpe
from groq import AsyncGroq
from groq.types.chat.chat_completion_message_param import \
    ChatCompletionMessageParam
from groq.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import Field

from .proxy import LazyProxy
from .tool import Tool

GroqModels: tpe.TypeAlias = tp.Literal[
    "llama-3.2-90b-vision-preview",  # $75 • $150
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
]


class GroqTool(Tool, LazyProxy[AsyncGroq], ABC):
    """
    An abstract base class representing a tool that can be used in chat completions.

    This class combines functionality from Pydantic's BaseModel, LazyProxy, and ABC to create
    a flexible and extensible tool structure for use with groq's chat completion API.
    """

    def __load__(self):
        """
        Initializes and returns an Asyncgroq client with predefined base URL and API key.

        This method is used by the LazyProxy to create the groq client when needed.

        Returns:
                Asyncgroq: An initialized Asyncgroq client.
        """
        return AsyncGroq()


class GroqAgent(GroqTool):
    """
    A class representing an AI agent capable of executing chat completions.

    This agent uses a specified language model to generate responses based on
    input messages and can execute various tools.

    Attributes:
        model (GroqModels): The language model to be used for chat completion.
            Defaults to "llama-3.3-70b-versatile".
        messages (list[ChatCompletionMessageParam]): A list of messages that form
            the conversation history for the agent. Defaults to an empty list.
        tools (list[ChatCompletionToolParam]): A list of tools that can be used
            during the chat completion. Defaults to an empty list.
    """

    model: GroqModels = Field(default="llama-3.3-70b-versatile")
    messages: list[ChatCompletionMessageParam] = Field(default_factory=list)
    tools: list[ChatCompletionToolParam] = Field(default_factory=list)

    async def run(self) -> tp.AsyncGenerator[str, None]:
        """
        Executes the chat completion process using the groq API.

        Yields:
            AsyncGenerator[str, None]: An asynchronous generator that yields the content of the chat completion.

        Raises:
            ValueError: If no content is found or if no tool calls are found.
        """
        client = self.__load__()
        response_stream = await client.chat.completions.create(
            model=self.model,
            tools=self.tools,
            tool_choice="auto",
            messages=self.messages,
            stream=True,
        )

        async for response in response_stream:
            tool_calls = response.choices[0].delta.tool_calls
            if not tool_calls:
                content = response.choices[0].delta.content
                if content:
                    yield content
                else:
                    raise ValueError("No content found in the response.")
            else:
                for tool_call in tool_calls:
                    function = tool_call.function
                    if function is None:
                        raise ValueError("No function was called in the tool call.")

                    for tool in self.tools:
                        if tool["function"]["name"] == function.name:
                            if function.arguments is None:
                                raise ValueError("Function arguments were not found.")

                            tool_instance = self.model_validate_json(function.arguments)
                            async for chunk in tool_instance.run():
                                yield chunk
                raise ValueError("No matching tool calls found.")
