import typing as tp
from abc import ABC, abstractmethod
from hashlib import md5

from mistralai import ToolTypedDict
from anthropic.types import ToolParam
from pydantic import BaseModel
from .proxy import LazyProxy

T = tp.TypeVar("T")


class Tool(BaseModel, LazyProxy[T], ABC):
    """Base Tool class for all vendors tool framework implementations"""

    @classmethod
    def tool_definition(cls) -> ToolTypedDict:
        return {
            "type": "function",
            "function": {
                "name": cls.__name__,
                "description": cls.__doc__ or "",
                "parameters": cls.model_json_schema().get("properties", {}),
            },
        }

    @classmethod
    def tool_param(cls) -> ToolParam:
        return ToolParam(
            input_schema=cls.model_json_schema(),
            name=cls.__name__,
            description=cls.__doc__ or "",
            cache_control={"type": "ephemeral"},
        )

    @abstractmethod
    def run(self) -> tp.AsyncGenerator[str, tp.Any]:
        raise NotImplementedError

    @abstractmethod
    def __load__(self) -> T:
        raise NotImplementedError

    def __hash__(self):
        return int(md5(self.model_dump_json().encode()).hexdigest(), 16) % (2**32)
