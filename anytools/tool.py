import typing as tp
from abc import ABC, abstractmethod

from pydantic import BaseModel

from ._proxy import LazyProxy

T = tp.TypeVar("T")


class Tool(BaseModel, LazyProxy[T], ABC):
    """Base Tool class for all vendors tool framework implementations"""

    @classmethod
    def tool_definition(cls):
        return {
            "type": "function",
            "function": {
                "name": cls.__name__,
                "description": cls.__doc__ or "",
                "parameters": cls.model_json_schema().get("properties", {}),
            },
        }

    @abstractmethod
    def __load__(self) -> T:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> tp.AsyncGenerator[str, tp.Any]:
        raise NotImplementedError
