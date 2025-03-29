import typing as tp
from abc import ABC, abstractmethod

from mistralai import ToolTypedDict
from pydantic import BaseModel

T = tp.TypeVar("T")


class Tool(BaseModel, ABC):
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

    @abstractmethod
    def run(self) -> tp.AsyncGenerator[str, tp.Any]:
        raise NotImplementedError
