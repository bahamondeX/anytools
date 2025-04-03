from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar, cast, Any

from typing_extensions import override

T = TypeVar("T")


class LazyProxy(Generic[T], ABC):
    """Implements data methods to pretend that an instance is another instance.

    This includes forwarding attribute access and other methods.
    """

    # Add the missing class_vars attribute for Pydantic compatibility
    __class_vars__ = set()

    # Note: we have to special case proxies that themselves return proxies
    # to support using a proxy as a catch-all for any random access, e.g. `proxy.foo.bar.baz`

    def __getattr__(self, attr: str) -> object:
        proxied = self.__get_proxied__()
        if isinstance(proxied, LazyProxy):
            return getattr(proxied, attr)
        try:
            return getattr(proxied, attr)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attr}'"
            )

    @override
    def __repr__(self) -> str:
        proxied = self.__get_proxied__()
        if isinstance(proxied, LazyProxy):
            return proxied.__class__.__name__
        return repr(proxied)

    @override
    def __str__(self) -> str:
        proxied = self.__get_proxied__()
        if isinstance(proxied, LazyProxy):
            return proxied.__class__.__name__
        return str(proxied)

    @override
    def __dir__(self) -> Iterable[str]:
        proxied = self.__get_proxied__()
        if isinstance(proxied, LazyProxy):
            return dir(proxied)
        return proxied.__dir__()

    @property  # type: ignore
    @override
    def __class__(self) -> type:  # pyright: ignore
        proxied = self.__get_proxied__()
        if issubclass(type(proxied), LazyProxy):
            return type(proxied)
        return proxied.__class__

    def __get_proxied__(self) -> T:
        return self.__load__()

    def __as_proxied__(self) -> T:
        """Helper method that returns the current proxy, typed as the loaded object"""
        return cast(T, self)

    @abstractmethod
    def __load__(self) -> T: ...

    # Add these methods to handle Pydantic's attribute access
    def __getattribute__(self, name: str) -> Any:
        # Special case for Pydantic's attribute checks
        if name in ("__class_vars__", "__class__", "__get_proxied__", "__load__"):
            return super().__getattribute__(name)
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self.__getattr__(name)

    def _setattr_handler(self, name: str, value: Any) -> Any:
        """Handle attribute setting for Pydantic compatibility"""
        return None  # Let the default __setattr__ handle it

    def __setattr__(self, name: str, value: Any) -> None:
        """Custom __setattr__ to handle both proxy and direct attribute setting"""
        try:
            # Try to set attribute on the proxied object first
            proxied = self.__get_proxied__()
            if not isinstance(proxied, LazyProxy):
                setattr(proxied, name, value)
            else:
                super().__setattr__(name, value)
        except (AttributeError, TypeError):
            # Fall back to setting on self
            super().__setattr__(name, value)
