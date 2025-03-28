from httpx import Client

from anytools._proxy import LazyProxy


class MyLazyProxy(LazyProxy[Client]):
    def __load__(self) -> Client:
        return Client()
