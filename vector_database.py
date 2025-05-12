from abc import ABC, abstractmethod

class VectorDatabase(ABC):
    @abstractmethod
    async def fetch_descriptions(self):
        pass

    @abstractmethod
    async def index_documents(self):
        pass

    @abstractmethod
    async def search_documents(self, query: str, top_k=5):
        pass
