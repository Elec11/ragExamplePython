import chromadb
from chromadb.api import Client as ChromaClient
from chromadb.config import Settings

class ChromaDB(VectorDatabase):
    def __init__(self, collection_name: str, persistent_storage_path: str):
        self.client = ChromaClient(Settings(persist_directory=persistent_storage_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    async def fetch_descriptions(self):
        # Implement fetching descriptions logic here
        pass

    async def index_documents(self):
        # Implement indexing documents logic here
        pass

    async def search_documents(self, query: str, top_k=5):
        results = self.collection.query(
            query_embeddings=[query],
            n_results=top_k
        )
        return results["documents"][0]
