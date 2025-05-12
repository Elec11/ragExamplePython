import asyncio
from vector_database import VectorDatabase, ChromaDB

async def main():
    db_config = {
        "host": "localhost",
        "port": 3306,
        "user": "root",                    # <-- Change this
        "password": "notSecureChangeMe",   # <-- Change this
        "database": "city_issues"          # <-- Change this  
    }
    embedding_model_name = "all-MiniLM-L6-v2"  # Small, fast, and good general-purpose model
    persistent_storage_path = "chroma_db"  # Folder to store ChromaDB data
    collection_name = "city_issues"  # Collection name in ChromaDB

    db = ChromaDB(collection_name=collection_name, persistent_storage_path=persistent_storage_path)

    await db.index_documents()
    results = await db.search_documents(query="example query")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
