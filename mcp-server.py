import asyncio
from datetime import datetime
import numpy as np
import sys
import os
import pandas as pd
import pymysql
from sentence_transformers import SentenceTransformer
import chromadb
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# --- CONFIGURATION ---
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",                    # <-- Change this
    "password": "notSecureChangeMe",   # <-- Change this
    "database": "city_issues"          # <-- Change this  
}
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast, and good general-purpose model
PERSISTENT_STORAGE_PATH = "chroma_db"  # Folder to store ChromaDB data
COLLECTION_NAME = "city_issues"  # Collection name in ChromaDB

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize ChromaDB
client = chromadb.Client()
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    collection = client.get_collection(COLLECTION_NAME)
else:
    collection = client.create_collection(name=COLLECTION_NAME)

# Fetch descriptions from MariaDB
def fetch_descriptions():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        query = "SELECT id, description FROM issues;"
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Fetched {len(df)} rows from the database.")
        return df
    except Exception as e:
        print("Error while fetching data from MariaDB:", e)

# Embed and store in ChromaDB (only if empty)
def index_documents():
    df = fetch_descriptions()
    if len(collection.get(ids=[])["ids"]) == 0:
        print("Indexing documents...")
        embeddings = model.encode(df["description"].tolist(), show_progress_bar=True).astype(np.float64)  # Use np.float64
        collection.add(
            documents=df["description"].tolist(),
            metadatas=[{"id": int(row["id"])} for _, row in df.iterrows()],
            embeddings=embeddings.tolist(),
            ids=[str(row["id"]) for _, row in df.iterrows()]
        )
        print(f"Indexed {len(df)} documents into ChromaDB.")

# Query ChromaDB
def search_documents(query: str, top_k=5):
    query_emb = model.encode([query]).astype(np.float64)  # Use np.float64
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    return results

# --- MCP SERVER ---
server = Server("city-issues-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search-issues",
            description="Search city issues database with a semantic query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None):
    if name == "search-issues":
        query = arguments["query"]
        results = search_documents(query)
        output = []
        
        # Loop through results and append issue description with metadata
        for meta, score in zip(results["metadatas"][0], results["distances"][0]):
            issue_id = meta['id']
            description = next((desc for _, desc in fetch_descriptions().iterrows() if desc['id'] == issue_id), None)
            
            output.append(
                types.TextContent(
                    type="text",
                    text=f"(ID: {issue_id}, Score: {score:.4f})\nDescription: {description}"
                )
            )
        return output
    raise ValueError(f"Unknown tool: {name}")

async def main():
    # Index documents if needed
    index_documents()

    # Set binary mode for stdin/stdout on Windows
    if sys.platform == 'win32':
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    # Start the MCP server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="city-issues-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
