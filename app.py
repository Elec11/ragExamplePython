import pymysql
import pandas as pd
import chromadb
import os
from sentence_transformers import SentenceTransformer
import traceback

# --- CONFIGURE THIS ---
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",           # <-- Change this
    "password": "notSecureChangeMe",   # <-- Change this
    "database": "city_issues"
}
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast, and good general-purpose model
PERSISTENT_STORAGE_PATH = "chroma_db"  # Folder to store ChromaDB data

# --- STEP 1: Fetch descriptions from MariaDB ---
def fetch_descriptions():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        query = "SELECT id, description FROM issues;"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print("Error while fetching data from MariaDB:", e)
        traceback.print_exc()

# --- STEP 2: Embed text using SentenceTransformer ---
def embed_descriptions(texts, model):
    try:
        return model.encode(texts, show_progress_bar=True).astype("float32")
    except Exception as e:
        print("Error while embedding descriptions:", e)
        traceback.print_exc()

# --- STEP 3: Initialize ChromaDB with persistent storage and add embeddings ---
def build_chroma_index(embeddings, df):
    try:
        print("📦 Creating a new collection...")

        # Ensure persistent storage path exists
        if not os.path.exists(PERSISTENT_STORAGE_PATH):
            os.makedirs(PERSISTENT_STORAGE_PATH)
            print(f"✅ Created persistent storage path: {PERSISTENT_STORAGE_PATH}")

        # Create or connect to a ChromaDB client (ChromaDB automatically persists its data to disk)
        client = chromadb.Client()
        collections = client.list_collections()  # List all collections
        print(f"Available collections: {collections}")

        collection_name = "city_issues"
        if collection_name in collections:
            collection = client.get_collection(collection_name)
            print(f"🔄 Loaded existing collection: {collection_name}")
        else:
            print(f"📦 Creating a new collection: {collection_name}")
            collection = client.create_collection(name=collection_name)

            # Add embeddings and metadata (IDs and descriptions)
            print("Adding documents to the collection...")
            collection.add(
                documents=df["description"].tolist(),
                metadatas=[{"id": row["id"]} for _, row in df.iterrows()],
                embeddings=embeddings.tolist(),
                ids=[str(row["id"]) for _, row in df.iterrows()]  # Add explicit IDs
            )
            print(f"✅ Added {len(df)} documents to collection: {collection_name}")

        return collection

    except Exception as e:
        print("Error in building ChromaDB index:", e)
        traceback.print_exc()

# --- STEP 4: Search with semantic query ---
def search_chroma_index(collection, query, model, top_k=5):
    try:
        query_emb = model.encode([query]).astype("float32")
        results = collection.query(
            query_embeddings=query_emb,
            n_results=top_k
        )
        return results
    except Exception as e:
        print("Error during search:", e)
        traceback.print_exc()

# --- MAIN LOGIC ---
def main():
    print("🔌 Connecting to MariaDB...")
    df = fetch_descriptions()
    print(f"✅ Loaded {len(df)} descriptions")

    print("🧠 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("⚙️ Generating embeddings...")
    embeddings = embed_descriptions(df["description"].tolist(), model)

    print("📦 Building ChromaDB index...")
    collection = build_chroma_index(embeddings, df)

    # Semantic search loop
    while True:
        query = input("\n🔍 Enter a search term (e.g., 'power outage') or type 'exit': ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        
        print("Searching...")
        results = search_chroma_index(collection, query, model)

        print("\n📝 Top matches:")
        for idx, score in zip(results["metadatas"][0], results["distances"][0]):
            print(f"- (ID: {idx['id']}) Score: {score:.4f}")
            print(f"  {df[df['id'] == int(idx['id'])]['description'].values[0]}")
        print()

if __name__ == "__main__":
    main()