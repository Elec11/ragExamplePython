import os

def list_databases():
    # Assuming databases are stored in a specific directory
    db_dir = "databases"
    if not os.path.exists(db_dir):
        return []
    
    databases = [f for f in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, f))]
    return databases

def select_database():
    databases = list_databases()
    print("Available Databases:")
    for i, db in enumerate(databases, 1):
        print(f"{i}. {db}")
    
    choice = input("Select a database number (or type 'new' to create a new one): ")
    
    if choice.isdigit() and int(choice) <= len(databases):
        return databases[int(choice) - 1]
    elif choice.lower() == "new":
        new_db_name = input("Enter the name for the new database: ")
        os.makedirs(os.path.join("databases", new_db_name), exist_ok=True)
        return new_db_name
    else:
        print("Invalid selection. Please try again.")
        return select_database()

def main():
    db_name = select_database()
    
    DB_CONFIG = {
        "host": "localhost",
        "port": 3306,
        "user": "root",           # <-- Change this
        "password": "notSecureChangeMe",   # <-- Change this
        "database": db_name
    }
    
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast, and good general-purpose model
    PERSISTENT_STORAGE_PATH = "chroma_db"  # Folder to store ChromaDB data
    
    # Rest of the main function...
