import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "upstage"

def get_client():
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT secrets in .env")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def init_db():
    client = get_client()
    
    if not client.collection_exists(COLLECTION_NAME):
        print(f"⚙️ Creating collection: {COLLECTION_NAME}...")
        
        # standard single vector config
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "visual": models.VectorParams(size=512, distance=models.Distance.COSINE),
                "audio": models.VectorParams(size=512, distance=models.Distance.COSINE),
            }
        )
        print("✅ Schema initialized.")
    else:
        print(f"Collection '{COLLECTION_NAME}' exists.")

    # create filtering indexes (idempotent)
    print("Ensuring payload indexes...")
    for field in ["role", "location", "tags"]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    print("✅ Indexes ready.")

if __name__ == "__main__":
    init_db()