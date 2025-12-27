import uuid
from qdrant_client.http import models
from src.db import get_client, COLLECTION_NAME
from src.embedders import VisualEngine

# load engine once
engine = VisualEngine()
client = get_client()

def ingest_creative(user_data, image_path):
    """
    user_data: dict
    image_path: str
    """
    print(f"Processing {user_data.get('name')}...")
    
    # 1. vectorize image
    vector = engine.get_image_vector(image_path)
    
    if not vector:
        print(f"⚠️ Skipping {user_data['name']} - Image not found or invalid.")
        return

    # 2. prepare payload 
    payload = user_data
    point_id = str(uuid.uuid4())

    # 3. upload to qdrant   
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector, # standard list[float]
                payload=payload
            )
        ]
    )
    print(f"✅ Indexed {user_data['name']} (ID: {point_id})")

if __name__ == "__main__":
    ingest_creative(
        user_data={
            "name": "Sarah J", 
            "role": "Model", 
            "height": 175, 
            "location": "London",
            "tags": ["grunge", "streetwear"]
        },
        image_path="test.jpg"
    )
    
    ingest_creative(
        user_data={
            "name": "Mike Ross", 
            "role": "Actor", 
            "union": "SAG", 
            "location": "NYC"
        },
        image_path="test.jpg" 
    )