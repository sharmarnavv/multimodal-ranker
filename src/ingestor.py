import uuid
from qdrant_client.http import models
from src.db import get_client, COLLECTION_NAME
from src.embedders import VibeEngine

engine = VibeEngine()
client = get_client()

def ingest_creative(user_data, image_path=None, audio_path=None):
    print(f"Processing {user_data.get('name')}...")
    
    # Dictionary to hold whichever vectors we generate
    vectors_to_store = {}
    
    # 1. Process Visual
    if image_path:
        vis_vec = engine.get_image_vector(image_path)
        if vis_vec:
            vectors_to_store["visual"] = vis_vec
            
    # 2. Process Audio
    if audio_path:
        aud_vec = engine.get_audio_vector(audio_path)
        if aud_vec:
            vectors_to_store["audio"] = aud_vec
            
    if not vectors_to_store:
        print("⚠️ Skipping: No valid image or audio provided.")
        return

    # 3. Upload with Named Vectors
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vectors_to_store, # Pass dict {'visual': [...], 'audio': [...]}
                payload=user_data
            )
        ]
    )
    print(f"✅ Indexed {user_data['name']} (ID: {point_id})")