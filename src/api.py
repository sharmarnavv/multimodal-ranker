from pydantic import BaseModel
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from qdrant_client import models
from src.embedders import VibeEngine
from src.db import get_client, COLLECTION_NAME
import shutil
import os
import uuid

app = FastAPI(title="Multimodal Creatives Search")

print("Loading Models...")
engine = VibeEngine()
client = get_client()
print("Server Ready.")

class SearchRequest(BaseModel):
    query: str
    modality: str = "visual" # "visual" or "audio"
    filters: Optional[Dict[str, str]] = None
    limit: int = 10

@app.post("/search")
def search_creatives(req: SearchRequest):
    """
    Search by text against either Image ('visual') or Audio ('audio') vectors.
    """
    if req.modality not in ["visual", "audio"]:
        raise HTTPException(400, "Modality must be 'visual' or 'audio'")

    # 1. Vectorize Query (Text -> Correct Vector Space)
    query_vector = engine.get_text_vector(req.query, modality=req.modality)
    
    # 2. Build Filter
    qdrant_filter = None
    if req.filters:
        conditions = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v)) 
            for k, v in req.filters.items()
        ]
        qdrant_filter = models.Filter(must=conditions)

    # 3. Search specific Named Vector
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=(req.modality, query_vector), # Tuple: (vector_name, vector_data)
        query_filter=qdrant_filter,
        limit=req.limit,
        with_payload=True
    )
    
    return {"matches": [{"id": h.id, "score": h.score, "data": h.payload} for h in hits]}

@app.post("/ingest")
async def ingest_endpoint(
    name: str = Form(...),
    role: str = Form(...),
    image: UploadFile = File(None), # Optional
    audio: UploadFile = File(None)  # Optional
):
    vectors = {}
    
    # Handle Image
    if image:
        with open("temp_img", "wb") as f:
            shutil.copyfileobj(image.file, f)
        vec = engine.get_image_vector("temp_img")
        if vec: vectors["visual"] = vec
        os.remove("temp_img")

    # Handle Audio
    if audio:
        with open("temp_audio", "wb") as f:
            shutil.copyfileobj(audio.file, f)
        vec = engine.get_audio_vector("temp_audio")
        if vec: vectors["audio"] = vec
        os.remove("temp_audio")

    if not vectors:
        raise HTTPException(400, "Must provide at least Image or Audio")

    # Upload
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vectors,
                payload={"name": name, "role": role}
            )
        ]
    )
    return {"status": "success", "id": point_id}