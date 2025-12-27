import shutil
import os
import uuid
from typing import Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from qdrant_client.http import models

# imports
from src.searcher import CreativeSearch
from src.db import COLLECTION_NAME

app = FastAPI(title="Creatives Search Engine")

# load model once (global)
# takes a few sec on startup
print("Starting Server & Loading AI Models...")
search_runner = CreativeSearch()
print("Server Ready.")

# --- models ---
class SearchRequest(BaseModel):
    query: str
    hard_filters: Optional[Dict[str, str]] = None
    soft_filters: Optional[Dict[str, str]] = None
    limit: int = 10

# --- endpoints ---

@app.get("/health")
def health_check():
    """simple health check"""
    return {"status": "online", "engine": "CLIP-ViT-B/32"}

@app.post("/search")
def search_creatives(req: SearchRequest):
    """
    hybrid search: vibe (text) + hard filters + soft boosts
    """
    results = search_runner.search(
        text_query=req.query,
        hard_filters=req.hard_filters,
        soft_filters=req.soft_filters,
        limit=req.limit
    )
    
    # format for frontend
    response = []
    for hit in results:
        response.append({
            "id": hit.id,
            "score": round(hit.score, 4),
            "data": hit.payload,
            "boosted": "_debug_boost" in hit.payload  # flag if boosted
        })
    
    return {"matches": response}

@app.post("/ingest")
async def ingest_user(
    name: str = Form(...),
    role: str = Form(...),
    location: str = Form(...),
    image: UploadFile = File(...)
):
    """
    uploads user + image -> db
    """
    # 1. save temp image
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # 2. reuse engine (saves ram)
        # access internal .engine
        vector = search_runner.engine.get_image_vector(temp_filename)
        
        if not vector:
            raise HTTPException(status_code=400, detail="Could not process image")

        # 3. upload to qdrant
        point_id = str(uuid.uuid4())
        payload = {
            "name": name,
            "role": role,
            "location": location
        }
        
        search_runner.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        
        return {"status": "success", "id": point_id, "name": name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # cleanup: delete temp image
        if os.path.exists(temp_filename):
            os.remove(temp_filename)