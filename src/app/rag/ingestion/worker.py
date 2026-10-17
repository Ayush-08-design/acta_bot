import asyncio
import os
from pathlib import Path

import redis
from celery import Celery
from celery.exceptions import Ignore

from src.app.rag.ingestion.processing import Chunker, Loader
from src.app.rag.store.vec_store import VectorStore

r = redis.Redis(
    host="localhost", port=6379, db=0, decode_responses=True
)  # decode_response ensures that redis return a python object and not a byte string

celery_app = Celery(
    "doc_ingestion_worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

### Background job

@celery_app.task(bind=True)
def doc_ingestion(self, file_path: str, user_id: str, thread_id: str):
    # 1. Check Redis Connection safely
    try:
        if not r.ping():
            raise redis.ConnectionError("Redis is down")
    except redis.ConnectionError as e:
        # Custom fail state or retry instead of sys.exit()
        self.update_state(state="FAILURE", meta={"exc_message": str(e)})
        raise Ignore() 

    # 2. Update state to PROGRESS
    self.update_state(
        state="PROGRESS", 
        meta={"current": 10, "status": "Extracting text..."}
    )
    
    file_path = Path(file_path)
    
    # 3. Text Extraction & Chunking
    loader = Loader(file_path)
    docs = loader.load()
    docs = Chunker().chunk(docs, thread_id)
    
    self.update_state(
        state="PROGRESS", 
        meta={"current": 50, "status": "Generating Embeddings..."}
    )
    # 4. Handle Async operation using asyncio.run
    store = VectorStore()
    asyncio.run(store.add_file(user_id, docs))
    
    # 5. Cleanup local temporary file
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return {"status": "Complete"}

