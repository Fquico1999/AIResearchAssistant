# --- SQLite Override ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully switched to pysqlite3-binary for SQLite.")
except ImportError:
    print("pysqlite3-binary not found or not needed, using system sqlite3.")
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import os
from typing import List, Dict, Any
import time 

# Import RAG specific components
try:
    from src.query_rag import (
        answer_question_with_rag,
        get_embedding_model_singleton,
        get_chroma_collection_singleton
    )
    RAG_ENABLED = True
    print("RAG components imported successfully.")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import RAG components: {e}. The RAG API will not function.")
    RAG_ENABLED = False
    # Define dummy functions if RAG is not enabled to prevent NameErrors later
    # This allows the app to start but RAG endpoints would fail gracefully.
    async def answer_question_with_rag(*args, **kwargs):
        raise NotImplementedError("RAG components failed to load.")
    def get_embedding_model_singleton():
        raise NotImplementedError("RAG components failed to load.")
    def get_chroma_collection_singleton():
        raise NotImplementedError("RAG components failed to load.")

# --- Basic Logging Configuration ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(levelname)s - PID:%(process)d - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Research Assistant - RAG API",
    description="Provides RAG capabilities for querying research papers.",
    version="0.1.0-rag" # Version specific to RAG focus
)

# --- Application Lifespan Events (for RAG components) ---
if RAG_ENABLED:
    @app.on_event("startup")
    async def startup_rag_components():
        logger.info(f"PID: {os.getpid()} - FastAPI app startup: Initializing RAG components...")
        try:
            # These functions from rag_processor.py should handle their own singleton logic
            get_embedding_model_singleton()
            get_chroma_collection_singleton()
            logger.info(f"PID: {os.getpid()} - RAG components initialized successfully.")
        except Exception as e:
            logger.error(f"PID: {os.getpid()} - CRITICAL: Failed to initialize RAG components during startup: {e}", exc_info=True)
            # Depending on how critical these are, you might want the app to exit or mark itself as unhealthy.
            # For now, it will log the error, and endpoints will likely fail if components aren't ready.
else:
    logger.error("RAG functionality is DISABLED due to import errors. API will be limited.")


# --- Pydantic Models ---
class RAGQueryRequest(BaseModel):
    question: str = Field(..., example="What are the main challenges in AI ethics?")
    top_k_chunks: int = Field(3, ge=1, le=10, example=3, description="Number of relevant chunks to retrieve.")

class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]] = Field([], description="Metadata of the source chunks used for the answer.")

# --- API Endpoints ---
@app.get("/", summary="Root Endpoint")
async def root():
    return {
        "message": "AI Research Assistant - RAG API is running.",
        "rag_status": "ENABLED" if RAG_ENABLED else "DISABLED - Check logs for import errors."
    }

if RAG_ENABLED:
    @app.post("/rag_query", response_model=RAGQueryResponse, summary="Answer a question using Retrieval Augmented Generation")
    async def rag_query_endpoint(request: RAGQueryRequest):
        worker_pid = os.getpid()
        logger.info(f"PID: {worker_pid} - Received RAG query: '{request.question}'")
        start_time = time.perf_counter()

        try:
            # The singleton getters in rag_processor should ensure components are ready
            # (either loaded at startup or on first call by this worker).
            result_data = await answer_question_with_rag(
                user_question=request.question,
                top_k_chunks=request.top_k_chunks
            )
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"PID: {worker_pid} - RAG query processed in {duration_ms:.2f}ms")

            return RAGQueryResponse(
                question=request.question,
                answer=result_data["answer"],
                sources=result_data["sources"]
            )
        except NotImplementedError as nie: # Catch if RAG components truly failed to load
            logger.error(f"PID: {worker_pid} - RAG query failed: RAG components not loaded. {nie}")
            raise HTTPException(status_code=503, detail="RAG service is currently unavailable due to component load failure.")
        except Exception as e:
            logger.error(f"PID: {worker_pid} - Error during RAG query for '{request.question}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An error occurred while processing your RAG query: {str(e)}")
else:
    # Provide a placeholder endpoint if RAG is disabled, so routes don't completely break if expected
    @app.post("/rag_query", summary="RAG Endpoint (Currently Disabled)")
    async def disabled_rag_query_endpoint(request: RAGQueryRequest):
        logger.warning("Attempted to call /rag_query, but RAG functionality is disabled due to import errors.")
        raise HTTPException(status_code=503, detail="RAG functionality is currently disabled. Please check server logs.")


@app.get("/healthz", summary="Health Check")
async def healthz():
    # Basic health check - app is running.
    # Could be expanded to check RAG component health if RAG_ENABLED.
    if RAG_ENABLED:
        try:
            # A light check, e.g., if clients are not None
            if get_embedding_model_singleton() is not None and get_chroma_collection_singleton() is not None:
                return {"status": "ok", "rag_components": "healthy", "pid": os.getpid()}
            else:
                return {"status": "degraded", "rag_components": "unhealthy_init", "pid": os.getpid()}
        except Exception as e:
             return {"status": "degraded", "rag_components": f"error_checking_health: {str(e)}", "pid": os.getpid()}
    return {"status": "ok", "rag_components": "disabled", "pid": os.getpid()}


# --- Main Execution (for local dev) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development (RAG API Focused)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)