"""
FastAPI Application Template
Quick reference for building production APIs
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import logging
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Your API Title",
    version="1.0.0",
    description="API Description"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model with validation"""
    question: str = Field(..., min_length=1, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=20)
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is LangChain?",
                "max_results": 5,
                "session_id": "user_123"
            }
        }


class QueryResponse(BaseModel):
    """Response model"""
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[str]
    latency_ms: float
    session_id: str


# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    logger.info(f"[{request_id}] {request.method} {request.url}")

    response = await call_next(request)

    duration = (time.time() - start_time) * 1000
    logger.info(f"[{request_id}] {response.status_code} ({duration:.2f}ms)")

    response.headers["X-Request-ID"] = request_id
    return response


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": str(exc), "type": "validation_error"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "type": "server_error"}
    )


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """Readiness check for dependencies"""
    checks = {
        "database": True,  # Replace with actual check
        "cache": True,     # Replace with actual check
    }

    all_ready = all(checks.values())
    status_code = status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "not_ready",
            "checks": checks
        }
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a query.

    - **question**: User's question (required, 1-1000 chars)
    - **max_results**: Number of results to return (1-20, default: 5)
    - **session_id**: Session identifier (optional)
    """
    start_time = time.time()

    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Process query (replace with actual logic)
        answer = f"Answer to: {request.question}"
        sources = ["source1", "source2"]
        confidence = 0.85

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            answer=answer,
            confidence=confidence,
            sources=sources[:request.max_results],
            latency_ms=latency_ms,
            session_id=session_id
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/items/{item_id}")
async def get_item(
    item_id: int,
    q: Optional[str] = None,
    skip: int = 0,
    limit: int = 10
):
    """
    Get item by ID with optional query parameters.

    - **item_id**: Item ID (path parameter)
    - **q**: Search query (optional)
    - **skip**: Number of items to skip (default: 0)
    - **limit**: Max items to return (default: 10)
    """
    return {
        "item_id": item_id,
        "q": q,
        "skip": skip,
        "limit": limit
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Application starting up...")
    # Initialize database connections, caches, etc.


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Application shutting down...")
    # Close database connections, cleanup resources


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        log_level="info"
    )
