# Challenge 01: FastAPI RAG Endpoint (Production)

**Difficulty**: Intermediate-Advanced
**Time Estimate**: 60-75 minutes
**Interview Section**: Integration of Sections 1, 2, 3

---

## 📋 Challenge Description

Build a **production-ready FastAPI endpoint** for a RAG system with:
1. Proper request/response models (Pydantic)
2. Caching layer
3. Error handling and logging
4. Input validation
5. API documentation

---

## 🎯 Requirements

### Part A: Request/Response Models (10 min)

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    max_results: int = Field(default=5, ge=1, le=20)
    use_cache: bool = True
    use_reranking: bool = False

class Source(BaseModel):
    document: str
    score: float
    index: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    cached: bool
    latency_ms: float
    session_id: str
```

### Part B: FastAPI Endpoint (20 min)

```python
from fastapi import FastAPI, HTTPException
import logging

app = FastAPI(title="RAG API", version="1.0.0")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a question using RAG system.

    - **question**: User's question (required)
    - **session_id**: For conversation context (optional)
    - **max_results**: Number of sources to return (1-20)
    - **use_cache**: Whether to use cached results
    - **use_reranking**: Whether to re-rank results
    """
```

### Part C: Caching Layer (15 min)

```python
import hashlib
from functools import wraps

def cache_query(ttl: int = 300):
    """
    Decorator to cache query results.

    Args:
        ttl: Time-to-live in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from question
            request = args[0]
            cache_key = hashlib.md5(
                request.question.encode()
            ).hexdigest()

            # Check cache
            if request.use_cache:
                cached = cache.get(cache_key)
                if cached:
                    return cached

            # Execute and cache result
            result = await func(*args, **kwargs)
            cache.put(cache_key, result, ttl=ttl)

            return result
        return wrapper
    return decorator
```

### Part D: Error Handling (10 min)

```python
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": str(exc), "type": "validation_error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "type": "server_error"}
    )
```

### Part E: Logging & Monitoring (10 min)

```python
import logging
import time

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()

    logger.info(f"Request: {request.method} {request.url}")

    response = await call_next(request)

    duration = (time.time() - start_time) * 1000
    logger.info(f"Response: {response.status_code} ({duration:.2f}ms)")

    return response
```

---

## 📊 Example API Calls

### Request 1: Simple Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is LangChain?",
    "max_results": 3,
    "use_cache": true
  }'
```

**Response:**
```json
{
  "answer": "LangChain is a framework for building LLM applications...",
  "sources": [
    {
      "document": "LangChain is a framework...",
      "score": 0.92,
      "index": 0
    },
    {
      "document": "LangChain provides...",
      "score": 0.85,
      "index": 1
    }
  ],
  "confidence": 0.89,
  "cached": false,
  "latency_ms": 145.3,
  "session_id": "auto-generated-uuid"
}
```

### Request 2: Conversational Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are its main components?",
    "session_id": "user_123",
    "use_reranking": true
  }'
```

### Request 3: Cached Query

```bash
# First call
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?", "use_cache": true}'

# Second call (cached, faster)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?", "use_cache": true}'
```

**Response (cached):**
```json
{
  "answer": "Python is a programming language...",
  "sources": [...],
  "confidence": 0.95,
  "cached": true,  ← Note: cached=true
  "latency_ms": 2.1,  ← Much faster!
  "session_id": "auto-generated-uuid"
}
```

---

## 💡 Implementation Tips

### Complete Endpoint Implementation

```python
from fastapi import FastAPI
import uuid
import time

app = FastAPI()
rag_system = ConversationalRAG()  # From previous challenges
cache = TTLCache(capacity=1000)

@app.post("/query", response_model=QueryResponse)
@cache_query(ttl=300)
async def query_endpoint(request: QueryRequest):
    start_time = time.time()

    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Execute RAG query
        result = await rag_system.query(
            question=request.question,
            session_id=session_id,
            top_k=request.max_results,
            use_reranking=request.use_reranking
        )

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Format sources
        sources = [
            Source(document=doc, score=0.9, index=i)
            for i, doc in enumerate(result["sources"])
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            confidence=result["confidence"],
            cached=False,
            latency_ms=latency_ms,
            session_id=session_id
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
```

---

## 🎓 Key Concepts

1. **Pydantic Models**: Type-safe request/response validation
2. **FastAPI Decorators**: Route handlers, middleware
3. **Caching Strategy**: MD5 hash for cache keys
4. **Error Handling**: Custom exception handlers
5. **Logging**: Structured logging for debugging
6. **API Documentation**: Auto-generated with FastAPI

---

## 🧪 Testing

```python
import pytest
from fastapi.testclient import TestClient

client = TestClient(app)

def test_query_endpoint():
    response = client.post("/query", json={
        "question": "What is LangChain?",
        "max_results": 3
    })

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) <= 3
    assert 0.0 <= data["confidence"] <= 1.0

def test_cached_query():
    # First call
    response1 = client.post("/query", json={
        "question": "Test question",
        "use_cache": True
    })

    # Second call (should be cached)
    response2 = client.post("/query", json={
        "question": "Test question",
        "use_cache": True
    })

    assert response2.json()["cached"] == True
    assert response2.json()["latency_ms"] < response1.json()["latency_ms"]
```

---

**Time Allocation**:
- Pydantic models: 10 min
- Endpoint implementation: 20 min
- Caching: 15 min
- Error handling: 10 min
- Logging: 10 min
- Testing: 10 min
- **Total**: 75 min

**Good luck!** 🎯
