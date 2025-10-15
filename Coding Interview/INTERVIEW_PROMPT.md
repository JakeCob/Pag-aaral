# ELGO AI - AI Engineer Coding Interview
## Candidate: Jacob Matthew Rafal

### About ELGO AI
ELGO AI is a Singapore-based startup building a cutting-edge no-code GenAI platform for enterprise customers. Our platform enables businesses to deploy production-grade RAG systems, multi-agent workflows, and intelligent automation without writing code. As an AI Engineer, you'll be working on:

- **RAG Systems**: Building scalable retrieval-augmented generation pipelines for enterprise knowledge bases
- **Multi-Agent Orchestration**: Designing workflows that coordinate specialized AI agents
- **API Integrations**: Creating robust integrations with LLM providers, vector databases, and enterprise tools
- **Production Infrastructure**: Ensuring reliability, security, and performance at scale

**Tech Stack**: Python, FastAPI, LangChain, LlamaIndex, ChromaDB, Pinecone, Weaviate, AWS (Lambda, ECS, S3, SageMaker), Docker, Redis

---

## Interview Structure

This interview consists of three sections designed to evaluate your skills across different dimensions:

1. **Section 1: RAG Implementation** (90 minutes) - Building production-ready RAG systems
2. **Section 2: Algorithmic Problem Solving** (45 minutes) - Data structures and algorithms
3. **Section 3: System Design** (60 minutes) - Multi-agent workflow architecture

**Total Time**: ~3 hours (take breaks as needed)

---

## Section 1: RAG Implementation Challenge (90 minutes)

### Context
At ELGO AI, we frequently need to build custom RAG systems for enterprise clients with varying requirements. This section tests your ability to design and implement a production-ready document Q&A system.

### Example Question with Complete Solution

**Question**: Build a simple document QA system that can ingest company documents and answer questions about them.

**Requirements**:
- Support document upload via REST API
- Chunk documents intelligently
- Store embeddings in a vector database
- Retrieve relevant chunks for queries
- Generate answers using an LLM

**Complete Solution**:

```python
"""
Simple Document QA System
Author: Interview Example Solution
"""

import os
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ELGO Document QA System",
    description="Simple RAG system for document Q&A",
    version="1.0.0"
)

# Initialize ChromaDB with persistence
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Document storage
document_store = {}


class Query(BaseModel):
    """Query request model"""
    question: str
    doc_id: str
    max_chunks: int = 3

    class Config:
        schema_extra = {
            "example": {
                "question": "What is the company's vacation policy?",
                "doc_id": "abc123",
                "max_chunks": 3
            }
        }


class UploadResponse(BaseModel):
    """Upload response model"""
    doc_id: str
    message: str
    chunks_created: int


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    source_chunks: List[str]
    confidence: float


class DocumentProcessor:
    """Handles document chunking and embedding"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor

        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = OpenAIEmbeddings()
        logger.info(f"DocumentProcessor initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def process_document(self, text: str, doc_id: str) -> Dict:
        """
        Process document into chunks and create embeddings

        Args:
            text: Document text content
            doc_id: Unique document identifier

        Returns:
            Dict with processing results

        Raises:
            HTTPException: If processing fails
        """
        try:
            logger.info(f"Processing document {doc_id}")

            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks for document {doc_id}")

            # Create metadata for each chunk
            metadatas = [
                {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]

            # Get or create collection for this document
            collection = client.get_or_create_collection(name=f"doc_{doc_id}")

            # Add documents with embeddings
            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            )

            logger.info(f"Successfully stored embeddings for document {doc_id}")

            return {
                "doc_id": doc_id,
                "chunks_created": len(chunks),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


class QueryEngine:
    """Handles query processing and response generation"""

    def __init__(self):
        """Initialize query engine with LLM and embeddings"""
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.7)
        logger.info("QueryEngine initialized")

    def query_document(self, question: str, doc_id: str, max_chunks: int = 3) -> Dict:
        """
        Query a specific document and return answer

        Args:
            question: User's question
            doc_id: Document to query
            max_chunks: Maximum chunks to retrieve

        Returns:
            Dict with answer, sources, and confidence

        Raises:
            HTTPException: If query fails
        """
        try:
            logger.info(f"Querying document {doc_id} with question: {question}")

            # Get the collection for this document
            collection = client.get_collection(name=f"doc_{doc_id}")

            # Create LangChain vectorstore from collection
            vectorstore = Chroma(
                client=client,
                collection_name=f"doc_{doc_id}",
                embedding_function=self.embeddings
            )

            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": max_chunks}
                ),
                return_source_documents=True
            )

            # Execute query
            result = qa_chain({"query": question})

            # Extract source chunks
            source_chunks = [doc.page_content for doc in result.get("source_documents", [])]

            # Calculate confidence (simplified - based on source availability)
            confidence = min(len(source_chunks) / max_chunks, 1.0)

            logger.info(f"Query completed for document {doc_id}, confidence: {confidence}")

            return {
                "answer": result["result"],
                "source_chunks": source_chunks,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error querying document {doc_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


# Initialize processors
doc_processor = DocumentProcessor()
query_engine = QueryEngine()


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document

    Args:
        file: Document file to upload

    Returns:
        UploadResponse with document ID and processing stats
    """
    try:
        logger.info(f"Received upload request for file: {file.filename}")

        # Read file content
        content = await file.read()
        text = content.decode('utf-8')

        # Generate document ID from content hash
        doc_id = hashlib.md5(text.encode()).hexdigest()[:8]

        # Process document
        result = doc_processor.process_document(text, doc_id)

        # Store document metadata
        document_store[doc_id] = {
            "filename": file.filename,
            "chunks": result["chunks_created"],
            "size_bytes": len(content)
        }

        return UploadResponse(
            doc_id=doc_id,
            message=f"Document {file.filename} processed successfully",
            chunks_created=result["chunks_created"]
        )

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_document(query: Query):
    """
    Query a processed document

    Args:
        query: Query request with question and document ID

    Returns:
        QueryResponse with answer and source chunks
    """
    if query.doc_id not in document_store:
        raise HTTPException(
            status_code=404,
            detail=f"Document {query.doc_id} not found. Please upload it first."
        )

    result = query_engine.query_document(
        query.question,
        query.doc_id,
        query.max_chunks
    )

    return QueryResponse(**result)


@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    return {
        "total_documents": len(document_store),
        "documents": document_store
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ELGO Document QA"}


# Error handling middleware
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"error": str(exc), "detail": "An unexpected error occurred"}
```

**Key Features of Example Solution**:
- ✅ Clean code structure with separate classes for concerns
- ✅ Comprehensive error handling
- ✅ Logging for observability
- ✅ Type hints throughout
- ✅ Proper documentation
- ✅ RESTful API design
- ✅ Health check endpoint

---

### Your Practice Problem: Advanced RAG System

Now it's your turn! Build an **enhanced RAG system** with the following requirements:

#### Part A: Multi-Format Document Management (30 points)

**Requirements**:
1. Support multiple file formats:
   - `.txt` - Plain text
   - `.pdf` - PDF documents (use PyPDF2 or pdfplumber)
   - `.json` - Structured JSON data
   - `.csv` - Tabular data

2. Implement document versioning:
   - Track multiple versions of the same document
   - Allow querying specific versions
   - Store version metadata (upload time, size, chunk count)

3. Add document lifecycle management:
   - `DELETE /documents/{doc_id}` - Remove document and embeddings
   - `GET /documents/{doc_id}/versions` - List all versions
   - `GET /documents/{doc_id}/metadata` - Get document metadata

**Starter Code**:
```python
from typing import List, Dict, Optional, Literal
from datetime import datetime
import PyPDF2
import csv
import json

class MultiFormatDocumentProcessor:
    """Process multiple document formats"""

    def __init__(self):
        pass

    def process_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF"""
        # TODO: Implement
        pass

    def process_json(self, file_content: bytes) -> str:
        """Extract and format JSON data"""
        # TODO: Implement
        pass

    def process_csv(self, file_content: bytes) -> str:
        """Extract and format CSV data"""
        # TODO: Implement
        pass

class DocumentVersionManager:
    """Manage document versions"""

    def __init__(self):
        self.versions = {}  # doc_id -> List[version_info]

    def add_version(self, doc_id: str, metadata: Dict) -> int:
        """Add new version and return version number"""
        # TODO: Implement
        pass

    def get_versions(self, doc_id: str) -> List[Dict]:
        """Get all versions of a document"""
        # TODO: Implement
        pass

    def delete_document(self, doc_id: str) -> bool:
        """Delete all versions of a document"""
        # TODO: Implement
        pass
```

#### Part B: Hybrid Search with Re-ranking (25 points)

**Requirements**:
1. Implement hybrid search combining:
   - Semantic search (dense embeddings)
   - Keyword search (BM25 algorithm)
   - Weighted combination of both

2. Add re-ranking capability:
   - Use cross-encoder model for re-ranking
   - Improve precision of top results
   - Return relevance scores

**Starter Code**:
```python
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class HybridRetriever:
    """Implement hybrid search with semantic + keyword"""

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for semantic search (1-alpha for BM25)
        """
        self.alpha = alpha
        self.bm25 = None
        # TODO: Initialize

    def index_documents(self, documents: List[str], doc_id: str):
        """Index documents for BM25"""
        # TODO: Implement BM25 indexing
        pass

    def retrieve(self, query: str, doc_id: str, k: int = 10) -> List[Dict]:
        """
        Retrieve using hybrid search

        Returns: List of {chunk, score, method} dicts
        """
        # TODO: Implement hybrid retrieval
        pass

class Reranker:
    """Re-rank results using cross-encoder"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[Dict]:
        """
        Re-rank documents and return top_k

        Returns: List of {chunk, score} sorted by score
        """
        # TODO: Implement re-ranking
        pass
```

#### Part C: Conversation Memory & Context Management (25 points)

**Requirements**:
1. Implement conversation memory:
   - Store chat history per session
   - Include previous Q&A pairs in context
   - Limit context window to prevent token overflow

2. Support follow-up questions:
   - Resolve pronouns and references
   - Maintain context across turns
   - Clear memory on explicit request

**Starter Code**:
```python
from collections import deque

class ConversationMemory:
    """Manage conversation history and context"""

    def __init__(self, max_history: int = 5, max_tokens: int = 2000):
        """
        Args:
            max_history: Maximum conversation turns to keep
            max_tokens: Maximum total tokens in context
        """
        self.sessions = {}  # session_id -> deque of messages
        self.max_history = max_history
        self.max_tokens = max_tokens

    def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        # TODO: Implement
        pass

    def get_context(self, session_id: str) -> str:
        """Get formatted conversation context"""
        # TODO: Implement with token limiting
        pass

    def clear_session(self, session_id: str):
        """Clear conversation history for session"""
        # TODO: Implement
        pass

class ContextAwareQueryEngine:
    """Query engine with conversation context"""

    def __init__(self, memory: ConversationMemory):
        self.memory = memory
        # TODO: Initialize LLM

    def query_with_context(
        self,
        question: str,
        doc_id: str,
        session_id: str
    ) -> Dict:
        """Answer question using conversation context"""
        # TODO: Implement
        pass
```

#### Part D: Evaluation & Monitoring (20 points)

**Requirements**:
1. Implement faithfulness scoring:
   - Measure if answer is supported by retrieved chunks
   - Use LLM-based evaluation or RAGAS framework
   - Return confidence score

2. Add query caching:
   - Cache frequent queries
   - Implement TTL (time-to-live)
   - Return cached responses when available

3. Track usage metrics:
   - Query latency
   - Cache hit rate
   - Average chunks retrieved
   - Total API calls

**Starter Code**:
```python
import time
from datetime import datetime, timedelta

class FaithfulnessEvaluator:
    """Evaluate answer faithfulness to sources"""

    def __init__(self):
        # TODO: Initialize LLM for evaluation
        pass

    def evaluate(self, question: str, answer: str, sources: List[str]) -> float:
        """
        Evaluate faithfulness score

        Returns: Score between 0.0 and 1.0
        """
        # TODO: Implement using RAGAS or custom LLM prompting
        pass

class QueryCache:
    """Cache for frequent queries"""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}  # query_hash -> (response, expiry)
        self.ttl = ttl_seconds

    def get(self, query: str, doc_id: str) -> Optional[Dict]:
        """Get cached response if available and not expired"""
        # TODO: Implement
        pass

    def put(self, query: str, doc_id: str, response: Dict):
        """Cache a response"""
        # TODO: Implement
        pass

class MetricsTracker:
    """Track system metrics"""

    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency_ms": 0,
            "total_chunks_retrieved": 0
        }

    def record_query(self, latency_ms: float, cached: bool, chunks: int):
        """Record query metrics"""
        # TODO: Implement
        pass

    def get_stats(self) -> Dict:
        """Get aggregated statistics"""
        # TODO: Calculate averages and rates
        pass
```

---

### Expected Deliverable

Create a complete FastAPI application that integrates all four parts (A, B, C, D). Your solution should:

1. **Work end-to-end** - All endpoints functional
2. **Handle errors gracefully** - Proper exception handling
3. **Be well-documented** - Docstrings and comments
4. **Include type hints** - Throughout the codebase
5. **Have logging** - For debugging and monitoring
6. **Be testable** - Clear separation of concerns

**Bonus Points** (+10):
- Unit tests for key components
- Docker containerization
- Rate limiting
- Authentication middleware
- OpenAPI documentation examples

---

## Section 2: Algorithmic Problem Solving (45 minutes)

### Example Question with Complete Solution

**Question**: Design a rate limiter for ELGO's API that limits requests using a sliding window approach.

**Complete Solution**:

```python
"""
Sliding Window Rate Limiter Implementation
Time Complexity: O(1) amortized
Space Complexity: O(n) where n is max_requests
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Dict
import threading

class SlidingWindowRateLimiter:
    """
    Implements a sliding window rate limiter for API requests.

    Example:
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        if limiter.is_allowed("user123"):
            # Process request
            pass
        else:
            # Return 429 Too Many Requests
            wait_time = limiter.get_wait_time("user123")
    """

    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize rate limiter

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # user_id -> deque of timestamps
        self.lock = threading.Lock()

    def is_allowed(self, user_id: str) -> bool:
        """
        Check if request is allowed for user

        Args:
            user_id: Unique user identifier

        Returns:
            True if allowed, False if rate limited
        """
        with self.lock:
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=self.window_seconds)

            # Initialize user's request queue if needed
            if user_id not in self.requests:
                self.requests[user_id] = deque()

            user_requests = self.requests[user_id]

            # Remove expired requests (outside sliding window)
            while user_requests and user_requests[0] < window_start:
                user_requests.popleft()

            # Check if under limit
            if len(user_requests) < self.max_requests:
                user_requests.append(current_time)
                return True

            return False

    def get_wait_time(self, user_id: str) -> float:
        """
        Get seconds until next request is allowed

        Args:
            user_id: Unique user identifier

        Returns:
            Seconds to wait (0.0 if can request now)
        """
        with self.lock:
            if user_id not in self.requests or not self.requests[user_id]:
                return 0.0

            oldest_request = self.requests[user_id][0]
            window_end = oldest_request + timedelta(seconds=self.window_seconds)
            wait_time = (window_end - datetime.now()).total_seconds()

            return max(0.0, wait_time)

    def get_remaining_requests(self, user_id: str) -> int:
        """
        Get remaining requests in current window

        Args:
            user_id: Unique user identifier

        Returns:
            Number of requests remaining
        """
        with self.lock:
            if user_id not in self.requests:
                return self.max_requests

            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=self.window_seconds)

            # Count valid requests in window
            valid_requests = sum(
                1 for req_time in self.requests[user_id]
                if req_time >= window_start
            )

            return max(0, self.max_requests - valid_requests)


class TokenBucketRateLimiter:
    """
    Token bucket algorithm for handling burst traffic
    More lenient for occasional bursts while maintaining average rate
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets = {}  # user_id -> (tokens, last_refill_time)
        self.lock = threading.Lock()

    def _refill_tokens(self, user_id: str) -> None:
        """Refill tokens based on elapsed time"""
        if user_id not in self.buckets:
            self.buckets[user_id] = (self.capacity, datetime.now())
            return

        tokens, last_refill = self.buckets[user_id]
        current_time = datetime.now()
        elapsed = (current_time - last_refill).total_seconds()

        # Calculate new tokens (capped at capacity)
        new_tokens = min(self.capacity, tokens + elapsed * self.refill_rate)
        self.buckets[user_id] = (new_tokens, current_time)

    def consume(self, user_id: str, tokens: int = 1) -> bool:
        """
        Try to consume tokens for a request

        Args:
            user_id: Unique user identifier
            tokens: Number of tokens to consume

        Returns:
            True if tokens available, False otherwise
        """
        with self.lock:
            self._refill_tokens(user_id)

            current_tokens, last_refill = self.buckets[user_id]

            if current_tokens >= tokens:
                self.buckets[user_id] = (current_tokens - tokens, last_refill)
                return True

            return False

    def get_available_tokens(self, user_id: str) -> float:
        """
        Get current available tokens

        Args:
            user_id: Unique user identifier

        Returns:
            Current token count
        """
        with self.lock:
            self._refill_tokens(user_id)
            tokens, _ = self.buckets[user_id]
            return tokens


# Testing code
def test_rate_limiters():
    """Comprehensive tests for both rate limiter implementations"""

    print("Testing Sliding Window Rate Limiter...")
    sliding_limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=10)

    user = "user123"
    results = []
    for i in range(5):
        allowed = sliding_limiter.is_allowed(user)
        results.append(allowed)
        remaining = sliding_limiter.get_remaining_requests(user)
        print(f"Request {i+1}: {'✓ Allowed' if allowed else '✗ Blocked'} (remaining: {remaining})")

    assert results == [True, True, True, False, False], "Sliding window test failed"

    print("\nTesting Token Bucket Rate Limiter...")
    token_limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)

    # Burst requests
    burst_results = []
    for i in range(7):
        allowed = token_limiter.consume(user)
        burst_results.append(allowed)
        tokens = token_limiter.get_available_tokens(user)
        print(f"Burst {i+1}: {'✓ Allowed' if allowed else '✗ Blocked'} (tokens: {tokens:.1f})")

    assert burst_results[:5] == [True] * 5, "Token bucket capacity test failed"
    assert burst_results[5:] == [False] * 2, "Token bucket depletion test failed"

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_rate_limiters()
```

---

### Your Practice Problem: Distributed Cache with LRU Eviction

**Problem**: Implement a thread-safe distributed cache for ELGO's RAG queries with LRU eviction and TTL support.

#### Requirements:

1. **LRU Eviction Policy**:
   - Evict least recently used items when cache is full
   - O(1) get and put operations
   - Use doubly-linked list + hashmap

2. **TTL (Time To Live) Support**:
   - Each entry can have custom TTL
   - Expired entries automatically invalid
   - Lazy deletion on access

3. **Thread Safety**:
   - Support concurrent reads/writes
   - Use appropriate locking mechanisms
   - No race conditions

4. **Statistics Tracking**:
   - Track hits, misses, evictions
   - Calculate hit rate
   - Monitor cache size

5. **Complex Object Serialization**:
   - Support caching embeddings (numpy arrays)
   - Support caching LLM responses (dict/list)
   - Efficient serialization/deserialization

#### Starter Code:

```python
from typing import Any, Optional, Dict, Tuple
import time
import threading
from collections import OrderedDict
import pickle
import numpy as np

class CacheNode:
    """Node for doubly-linked list"""

    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev = None
        self.next = None

class DistributedCache:
    """
    Thread-safe LRU cache with TTL support

    Time Complexity:
        - get: O(1)
        - put: O(1)
        - evict: O(1)

    Space Complexity: O(n) where n is max_size
    """

    def __init__(self, max_size: int, default_ttl: int = 3600):
        """
        Initialize cache

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        # TODO: Implement initialization
        # - Initialize doubly-linked list (head, tail)
        # - Initialize hashmap for O(1) lookup
        # - Initialize locks
        # - Initialize statistics
        pass

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Value if found and not expired, None otherwise

        Implementation:
            1. Check if key exists in hashmap
            2. Check if entry is expired (lazy deletion)
            3. Move node to front (most recently used)
            4. Update statistics
        """
        # TODO: Implement
        pass

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Put value in cache with optional TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)

        Implementation:
            1. If key exists, update and move to front
            2. If cache is full, evict LRU item
            3. Create new node and add to front
            4. Update statistics
        """
        # TODO: Implement
        pass

    def _evict_lru(self) -> None:
        """
        Evict least recently used item (tail of list)
        """
        # TODO: Implement
        pass

    def _move_to_front(self, node: CacheNode) -> None:
        """
        Move node to front of list (most recently used)
        """
        # TODO: Implement
        pass

    def _remove_node(self, node: CacheNode) -> None:
        """
        Remove node from linked list
        """
        # TODO: Implement
        pass

    def _add_to_front(self, node: CacheNode) -> None:
        """
        Add node to front of list
        """
        # TODO: Implement
        pass

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics

        Returns:
            Dict with hits, misses, evictions, hit_rate, current_size
        """
        # TODO: Implement
        pass

    def clear(self) -> None:
        """Clear all cache entries"""
        # TODO: Implement
        pass

    def serialize_value(self, value: Any) -> bytes:
        """
        Serialize complex objects for storage

        Handles:
            - Numpy arrays (embeddings)
            - Dicts and lists (LLM responses)
            - Custom objects
        """
        # TODO: Implement using pickle or msgpack
        pass

    def deserialize_value(self, data: bytes) -> Any:
        """Deserialize stored values"""
        # TODO: Implement
        pass


# Test cases
def test_distributed_cache():
    """Comprehensive test suite"""

    print("Testing LRU Cache Implementation...")

    # Test 1: Basic put/get
    cache = DistributedCache(max_size=3, default_ttl=10)
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1", "Basic get failed"
    print("✓ Test 1 passed: Basic put/get")

    # Test 2: LRU eviction
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    cache.put("key4", "value4")  # Should evict key1
    assert cache.get("key1") is None, "LRU eviction failed"
    assert cache.get("key2") == "value2", "LRU eviction removed wrong item"
    print("✓ Test 2 passed: LRU eviction")

    # Test 3: TTL expiration
    cache_ttl = DistributedCache(max_size=5, default_ttl=1)
    cache_ttl.put("temp", "expires_soon")
    assert cache_ttl.get("temp") == "expires_soon"
    time.sleep(1.5)
    assert cache_ttl.get("temp") is None, "TTL expiration failed"
    print("✓ Test 3 passed: TTL expiration")

    # Test 4: Complex objects (numpy arrays)
    cache_complex = DistributedCache(max_size=5)
    embedding = np.array([0.1, 0.2, 0.3, 0.4])
    cache_complex.put("embedding", embedding)
    retrieved = cache_complex.get("embedding")
    assert np.array_equal(retrieved, embedding), "Numpy serialization failed"
    print("✓ Test 4 passed: Complex object serialization")

    # Test 5: Thread safety
    def worker(cache, thread_id):
        for i in range(100):
            cache.put(f"thread{thread_id}_key{i}", f"value{i}")
            cache.get(f"thread{thread_id}_key{i}")

    cache_concurrent = DistributedCache(max_size=50)
    threads = [
        threading.Thread(target=worker, args=(cache_concurrent, i))
        for i in range(5)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = cache_concurrent.get_stats()
    assert stats["hits"] + stats["misses"] > 0, "Thread safety test failed"
    print(f"✓ Test 5 passed: Thread safety (stats: {stats})")

    # Test 6: Statistics
    cache_stats = DistributedCache(max_size=3)
    cache_stats.put("a", 1)
    cache_stats.get("a")  # hit
    cache_stats.get("b")  # miss
    stats = cache_stats.get_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1, "Statistics tracking failed"
    print(f"✓ Test 6 passed: Statistics tracking")

    print("\n✅ All cache tests passed!")


if __name__ == "__main__":
    test_distributed_cache()
```

#### Expected Features:

1. **LRU Implementation**:
   - Doubly-linked list for O(1) operations
   - HashMap for O(1) key lookup
   - Proper head/tail pointer management

2. **TTL Management**:
   - Store expiry timestamp with each entry
   - Lazy deletion on get()
   - Optional background cleanup thread

3. **Thread Safety**:
   - Use `threading.Lock()` for critical sections
   - Minimize lock contention
   - Consider read-write locks for optimization

4. **Statistics**:
   - Atomic counter updates
   - Calculate hit rate: `hits / (hits + misses)`
   - Track cache size in real-time

5. **Serialization**:
   - Use `pickle` for general objects
   - Handle numpy arrays efficiently
   - Graceful error handling for unsupported types

---

## Section 3: System Design - Multi-Agent Workflow (60 minutes)

### Problem Description

Design and implement a **multi-agent workflow system** for ELGO AI that can intelligently route queries to specialized agents, manage dependencies, and execute workflows efficiently.

### Context

ELGO AI's platform needs to handle diverse enterprise queries:
- **Document Q&A**: RAG-based answers from knowledge bases
- **SQL Generation**: Natural language to SQL queries
- **Code Generation**: Generate Python/JavaScript code
- **Data Analysis**: Analyze CSV/Excel files
- **Web Search**: Real-time information retrieval

Each task requires a specialized agent. Your system should:
1. Classify incoming queries
2. Route to appropriate agent(s)
3. Handle multi-step workflows
4. Execute agents in parallel when possible
5. Aggregate results from multiple agents

### Requirements

#### Part A: Agent Architecture (20 points)

**Requirements**:
1. Create base `Agent` class with common interface
2. Implement at least 3 specialized agents:
   - `RAGAgent`: Answer questions from documents
   - `SQLAgent`: Generate and execute SQL queries
   - `CodeAgent`: Generate code snippets

3. Each agent should:
   - Have clear input/output schemas
   - Handle errors gracefully
   - Return confidence scores
   - Support async execution

**Starter Code**:
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel
from enum import Enum
import asyncio

class AgentType(Enum):
    """Supported agent types"""
    RAG = "rag"
    SQL = "sql"
    CODE = "code"
    ANALYSIS = "analysis"
    WEB_SEARCH = "web_search"

class AgentInput(BaseModel):
    """Standard input for all agents"""
    query: str
    context: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}

class AgentOutput(BaseModel):
    """Standard output from all agents"""
    result: Any
    confidence: float
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None

class Agent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str, agent_type: AgentType):
        self.name = name
        self.agent_type = agent_type

    @abstractmethod
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute agent logic

        Args:
            input_data: Standardized input

        Returns:
            AgentOutput with results
        """
        pass

    @abstractmethod
    def can_handle(self, query: str) -> float:
        """
        Determine if agent can handle query

        Returns:
            Confidence score (0.0 to 1.0)
        """
        pass

class RAGAgent(Agent):
    """Agent for document Q&A using RAG"""

    def __init__(self):
        super().__init__("RAG Agent", AgentType.RAG)
        # TODO: Initialize vectorstore, LLM, etc.

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """Execute RAG query"""
        # TODO: Implement
        pass

    def can_handle(self, query: str) -> float:
        """Check if query is document-related"""
        # TODO: Implement classification logic
        pass

class SQLAgent(Agent):
    """Agent for SQL generation and execution"""

    def __init__(self, db_connection_string: str):
        super().__init__("SQL Agent", AgentType.SQL)
        # TODO: Initialize DB connection

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """Generate and execute SQL"""
        # TODO: Implement
        pass

    def can_handle(self, query: str) -> float:
        """Check if query requires database access"""
        # TODO: Implement
        pass

class CodeAgent(Agent):
    """Agent for code generation"""

    def __init__(self):
        super().__init__("Code Agent", AgentType.CODE)
        # TODO: Initialize code generation LLM

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """Generate code"""
        # TODO: Implement
        pass

    def can_handle(self, query: str) -> float:
        """Check if query requires code generation"""
        # TODO: Implement
        pass
```

#### Part B: Query Router (15 points)

**Requirements**:
1. Implement intelligent query classification
2. Select best agent(s) for each query
3. Support multi-agent queries (e.g., "Query database and summarize results")
4. Handle ambiguous queries

**Starter Code**:
```python
from typing import List, Tuple

class QueryRouter:
    """Route queries to appropriate agents"""

    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def route(self, query: str) -> List[Tuple[Agent, float]]:
        """
        Route query to agent(s)

        Args:
            query: User query

        Returns:
            List of (agent, confidence) tuples, sorted by confidence
        """
        # TODO: Implement routing logic
        # - Check each agent's can_handle() score
        # - Return agents above confidence threshold
        # - Sort by confidence descending
        pass

    def classify_query(self, query: str) -> AgentType:
        """
        Classify query to determine agent type

        Use keyword matching, embeddings, or small classifier model
        """
        # TODO: Implement
        pass
```

#### Part C: Workflow Orchestrator (25 points)

**Requirements**:
1. Create workflow execution engine
2. Support DAG (Directed Acyclic Graph) execution
3. Execute independent agents in parallel
4. Handle agent dependencies
5. Implement retry logic for failures
6. Aggregate results from multiple agents

**Starter Code**:
```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import asyncio

@dataclass
class WorkflowStep:
    """Single step in workflow"""
    agent: Agent
    input_data: AgentInput
    depends_on: List[str] = None  # IDs of steps this depends on
    step_id: str = None

class WorkflowOrchestrator:
    """
    Orchestrate multi-agent workflows

    Features:
        - DAG execution
        - Parallel execution where possible
        - Error handling and retries
        - Result aggregation
    """

    def __init__(self, max_retries: int = 3, timeout_seconds: int = 30):
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        self.execution_history = []

    async def execute_workflow(
        self,
        query: str,
        agents: List[Agent]
    ) -> Dict[str, Any]:
        """
        Execute complete workflow for a query

        Args:
            query: User query
            agents: List of agents to potentially use

        Returns:
            Aggregated results from all agents
        """
        # TODO: Implement workflow execution
        # 1. Create workflow steps from query
        # 2. Build dependency graph
        # 3. Execute in topological order
        # 4. Handle parallel execution
        # 5. Aggregate results
        pass

    async def execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> AgentOutput:
        """
        Execute single workflow step with retry logic

        Args:
            step: Workflow step to execute
            context: Results from previous steps

        Returns:
            Agent output
        """
        # TODO: Implement with retries
        pass

    def build_dependency_graph(
        self,
        steps: List[WorkflowStep]
    ) -> Dict[str, List[str]]:
        """
        Build DAG from workflow steps

        Returns:
            Adjacency list representation
        """
        # TODO: Implement
        pass

    def topological_sort(
        self,
        graph: Dict[str, List[str]]
    ) -> List[List[str]]:
        """
        Sort workflow steps for execution

        Returns:
            List of levels (steps at same level can run in parallel)
        """
        # TODO: Implement Kahn's algorithm or DFS
        pass

    async def execute_parallel(
        self,
        steps: List[WorkflowStep],
        context: Dict[str, Any]
    ) -> List[AgentOutput]:
        """
        Execute multiple steps in parallel

        Args:
            steps: Steps to execute (no dependencies between them)
            context: Shared context

        Returns:
            List of outputs
        """
        # TODO: Implement using asyncio.gather
        pass

    def aggregate_results(
        self,
        outputs: List[AgentOutput]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents

        Combine outputs intelligently based on:
            - Confidence scores
            - Agent types
            - Query intent
        """
        # TODO: Implement
        pass
```

#### Part D: Error Handling & Observability (15 points)

**Requirements**:
1. Graceful error handling
2. Retry logic with exponential backoff
3. Comprehensive logging
4. Execution tracing
5. Performance metrics

**Starter Code**:
```python
import logging
from datetime import datetime
import time

class WorkflowExecutionTracer:
    """Trace workflow execution for debugging"""

    def __init__(self):
        self.traces = []
        self.logger = logging.getLogger("WorkflowTracer")

    def start_trace(self, workflow_id: str, query: str):
        """Start tracing a workflow"""
        # TODO: Implement
        pass

    def log_step(
        self,
        workflow_id: str,
        step_id: str,
        agent_name: str,
        status: str,
        duration_ms: float
    ):
        """Log individual step execution"""
        # TODO: Implement
        pass

    def end_trace(self, workflow_id: str, total_duration_ms: float):
        """End workflow trace"""
        # TODO: Implement
        pass

    def get_trace(self, workflow_id: str) -> Dict:
        """Get execution trace for analysis"""
        # TODO: Implement
        pass

async def retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    Retry async function with exponential backoff

    Args:
        func: Async function to retry
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for each retry

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    # TODO: Implement
    pass
```

### Expected Deliverable

Create a complete multi-agent system that:

1. **Defines multiple specialized agents** with clear responsibilities
2. **Routes queries intelligently** to appropriate agents
3. **Executes workflows** with dependency management
4. **Handles errors** and retries gracefully
5. **Provides observability** through logging and tracing

### Example Usage:

```python
# Initialize agents
rag_agent = RAGAgent()
sql_agent = SQLAgent("postgresql://localhost/db")
code_agent = CodeAgent()

# Create router
router = QueryRouter(agents=[rag_agent, sql_agent, code_agent])

# Create orchestrator
orchestrator = WorkflowOrchestrator()

# Execute query
query = "What were our top 3 products by revenue last quarter? Show me the SQL query and results."

# This query requires:
# 1. SQL Agent to generate and execute query
# 2. RAG Agent to provide context/explanation
# 3. Results aggregation

result = await orchestrator.execute_workflow(query, agents=[sql_agent, rag_agent])
print(result)
```

---

## Evaluation Criteria

Your solutions will be evaluated based on:

### 1. Correctness (40%)

- **Functionality**: Does it solve the problem?
  - All requirements implemented
  - Edge cases handled
  - No critical bugs

- **Robustness**: Error handling
  - Graceful failures
  - Informative error messages
  - Input validation

### 2. Code Quality (30%)

- **Readability**:
  - Clear variable/function names
  - Logical code organization
  - Consistent formatting

- **Documentation**:
  - Comprehensive docstrings
  - Inline comments for complex logic
  - API documentation

- **Best Practices**:
  - Type hints throughout
  - DRY principle
  - SOLID principles
  - Pythonic code

### 3. Performance (15%)

- **Time Complexity**:
  - Efficient algorithms
  - No unnecessary operations
  - Proper data structure selection

- **Space Complexity**:
  - Memory-efficient solutions
  - No memory leaks
  - Resource cleanup

- **Scalability**:
  - Handles large inputs
  - Concurrent request support
  - Caching strategies

### 4. Production Readiness (15%)

- **Observability**:
  - Comprehensive logging
  - Metrics tracking
  - Error monitoring

- **Security**:
  - Input sanitization
  - Authentication/authorization considerations
  - Secret management

- **Testing**:
  - Unit tests included
  - Integration test considerations
  - Test coverage

- **Deployment**:
  - Docker containerization
  - Environment configuration
  - Documentation

---

## Detailed Grading Rubric

### Excellent (90-100%)

**Characteristics**:
- ✅ All requirements implemented flawlessly
- ✅ Production-ready code with comprehensive testing
- ✅ Advanced optimizations and design patterns
- ✅ Exceptional error handling and edge case coverage
- ✅ Professional-grade documentation
- ✅ Demonstrates deep understanding of concepts

**Example Indicators**:
- Uses design patterns appropriately (Factory, Strategy, Observer)
- Implements comprehensive monitoring and alerting
- Includes performance benchmarks
- Has >80% test coverage
- Dockerized with CI/CD considerations
- Implements caching and optimization strategies

### Good (70-89%)

**Characteristics**:
- ✅ All core requirements met
- ✅ Good code quality with minor issues
- ✅ Basic error handling present
- ✅ Decent documentation
- ⚠️ Some optimization opportunities missed
- ⚠️ Limited test coverage

**Example Indicators**:
- Clean, readable code
- Type hints used consistently
- Basic logging implemented
- Some tests included
- Handles common edge cases
- Could benefit from refactoring

### Satisfactory (50-69%)

**Characteristics**:
- ✅ Basic functionality works
- ⚠️ Code quality issues present
- ⚠️ Minimal error handling
- ⚠️ Limited documentation
- ❌ Missing some requirements
- ❌ No tests included

**Example Indicators**:
- Works for happy path
- Inconsistent code style
- Minimal comments
- No type hints
- Crashes on edge cases
- Inefficient algorithms

### Needs Improvement (<50%)

**Characteristics**:
- ❌ Incomplete implementation
- ❌ Significant bugs present
- ❌ Poor code organization
- ❌ No error handling
- ❌ Missing key requirements
- ❌ Doesn't run or compile

---

## Common Pitfalls to Avoid

### RAG System (Section 1)

❌ **Don't**:
- Forget to chunk large documents
- Ignore character encoding issues
- Hard-code API keys in code
- Skip input validation
- Return raw LLM output without sources
- Ignore memory management for large files

✅ **Do**:
- Implement smart chunking with overlap
- Handle multiple encodings gracefully
- Use environment variables for secrets
- Validate all inputs thoroughly
- Always return source citations
- Stream large file processing

### Cache Implementation (Section 2)

❌ **Don't**:
- Use nested locks (deadlock risk)
- Forget to check TTL on get()
- Leak memory by not cleaning expired entries
- Use global variables without locks
- Serialize everything with pickle (security risk)

✅ **Do**:
- Use single lock or fine-grained locking
- Implement lazy expiration checking
- Consider background cleanup thread
- Protect all shared state with locks
- Validate deserialized objects

### Multi-Agent System (Section 3)

❌ **Don't**:
- Execute agents sequentially when parallel is possible
- Ignore agent failures (cascade failures)
- Hard-code agent selection logic
- Skip dependency validation (circular dependencies)
- Return raw errors to users

✅ **Do**:
- Use asyncio.gather() for parallel execution
- Implement circuit breakers for failing agents
- Use confidence scores for routing
- Validate DAG before execution
- Return user-friendly error messages

---

## Tips for Success

### Time Management

1. **Read requirements carefully** (5 min per section)
2. **Plan your approach** (10 min per section)
3. **Implement core functionality first** (50-60% of time)
4. **Add error handling and edge cases** (20-30% of time)
5. **Polish and document** (10-20% of time)

### Coding Strategy

1. **Start simple**: Get basic version working first
2. **Test as you go**: Don't wait until the end
3. **Refactor iteratively**: Improve incrementally
4. **Comment complex logic**: Help reviewers understand
5. **Use descriptive names**: Self-documenting code

### What Impresses Interviewers

1. **Clean separation of concerns**: Single responsibility principle
2. **Defensive programming**: Input validation, error handling
3. **Performance awareness**: Big O notation understanding
4. **Production mindset**: Logging, monitoring, deployment
5. **Testing culture**: Unit tests, edge cases
6. **Documentation**: Clear docstrings, README

### Resources Provided

You will have access to:
- Python 3.9+ with standard library
- FastAPI, LangChain, ChromaDB, OpenAI SDK
- numpy, pandas, pytest
- Internet access for documentation
- Code editor of your choice

---

## Submission Instructions

1. **Code Organization**:
   ```
   elgo-interview/
   ├── section1_rag/
   │   ├── solution.py
   │   ├── test_solution.py
   │   └── README.md
   ├── section2_cache/
   │   ├── solution.py
   │   ├── test_solution.py
   │   └── README.md
   ├── section3_multiagent/
   │   ├── solution.py
   │   ├── test_solution.py
   │   └── README.md
   ├── requirements.txt
   └── README.md
   ```

2. **Each solution should include**:
   - Working Python code
   - Unit tests demonstrating functionality
   - README with setup and usage instructions
   - Comments explaining complex logic

3. **Submit via**:
   - GitHub repository (preferred)
   - ZIP file via email
   - Shared drive link

---

## Interview Day Logistics

### Before the Interview

- [ ] Test your development environment
- [ ] Ensure Python 3.9+ is installed
- [ ] Install required dependencies
- [ ] Set up OpenAI API key
- [ ] Test internet connectivity
- [ ] Have documentation bookmarks ready

### During the Interview

- You can ask clarifying questions
- Use Google/Stack Overflow/documentation
- Take short breaks between sections
- Communicate your thought process
- Start with working solution, then optimize

### After Each Section

- Quick 5-minute break
- Stretch and hydrate
- Clear your mind before next section

---

## Questions?

If you have any questions before or during the interview:

- **Technical Questions**: Ask the interviewer directly
- **Clarifications**: Don't hesitate to ask
- **Stuck on a problem**: Explain your approach and ask for hints
- **Time management**: Ask for time remaining

---

## Good Luck!

Remember: We're not just evaluating your ability to write code, but also your problem-solving approach, communication skills, and potential to grow with our team at ELGO AI.

**Focus on**:
- Writing clean, maintainable code
- Solving problems methodically
- Communicating your thought process
- Demonstrating production awareness

We're excited to see what you build!

---

**ELGO AI Engineering Team**
