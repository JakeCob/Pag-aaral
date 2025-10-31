"""
ELGO AI - Advanced RAG System Solution
Section 1: Complete Implementation

Features:
- Multi-format document processing (txt, pdf, json, csv)
- Document versioning
- Hybrid search (semantic + BM25)
- Re-ranking with cross-encoder
- Conversation memory
- Faithfulness evaluation
- Query caching
- Metrics tracking

Author: Reference Solution
"""

import os
import io
import json
import csv
import hashlib
import logging
import pickle
import time
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings

# Document processing
import PyPDF2

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Hybrid search and re-ranking
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ELGO Advanced RAG System",
    description="Production-grade RAG with multi-format support, versioning, and advanced retrieval",
    version="2.0.0"
)

# Initialize ChromaDB
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db_advanced"
))


# ============================================================================
# MODELS
# ============================================================================

class DocumentFormat(str, Enum):
    """Supported document formats"""
    TXT = "txt"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"


class DocumentMetadata(BaseModel):
    """Document metadata"""
    doc_id: str
    version: int
    filename: str
    format: DocumentFormat
    size_bytes: int
    chunks_created: int
    uploaded_at: datetime
    content_hash: str


class Query(BaseModel):
    """Query request"""
    question: str
    doc_id: str
    session_id: Optional[str] = "default"
    max_chunks: int = Field(default=5, ge=1, le=20)
    use_hybrid_search: bool = True
    use_reranking: bool = True


class UploadResponse(BaseModel):
    """Upload response"""
    doc_id: str
    version: int
    message: str
    chunks_created: int
    format: DocumentFormat


class QueryResponse(BaseModel):
    """Query response"""
    answer: str
    source_chunks: List[str]
    confidence: float
    faithfulness_score: float
    cached: bool
    latency_ms: float
    metadata: Dict[str, Any] = {}


class ConversationMessage(BaseModel):
    """Conversation message"""
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime


# ============================================================================
# PART A: MULTI-FORMAT DOCUMENT PROCESSING
# ============================================================================

class MultiFormatDocumentProcessor:
    """Process multiple document formats"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize processor

        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.embeddings = OpenAIEmbeddings()
        logger.info(f"MultiFormatDocumentProcessor initialized: chunk_size={chunk_size}")

    def process_pdf(self, file_content: bytes) -> str:
        """
        Extract text from PDF

        Args:
            file_content: PDF file bytes

        Returns:
            Extracted text
        """
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()

            logger.info(f"Extracted text from {len(pdf_reader.pages)} PDF pages")
            return text.strip()

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")

    def process_json(self, file_content: bytes) -> str:
        """
        Extract and format JSON data

        Args:
            file_content: JSON file bytes

        Returns:
            Formatted text representation
        """
        try:
            data = json.loads(file_content.decode('utf-8'))

            # Convert JSON to readable text format
            def json_to_text(obj, prefix=""):
                lines = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            lines.append(f"{prefix}{key}:")
                            lines.extend(json_to_text(value, prefix + "  "))
                        else:
                            lines.append(f"{prefix}{key}: {value}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            lines.append(f"{prefix}Item {i+1}:")
                            lines.extend(json_to_text(item, prefix + "  "))
                        else:
                            lines.append(f"{prefix}- {item}")
                else:
                    lines.append(f"{prefix}{obj}")
                return lines

            text = "\n".join(json_to_text(data))
            logger.info(f"Processed JSON with {len(text)} characters")
            return text

        except json.JSONDecodeError as e:
            logger.error(f"Error processing JSON: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")

    def process_csv(self, file_content: bytes) -> str:
        """
        Extract and format CSV data

        Args:
            file_content: CSV file bytes

        Returns:
            Formatted text representation
        """
        try:
            csv_file = io.StringIO(file_content.decode('utf-8'))
            csv_reader = csv.DictReader(csv_file)

            lines = []
            headers = csv_reader.fieldnames
            lines.append(f"CSV Table with columns: {', '.join(headers)}\n")

            for i, row in enumerate(csv_reader):
                lines.append(f"Row {i+1}:")
                for key, value in row.items():
                    lines.append(f"  {key}: {value}")
                lines.append("")  # Blank line between rows

            text = "\n".join(lines)
            logger.info(f"Processed CSV with {i+1} rows")
            return text

        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    def process_text(self, file_content: bytes) -> str:
        """
        Process plain text file

        Args:
            file_content: Text file bytes

        Returns:
            Decoded text
        """
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return file_content.decode(encoding)
                except:
                    continue
            raise HTTPException(status_code=400, detail="Unable to decode text file")

    def process_document(
        self,
        file_content: bytes,
        filename: str,
        doc_id: str,
        version: int
    ) -> Dict:
        """
        Process document based on format

        Args:
            file_content: File bytes
            filename: Original filename
            doc_id: Document ID
            version: Version number

        Returns:
            Processing result with chunks
        """
        # Determine format from filename
        extension = filename.lower().split('.')[-1]

        try:
            format_enum = DocumentFormat(extension)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {extension}. Supported: txt, pdf, json, csv"
            )

        # Extract text based on format
        if format_enum == DocumentFormat.PDF:
            text = self.process_pdf(file_content)
        elif format_enum == DocumentFormat.JSON:
            text = self.process_json(file_content)
        elif format_enum == DocumentFormat.CSV:
            text = self.process_csv(file_content)
        else:  # TXT
            text = self.process_text(file_content)

        # Chunk the text
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks for {filename}")

        # Create metadata for each chunk
        metadatas = [
            {
                "doc_id": doc_id,
                "version": version,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "format": extension
            }
            for i in range(len(chunks))
        ]

        # Store in ChromaDB
        collection_name = f"doc_{doc_id}_v{version}"
        collection = client.get_or_create_collection(name=collection_name)

        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=[f"{doc_id}_v{version}_chunk_{i}" for i in range(len(chunks))]
        )

        return {
            "doc_id": doc_id,
            "version": version,
            "chunks_created": len(chunks),
            "format": format_enum,
            "chunks": chunks  # Return for BM25 indexing
        }


class DocumentVersionManager:
    """Manage document versions"""

    def __init__(self):
        """Initialize version manager"""
        self.versions = {}  # doc_id -> List[DocumentMetadata]
        logger.info("DocumentVersionManager initialized")

    def add_version(
        self,
        doc_id: str,
        filename: str,
        format: DocumentFormat,
        size_bytes: int,
        chunks_created: int,
        content_hash: str
    ) -> int:
        """
        Add new version and return version number

        Args:
            doc_id: Document ID
            filename: Original filename
            format: Document format
            size_bytes: File size
            chunks_created: Number of chunks
            content_hash: Hash of content

        Returns:
            Version number
        """
        if doc_id not in self.versions:
            self.versions[doc_id] = []

        version = len(self.versions[doc_id]) + 1

        metadata = DocumentMetadata(
            doc_id=doc_id,
            version=version,
            filename=filename,
            format=format,
            size_bytes=size_bytes,
            chunks_created=chunks_created,
            uploaded_at=datetime.now(),
            content_hash=content_hash
        )

        self.versions[doc_id].append(metadata)
        logger.info(f"Added version {version} for document {doc_id}")

        return version

    def get_versions(self, doc_id: str) -> List[DocumentMetadata]:
        """
        Get all versions of a document

        Args:
            doc_id: Document ID

        Returns:
            List of metadata for all versions
        """
        return self.versions.get(doc_id, [])

    def get_latest_version(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get latest version of document"""
        versions = self.get_versions(doc_id)
        return versions[-1] if versions else None

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete all versions of a document

        Args:
            doc_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        if doc_id not in self.versions:
            return False

        # Delete all versions from ChromaDB
        for version_meta in self.versions[doc_id]:
            collection_name = f"doc_{doc_id}_v{version_meta.version}"
            try:
                client.delete_collection(name=collection_name)
                logger.info(f"Deleted collection {collection_name}")
            except Exception as e:
                logger.warning(f"Failed to delete collection {collection_name}: {e}")

        # Remove from version tracking
        del self.versions[doc_id]
        logger.info(f"Deleted all versions of document {doc_id}")

        return True


# ============================================================================
# PART B: HYBRID SEARCH WITH RE-RANKING
# ============================================================================

class HybridRetriever:
    """Implement hybrid search with semantic + keyword"""

    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid retriever

        Args:
            alpha: Weight for semantic search (1-alpha for BM25)
        """
        self.alpha = alpha
        self.bm25_indexes = {}  # doc_id_version -> BM25Okapi
        self.doc_chunks = {}  # doc_id_version -> List[str]
        self.embeddings = OpenAIEmbeddings()
        logger.info(f"HybridRetriever initialized with alpha={alpha}")

    def index_documents(self, documents: List[str], doc_id: str, version: int):
        """
        Index documents for BM25

        Args:
            documents: List of document chunks
            doc_id: Document ID
            version: Version number
        """
        key = f"{doc_id}_v{version}"

        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25_indexes[key] = BM25Okapi(tokenized_docs)
        self.doc_chunks[key] = documents

        logger.info(f"Indexed {len(documents)} documents for BM25: {key}")

    def retrieve(
        self,
        query: str,
        doc_id: str,
        version: int,
        k: int = 10
    ) -> List[Dict]:
        """
        Retrieve using hybrid search

        Args:
            query: Search query
            doc_id: Document ID
            version: Version number
            k: Number of results to return

        Returns:
            List of {chunk, score, method} dicts
        """
        key = f"{doc_id}_v{version}"

        if key not in self.bm25_indexes:
            logger.warning(f"No BM25 index found for {key}, using semantic only")
            return self._semantic_search(query, doc_id, version, k)

        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_indexes[key].get_scores(tokenized_query)

        # Semantic search scores
        collection = client.get_collection(name=f"doc_{doc_id}_v{version}")
        semantic_results = collection.query(
            query_texts=[query],
            n_results=min(k, len(self.doc_chunks[key]))
        )

        # Normalize scores
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        # Create semantic score mapping
        semantic_scores = {}
        if semantic_results['documents'] and semantic_results['documents'][0]:
            for doc, distance in zip(
                semantic_results['documents'][0],
                semantic_results['distances'][0]
            ):
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1 / (1 + distance)
                semantic_scores[doc] = similarity

        # Combine scores
        combined_results = []
        for i, chunk in enumerate(self.doc_chunks[key]):
            bm25_score = bm25_scores_norm[i]
            semantic_score = semantic_scores.get(chunk, 0.0)

            # Normalize semantic scores
            if semantic_scores:
                max_sem_score = max(semantic_scores.values())
                if max_sem_score > 0:
                    semantic_score = semantic_score / max_sem_score

            # Hybrid score
            hybrid_score = (
                self.alpha * semantic_score +
                (1 - self.alpha) * bm25_score
            )

            combined_results.append({
                "chunk": chunk,
                "score": hybrid_score,
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
                "method": "hybrid"
            })

        # Sort by hybrid score and return top k
        combined_results.sort(key=lambda x: x['score'], reverse=True)

        return combined_results[:k]

    def _semantic_search(
        self,
        query: str,
        doc_id: str,
        version: int,
        k: int
    ) -> List[Dict]:
        """Fallback to semantic-only search"""
        collection = client.get_collection(name=f"doc_{doc_id}_v{version}")
        results = collection.query(query_texts=[query], n_results=k)

        return [
            {
                "chunk": doc,
                "score": 1 / (1 + dist),
                "method": "semantic"
            }
            for doc, dist in zip(results['documents'][0], results['distances'][0])
        ]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)


class Reranker:
    """Re-rank results using cross-encoder"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker

        Args:
            model_name: Cross-encoder model name
        """
        self.model = CrossEncoder(model_name)
        logger.info(f"Reranker initialized with model: {model_name}")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Re-rank documents and return top_k

        Args:
            query: Search query
            documents: List of document dicts with 'chunk' key
            top_k: Number of top results to return

        Returns:
            List of {chunk, score, rerank_score} sorted by rerank score
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc['chunk']] for doc in documents]

        # Get reranking scores
        rerank_scores = self.model.predict(pairs)

        # Add rerank scores to documents
        for doc, score in zip(documents, rerank_scores):
            doc['rerank_score'] = float(score)

        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        logger.info(f"Reranked {len(documents)} documents, returning top {top_k}")

        return reranked[:top_k]


# ============================================================================
# PART C: CONVERSATION MEMORY & CONTEXT
# ============================================================================

class ConversationMemory:
    """Manage conversation history and context"""

    def __init__(self, max_history: int = 5, max_tokens: int = 2000):
        """
        Initialize conversation memory

        Args:
            max_history: Maximum conversation turns to keep
            max_tokens: Maximum total tokens in context
        """
        self.sessions = {}  # session_id -> deque of ConversationMessage
        self.max_history = max_history
        self.max_tokens = max_tokens
        logger.info(f"ConversationMemory initialized: max_history={max_history}, max_tokens={max_tokens}")

    def add_message(self, session_id: str, role: str, content: str):
        """
        Add message to conversation history

        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_history)

        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        )

        self.sessions[session_id].append(message)
        logger.debug(f"Added {role} message to session {session_id}")

    def get_context(self, session_id: str) -> str:
        """
        Get formatted conversation context with token limiting

        Args:
            session_id: Session identifier

        Returns:
            Formatted context string
        """
        if session_id not in self.sessions:
            return ""

        messages = list(self.sessions[session_id])
        context_lines = []
        total_tokens = 0

        # Add messages from most recent, respecting token limit
        for message in reversed(messages):
            # Rough token estimation (1 token ≈ 4 characters)
            message_tokens = len(message.content) // 4

            if total_tokens + message_tokens > self.max_tokens:
                break

            context_lines.insert(0, f"{message.role}: {message.content}")
            total_tokens += message_tokens

        context = "\n".join(context_lines)
        logger.debug(f"Retrieved context for session {session_id}: ~{total_tokens} tokens")

        return context

    def clear_session(self, session_id: str):
        """
        Clear conversation history for session

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")


class ContextAwareQueryEngine:
    """Query engine with conversation context"""

    def __init__(self, memory: ConversationMemory):
        """
        Initialize context-aware query engine

        Args:
            memory: Conversation memory instance
        """
        self.memory = memory
        self.llm = OpenAI(temperature=0.7)
        logger.info("ContextAwareQueryEngine initialized")

    def query_with_context(
        self,
        question: str,
        doc_id: str,
        version: int,
        session_id: str,
        retrieved_chunks: List[str]
    ) -> str:
        """
        Answer question using conversation context

        Args:
            question: User question
            doc_id: Document ID
            version: Document version
            session_id: Session ID
            retrieved_chunks: Retrieved document chunks

        Returns:
            Generated answer
        """
        # Get conversation context
        conversation_context = self.memory.get_context(session_id)

        # Build prompt with context
        prompt = self._build_prompt(question, conversation_context, retrieved_chunks)

        # Generate answer
        answer = self.llm(prompt)

        # Store in memory
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", answer)

        return answer

    def _build_prompt(
        self,
        question: str,
        conversation_context: str,
        chunks: List[str]
    ) -> str:
        """Build prompt with context and chunks"""
        prompt_parts = []

        if conversation_context:
            prompt_parts.append("Previous conversation:")
            prompt_parts.append(conversation_context)
            prompt_parts.append("")

        prompt_parts.append("Relevant information from documents:")
        for i, chunk in enumerate(chunks, 1):
            prompt_parts.append(f"[{i}] {chunk}")

        prompt_parts.append("")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("")
        prompt_parts.append(
            "Answer the question based on the provided information. "
            "If the conversation context helps understand the question, use it. "
            "If the information is not in the documents, say so."
        )

        return "\n".join(prompt_parts)


# ============================================================================
# PART D: EVALUATION & MONITORING
# ============================================================================

class FaithfulnessEvaluator:
    """Evaluate answer faithfulness to sources"""

    def __init__(self):
        """Initialize faithfulness evaluator"""
        self.llm = OpenAI(temperature=0.0)  # Low temp for consistency
        logger.info("FaithfulnessEvaluator initialized")

    def evaluate(self, question: str, answer: str, sources: List[str]) -> float:
        """
        Evaluate faithfulness score using LLM

        Args:
            question: Original question
            answer: Generated answer
            sources: Source chunks used

        Returns:
            Faithfulness score between 0.0 and 1.0
        """
        prompt = f"""
You are an AI evaluation system. Your task is to determine if the answer is faithful to the source documents.

Question: {question}

Answer: {answer}

Source Documents:
{chr(10).join(f"[{i+1}] {src}" for i, src in enumerate(sources))}

Evaluate the answer on a scale from 0.0 to 1.0:
- 1.0 = Answer is completely supported by sources
- 0.7-0.9 = Answer is mostly supported with minor inferences
- 0.4-0.6 = Answer contains some unsupported claims
- 0.0-0.3 = Answer is mostly unsupported or contradicts sources

Return ONLY a number between 0.0 and 1.0, nothing else.
"""

        try:
            response = self.llm(prompt).strip()
            score = float(response)
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            logger.debug(f"Faithfulness score: {score}")
            return score
        except ValueError:
            logger.warning(f"Failed to parse faithfulness score: {response}")
            return 0.5  # Default to medium confidence


class QueryCache:
    """Cache for frequent queries"""

    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize query cache

        Args:
            ttl_seconds: Time to live in seconds
        """
        self.cache = {}  # query_hash -> (response, expiry)
        self.ttl = ttl_seconds
        logger.info(f"QueryCache initialized with TTL={ttl_seconds}s")

    def _get_cache_key(self, query: str, doc_id: str, version: int) -> str:
        """Generate cache key from query and doc"""
        key_str = f"{query}:{doc_id}:v{version}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, doc_id: str, version: int) -> Optional[Dict]:
        """
        Get cached response if available and not expired

        Args:
            query: Query string
            doc_id: Document ID
            version: Document version

        Returns:
            Cached response or None
        """
        cache_key = self._get_cache_key(query, doc_id, version)

        if cache_key in self.cache:
            response, expiry = self.cache[cache_key]

            if datetime.now() < expiry:
                logger.info(f"Cache HIT for query: {query[:50]}...")
                return response
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
                logger.debug(f"Cache entry expired and removed")

        logger.debug(f"Cache MISS for query: {query[:50]}...")
        return None

    def put(self, query: str, doc_id: str, version: int, response: Dict):
        """
        Cache a response

        Args:
            query: Query string
            doc_id: Document ID
            version: Document version
            response: Response to cache
        """
        cache_key = self._get_cache_key(query, doc_id, version)
        expiry = datetime.now() + timedelta(seconds=self.ttl)

        self.cache[cache_key] = (response, expiry)
        logger.debug(f"Cached response for query: {query[:50]}...")

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        logger.info("Cache cleared")


class MetricsTracker:
    """Track system metrics"""

    def __init__(self):
        """Initialize metrics tracker"""
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency_ms": 0.0,
            "total_chunks_retrieved": 0,
            "total_faithfulness_score": 0.0
        }
        logger.info("MetricsTracker initialized")

    def record_query(
        self,
        latency_ms: float,
        cached: bool,
        chunks: int,
        faithfulness: float
    ):
        """
        Record query metrics

        Args:
            latency_ms: Query latency in milliseconds
            cached: Whether response was cached
            chunks: Number of chunks retrieved
            faithfulness: Faithfulness score
        """
        self.metrics["total_queries"] += 1

        if cached:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

        self.metrics["total_latency_ms"] += latency_ms
        self.metrics["total_chunks_retrieved"] += chunks
        self.metrics["total_faithfulness_score"] += faithfulness

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics

        Returns:
            Dictionary with calculated metrics
        """
        total_queries = self.metrics["total_queries"]

        if total_queries == 0:
            return {**self.metrics, "avg_latency_ms": 0.0, "cache_hit_rate": 0.0}

        return {
            **self.metrics,
            "avg_latency_ms": self.metrics["total_latency_ms"] / total_queries,
            "avg_chunks_per_query": self.metrics["total_chunks_retrieved"] / total_queries,
            "avg_faithfulness_score": self.metrics["total_faithfulness_score"] / total_queries,
            "cache_hit_rate": self.metrics["cache_hits"] / total_queries
        }


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

doc_processor = MultiFormatDocumentProcessor()
version_manager = DocumentVersionManager()
hybrid_retriever = HybridRetriever(alpha=0.6)
reranker = Reranker()
conversation_memory = ConversationMemory()
context_engine = ContextAwareQueryEngine(conversation_memory)
faithfulness_evaluator = FaithfulnessEvaluator()
query_cache = QueryCache(ttl_seconds=1800)  # 30 minutes
metrics_tracker = MetricsTracker()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document

    Supports: txt, pdf, json, csv
    Automatically versions documents
    """
    try:
        logger.info(f"Upload request: {file.filename}")

        # Read file content
        content = await file.read()

        # Generate document ID from filename (could use content hash for duplicates)
        base_name = file.filename.rsplit('.', 1)[0]
        doc_id = hashlib.md5(base_name.encode()).hexdigest()[:8]

        # Calculate content hash
        content_hash = hashlib.sha256(content).hexdigest()

        # Process document
        result = doc_processor.process_document(content, file.filename, doc_id, 1)

        # Add version
        version = version_manager.add_version(
            doc_id=doc_id,
            filename=file.filename,
            format=result['format'],
            size_bytes=len(content),
            chunks_created=result['chunks_created'],
            content_hash=content_hash
        )

        # Index for BM25
        hybrid_retriever.index_documents(
            result['chunks'],
            doc_id,
            version
        )

        return UploadResponse(
            doc_id=doc_id,
            version=version,
            message=f"Document {file.filename} processed successfully",
            chunks_created=result['chunks_created'],
            format=result['format']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_document(query: Query):
    """
    Query a processed document with advanced features:
    - Hybrid search (semantic + BM25)
    - Re-ranking
    - Conversation context
    - Faithfulness evaluation
    - Caching
    """
    start_time = time.time()

    try:
        # Check if document exists
        doc_meta = version_manager.get_latest_version(query.doc_id)
        if not doc_meta:
            raise HTTPException(status_code=404, detail=f"Document {query.doc_id} not found")

        version = doc_meta.version

        # Check cache
        cached_response = query_cache.get(query.question, query.doc_id, version)
        if cached_response:
            latency_ms = (time.time() - start_time) * 1000
            metrics_tracker.record_query(
                latency_ms=latency_ms,
                cached=True,
                chunks=len(cached_response['source_chunks']),
                faithfulness=cached_response['faithfulness_score']
            )
            cached_response['cached'] = True
            cached_response['latency_ms'] = latency_ms
            return QueryResponse(**cached_response)

        # Retrieve chunks
        if query.use_hybrid_search:
            retrieved_docs = hybrid_retriever.retrieve(
                query.question,
                query.doc_id,
                version,
                k=query.max_chunks * 2  # Get more for reranking
            )
        else:
            # Semantic only
            collection = client.get_collection(name=f"doc_{query.doc_id}_v{version}")
            results = collection.query(query_texts=[query.question], n_results=query.max_chunks)
            retrieved_docs = [
                {"chunk": doc, "score": 1 / (1 + dist)}
                for doc, dist in zip(results['documents'][0], results['distances'][0])
            ]

        # Re-rank if enabled
        if query.use_reranking and len(retrieved_docs) > 0:
            retrieved_docs = reranker.rerank(
                query.question,
                retrieved_docs,
                top_k=query.max_chunks
            )
        else:
            retrieved_docs = retrieved_docs[:query.max_chunks]

        # Extract chunks
        source_chunks = [doc['chunk'] for doc in retrieved_docs]

        # Generate answer with context
        answer = context_engine.query_with_context(
            question=query.question,
            doc_id=query.doc_id,
            version=version,
            session_id=query.session_id,
            retrieved_chunks=source_chunks
        )

        # Evaluate faithfulness
        faithfulness_score = faithfulness_evaluator.evaluate(
            query.question,
            answer,
            source_chunks
        )

        # Calculate confidence from retrieval scores
        avg_score = sum(doc.get('score', 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
        confidence = min(avg_score, 1.0)

        # Prepare response
        response_data = {
            "answer": answer,
            "source_chunks": source_chunks,
            "confidence": confidence,
            "faithfulness_score": faithfulness_score,
            "cached": False,
            "latency_ms": 0.0,  # Will be updated
            "metadata": {
                "chunks_used": len(source_chunks),
                "hybrid_search": query.use_hybrid_search,
                "reranked": query.use_reranking,
                "session_id": query.session_id
            }
        }

        # Cache response
        query_cache.put(query.question, query.doc_id, version, response_data)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        response_data['latency_ms'] = latency_ms

        # Record metrics
        metrics_tracker.record_query(
            latency_ms=latency_ms,
            cached=False,
            chunks=len(source_chunks),
            faithfulness=faithfulness_score
        )

        return QueryResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{doc_id}/versions")
async def get_document_versions(doc_id: str):
    """Get all versions of a document"""
    versions = version_manager.get_versions(doc_id)

    if not versions:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    return {
        "doc_id": doc_id,
        "total_versions": len(versions),
        "versions": [v.dict() for v in versions]
    }


@app.get("/documents/{doc_id}/metadata")
async def get_document_metadata(doc_id: str):
    """Get document metadata"""
    latest = version_manager.get_latest_version(doc_id)

    if not latest:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    return latest.dict()


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete all versions of a document"""
    success = version_manager.delete_document(doc_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    return {"message": f"Document {doc_id} and all versions deleted successfully"}


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session"""
    conversation_memory.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@app.get("/metrics")
async def get_metrics():
    """Get system metrics and statistics"""
    return metrics_tracker.get_stats()


@app.post("/cache/clear")
async def clear_cache():
    """Clear query cache"""
    query_cache.clear()
    return {"message": "Cache cleared successfully"}


@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    all_docs = {}

    for doc_id, versions in version_manager.versions.items():
        latest = versions[-1]
        all_docs[doc_id] = {
            "doc_id": doc_id,
            "filename": latest.filename,
            "format": latest.format,
            "total_versions": len(versions),
            "latest_version": latest.version,
            "last_updated": latest.uploaded_at.isoformat()
        }

    return {
        "total_documents": len(all_docs),
        "documents": all_docs
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ELGO Advanced RAG System",
        "version": "2.0.0",
        "features": [
            "multi-format documents",
            "versioning",
            "hybrid search",
            "re-ranking",
            "conversation memory",
            "faithfulness evaluation",
            "caching",
            "metrics"
        ]
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
