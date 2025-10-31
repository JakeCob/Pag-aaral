# RAG Architecture Guide
**Reference Guide for ELGO AI Interview Preparation**

Based on: `section1_rag_solution.py`

---

## Table of Contents
1. [Overview](#overview)
2. [Document Processing Pipeline](#document-processing-pipeline)
3. [Hybrid Search Implementation](#hybrid-search-implementation)
4. [Re-ranking with Cross-Encoders](#re-ranking-with-cross-encoders)
5. [Conversation Memory Management](#conversation-memory-management)
6. [Evaluation & Monitoring](#evaluation--monitoring)
7. [Production Patterns](#production-patterns)
8. [Common Pitfalls](#common-pitfalls)
9. [Code Templates](#code-templates)

---

## Overview

### What is RAG?
**Retrieval-Augmented Generation (RAG)** combines:
- **Retrieval**: Finding relevant documents from a knowledge base
- **Augmentation**: Adding retrieved context to the prompt
- **Generation**: Using an LLM to generate answers based on context

### RAG Pipeline Architecture
```
User Query → Retrieval → Re-ranking → Context Building → LLM → Answer
               ↓                          ↓
         Vector Search            Conversation Memory
         + BM25 Search                   ↓
                                    Evaluation
```

### Key Components
1. **Document Processor** - Multi-format ingestion (txt, pdf, json, csv)
2. **Vector Store** - ChromaDB with embeddings
3. **Hybrid Retriever** - Semantic + keyword search
4. **Re-ranker** - Cross-encoder for precision
5. **Memory Manager** - Session-based conversation history
6. **Query Engine** - Context-aware answer generation
7. **Evaluator** - Faithfulness scoring
8. **Cache** - Query response caching

---

## Document Processing Pipeline

### Step 1: Multi-Format Support

#### Text Extraction by Format

**PDF Processing** (`section1_rag_solution.py:158-182`)
```python
def process_pdf(self, file_content: bytes) -> str:
    """Extract text from PDF with page markers"""
    pdf_file = io.BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        text += f"\n--- Page {page_num + 1} ---\n"
        text += page.extract_text()

    return text.strip()
```

**Key Pattern**: Page markers help with source attribution later.

**JSON Processing** (`section1_rag_solution.py:184-224`)
```python
def process_json(self, file_content: bytes) -> str:
    """Convert JSON to readable text format"""
    data = json.loads(file_content.decode('utf-8'))

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
                lines.append(f"{prefix}Item {i+1}:")
                lines.extend(json_to_text(item, prefix + "  "))
        return lines

    return "\n".join(json_to_text(data))
```

**Key Pattern**: Recursive flattening preserves structure while making text searchable.

**CSV Processing** (`section1_rag_solution.py:226-256`)
```python
def process_csv(self, file_content: bytes) -> str:
    """Convert CSV to row-by-row text"""
    csv_file = io.StringIO(file_content.decode('utf-8'))
    csv_reader = csv.DictReader(csv_file)

    lines = [f"CSV Table with columns: {', '.join(csv_reader.fieldnames)}\n"]

    for i, row in enumerate(csv_reader):
        lines.append(f"Row {i+1}:")
        for key, value in row.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    return "\n".join(lines)
```

**Key Pattern**: Structured format maintains column-value relationships.

### Step 2: Text Chunking

**Recursive Character Splitter** (`section1_rag_solution.py:149-154`)
```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,           # Characters per chunk
    chunk_overlap=50,         # Overlap between chunks (10%)
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]  # Priority order
)
```

**Why Recursive?**
- Tries to split on `\n\n` (paragraphs) first
- Falls back to `\n` (lines), then `.` (sentences), then ` ` (words)
- Maintains semantic coherence

**Chunking Parameters**:
- **chunk_size=500**: Good balance for embeddings (not too long/short)
- **chunk_overlap=50**: Prevents splitting mid-concept (10% overlap is standard)

### Step 3: Embedding & Storage

**Vector Storage** (`section1_rag_solution.py:336-343`)
```python
collection = client.get_or_create_collection(name=f"doc_{doc_id}_v{version}")

collection.add(
    documents=chunks,
    metadatas=[{
        "doc_id": doc_id,
        "version": version,
        "chunk_index": i,
        "total_chunks": len(chunks),
        "format": extension
    } for i in range(len(chunks))],
    ids=[f"{doc_id}_v{version}_chunk_{i}" for i in range(len(chunks))]
)
```

**Key Patterns**:
1. **Collection per document version** - Enables versioning
2. **Rich metadata** - Helps with filtering and debugging
3. **Structured IDs** - doc_id + version + chunk index

### Step 4: Document Versioning

**Version Management** (`section1_rag_solution.py:362-404`)
```python
class DocumentVersionManager:
    def add_version(self, doc_id: str, filename: str, ...) -> int:
        if doc_id not in self.versions:
            self.versions[doc_id] = []

        version = len(self.versions[doc_id]) + 1

        metadata = DocumentMetadata(
            doc_id=doc_id,
            version=version,
            content_hash=content_hash,  # SHA256 hash
            ...
        )

        self.versions[doc_id].append(metadata)
        return version
```

**Key Insight**: Content hashing enables duplicate detection.

---

## Hybrid Search Implementation

### Why Hybrid Search?

**Semantic Search** (Vector similarity):
- ✅ Understands context and synonyms
- ✅ Good for conceptual queries
- ❌ Misses exact keyword matches
- ❌ Struggles with rare terms/names

**Keyword Search** (BM25):
- ✅ Excellent for exact matches
- ✅ Handles rare terms well
- ❌ No understanding of context
- ❌ Misses paraphrased queries

**Hybrid = Best of Both Worlds**

### What is BM25 Search?

**BM25 (Best Match 25)** is a probabilistic ranking algorithm for keyword-based search. It's the industry standard for traditional information retrieval and is used in search engines like Elasticsearch and Apache Solr.

#### 🎯 Simple Analogy: BM25 as a Library Librarian

Imagine you're a librarian helping someone find books about "machine learning":

**The Old Way (Simple Keyword Matching)**:
```
Student: "I need books about machine learning"
Bad Librarian: *Counts how many times "machine learning" appears*
→ Picks a 500-page encyclopedia that mentions it 100 times
→ Ignores a focused 50-page guide that mentions it 5 times
```

**The BM25 Way (Smart Ranking)**:
```
Student: "I need books about machine learning"
Smart Librarian: Thinks about 3 things:

1. RELEVANCE (Term Frequency with Saturation)
   "Does this book talk about 'machine learning'?"
   ✅ 5 mentions in a focused book = very relevant
   ⚠️ 100 mentions in an encyclopedia = not necessarily better
   💡 "More isn't always better - after 5-10 mentions, we get it!"

2. UNIQUENESS (Inverse Document Frequency)
   "How special are these words?"
   ✅ "machine learning" (rare) = pay attention!
   ❌ "the", "is", "a" (common) = ignore
   💡 "If every book says it, it's not helpful for narrowing down!"

3. FOCUS (Document Length Normalization)
   "Is this book focused or scattered?"
   ✅ Short book (50 pages) with 5 mentions = very focused
   ❌ Long book (500 pages) with 5 mentions = probably not focused
   💡 "A focused book is better than a scattered encyclopedia!"

→ Returns the 50-page focused guide first!
```

#### 🏆 Real-World Example

**Your Documents (Chunks)**:
```
Chunk A: "Python is great for machine learning and AI"  (9 words)
Chunk B: "Machine learning"  (2 words)
Chunk C: "This comprehensive guide covers programming, databases,
          web development, machine learning, testing, deployment..."  (50+ words)
```

**Query**: "machine learning"

**What Each Search Method Does**:

1. **Simple Keyword Count**:
   - Chunk A: 1 mention → Score = 1
   - Chunk B: 1 mention → Score = 1
   - Chunk C: 1 mention → Score = 1
   - **Problem**: All tied! 🤷

2. **BM25 Smart Ranking**:
   - **Chunk B** (2 words):
     - "machine learning" = 100% of content
     - Super focused! 🎯
     - **Score: 9.5 (HIGHEST)**

   - **Chunk A** (9 words):
     - "machine learning" = ~22% of content
     - Focused and contextual! 👍
     - **Score: 7.2**

   - **Chunk C** (50+ words):
     - "machine learning" = ~4% of content
     - Just mentioned in passing 😕
     - **Score: 3.1 (LOWEST)**

**BM25 Winner**: Chunk B (most focused on your query!)

#### 🍕 Another Analogy: Pizza Menu Search

You search for "pepperoni pizza":

**Menu Item 1**: "Pepperoni Pizza - Classic pepperoni with cheese" (7 words)
- 2 mentions of "pepperoni", very focused ✅
- **BM25 loves this!**

**Menu Item 2**: "Supreme Pizza - A delicious pizza loaded with vegetables, meats including pepperoni, olives, mushrooms, bell peppers, sausage, and bacon on a crispy crust with our special sauce..." (30 words)
- 1 mention of "pepperoni", buried in text ❌
- **BM25 ranks this lower**

**Menu Item 3**: "Pepperoni pepperoni pepperoni pepperoni pepperoni pepperoni pepperoni pepperoni pepperoni pepperoni" (10 words, all "pepperoni")
- 10 mentions but weird! 🤔
- **BM25's saturation kicks in**: "OK, we get it after 2-3 mentions!"
- **Not ranked much higher than Item 1**

**Key Insight**: BM25 finds the **most naturally relevant** item, not just the one that repeats keywords!

#### How BM25 Works

BM25 scores documents based on **term frequency** and **inverse document frequency**:

```
BM25(query, document) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
```

Where:
- **qi**: Query term i
- **f(qi, D)**: Frequency of term qi in document D
- **|D|**: Length of document D (in words)
- **avgdl**: Average document length in the collection
- **k1**: Term frequency saturation parameter (typically 1.2-2.0)
- **b**: Length normalization parameter (typically 0.75)
- **IDF(qi)**: Inverse document frequency of term qi

**In Simple Terms**:
1. **Term Frequency (TF)**: How often does the query term appear in this document?
   - More occurrences = higher score
   - But with diminishing returns (saturation)

2. **Inverse Document Frequency (IDF)**: How rare is this term across all documents?
   - Rare terms (e.g., "ChromaDB") = higher weight
   - Common terms (e.g., "the", "is") = lower weight

3. **Document Length Normalization**: Penalize longer documents
   - Prevents long documents from having unfair advantage
   - Shorter documents with matching terms rank higher

#### Why BM25 vs Simple TF-IDF?

**TF-IDF Issues**:
- Linear term frequency growth (100 occurrences scores 100× more than 1)
- No document length normalization
- Poor handling of term saturation

**BM25 Improvements**:
- ✅ **Saturation**: 10 occurrences barely scores more than 5 (diminishing returns)
- ✅ **Length normalization**: Adjusts for document length bias
- ✅ **Tunable parameters**: k1 and b can be optimized for your data

#### Example: BM25 in Action

**Documents**:
```
Doc1: "Machine learning is awesome"
Doc2: "Machine learning is a subset of artificial intelligence"
Doc3: "Deep learning uses neural networks"
```

**Query**: "machine learning"

**BM25 Scoring**:
1. **"machine"** appears in Doc1 (1×), Doc2 (1×) → rare term
2. **"learning"** appears in Doc1 (1×), Doc2 (1×), Doc3 (1×) → less rare
3. Doc1 is shorter (4 words) → gets boost
4. Doc2 is longer (9 words) → gets penalty

**Result**: Doc1 scores higher than Doc2 despite both containing "machine learning" because Doc1 is shorter and more focused.

#### Why BM25 for RAG?

**Perfect for**:
- ✅ Exact keyword matches (product names, IDs, acronyms)
- ✅ Rare technical terms that semantic search might miss
- ✅ Domain-specific jargon
- ✅ Names, dates, numbers

**Example Use Cases**:
```
Query: "What is the error code E404?"
→ BM25 finds exact "E404" match
→ Semantic search might miss the exact code

Query: "ChromaDB collection operations"
→ BM25 finds exact "ChromaDB" keyword
→ Semantic search might return generic database docs
```

#### BM25 Parameters in Practice

**k1 (Term Frequency Saturation)**:
- **k1 = 0**: Binary matching (term present or not)
- **k1 = 1.2**: Moderate saturation (default)
- **k1 = 2.0**: Less saturation (more weight to frequency)

**b (Length Normalization)**:
- **b = 0**: No length normalization
- **b = 0.75**: Moderate normalization (default)
- **b = 1.0**: Full length normalization

**For RAG systems**: Use defaults (k1=1.2, b=0.75) unless you have short, uniform documents (then reduce b).

### BM25 Implementation

**Indexing** (`section1_rag_solution.py:472-488`)
```python
class HybridRetriever:
    def index_documents(self, documents: List[str], doc_id: str, version: int):
        key = f"{doc_id}_v{version}"

        # Tokenize for BM25 (simple lowercase splitting)
        tokenized_docs = [doc.lower().split() for doc in documents]

        # Create BM25 index
        self.bm25_indexes[key] = BM25Okapi(tokenized_docs)
        self.doc_chunks[key] = documents
```

**BM25 Scoring** (`section1_rag_solution.py:516-517`)
```python
tokenized_query = query.lower().split()
bm25_scores = self.bm25_indexes[key].get_scores(tokenized_query)
```

### Semantic Search

**Vector Similarity** (`section1_rag_solution.py:520-524`)
```python
collection = client.get_collection(name=f"doc_{doc_id}_v{version}")
semantic_results = collection.query(
    query_texts=[query],
    n_results=min(k, len(self.doc_chunks[key]))
)
```

**Distance to Similarity** (`section1_rag_solution.py:536-538`)
```python
# ChromaDB returns L2 distance (lower = more similar)
similarity = 1 / (1 + distance)
```

### Score Fusion

**Normalization** (`section1_rag_solution.py:591-602`)
```python
def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1]"""
    if len(scores) == 0:
        return scores

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.ones_like(scores)  # All equal

    return (scores - min_score) / (max_score - min_score)
```

**Hybrid Score Calculation** (`section1_rag_solution.py:542-556`)
```python
for i, chunk in enumerate(self.doc_chunks[key]):
    bm25_score = bm25_scores_norm[i]
    semantic_score = semantic_scores.get(chunk, 0.0)

    # Normalize semantic scores
    if semantic_scores:
        max_sem_score = max(semantic_scores.values())
        if max_sem_score > 0:
            semantic_score = semantic_score / max_sem_score

    # Weighted combination (alpha=0.6 means 60% semantic, 40% BM25)
    hybrid_score = (
        self.alpha * semantic_score +
        (1 - self.alpha) * bm25_score
    )
```

**Tuning Alpha**:
- `alpha=1.0` → Pure semantic search
- `alpha=0.5` → Equal weighting
- `alpha=0.6` → Slightly favor semantic (good default)
- `alpha=0.0` → Pure BM25

---

## Re-ranking with Cross-Encoders

### Why Re-rank?

**Initial Retrieval** (Fast but approximate):
- Bi-encoders (like OpenAI embeddings) encode query and documents separately
- Compare with dot product or cosine similarity
- Fast but doesn't consider query-document interaction

**Re-ranking** (Slow but precise):
- Cross-encoders process query + document together
- Learns relationships between query and document
- More accurate but expensive

### Cross-Encoder Implementation

**Initialization** (`section1_rag_solution.py:608-616`)
```python
class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
```

**Popular Models**:
- `ms-marco-MiniLM-L-6-v2` (fast, 80MB)
- `ms-marco-MiniLM-L-12-v2` (better, 120MB)
- `ms-marco-electra-base` (best, 400MB)

**Re-ranking Process** (`section1_rag_solution.py:618-653`)
```python
def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
    if not documents:
        return []

    # Create query-document pairs
    pairs = [[query, doc['chunk']] for doc in documents]

    # Get scores from cross-encoder
    rerank_scores = self.model.predict(pairs)

    # Add scores to documents
    for doc, score in zip(documents, rerank_scores):
        doc['rerank_score'] = float(score)

    # Sort by rerank score
    reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

    return reranked[:top_k]
```

**Strategy**:
1. Retrieve 10-20 candidates with hybrid search (fast)
2. Re-rank top candidates with cross-encoder (slow but accurate)
3. Return top 3-5 for LLM context

---

## Conversation Memory Management

### Session-Based Memory

**Data Structure** (`section1_rag_solution.py:671-674`)
```python
class ConversationMemory:
    def __init__(self, max_history: int = 5, max_tokens: int = 2000):
        self.sessions = {}  # session_id -> deque of ConversationMessage
        self.max_history = max_history
        self.max_tokens = max_tokens
```

**Why deque?**
- Automatic FIFO with `maxlen` (oldest messages auto-removed)
- O(1) append and popleft operations

### Token-Aware Context

**Context Retrieval** (`section1_rag_solution.py:697-728`)
```python
def get_context(self, session_id: str) -> str:
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

    return "\n".join(context_lines)
```

**Key Pattern**: Start from most recent, work backward until token limit hit.

### Context-Aware Prompting

**Prompt Building** (`section1_rag_solution.py:792-819`)
```python
def _build_prompt(self, question: str, conversation_context: str, chunks: List[str]) -> str:
    prompt_parts = []

    # Add conversation history if available
    if conversation_context:
        prompt_parts.append("Previous conversation:")
        prompt_parts.append(conversation_context)
        prompt_parts.append("")

    # Add retrieved chunks
    prompt_parts.append("Relevant information from documents:")
    for i, chunk in enumerate(chunks, 1):
        prompt_parts.append(f"[{i}] {chunk}")

    # Add question and instructions
    prompt_parts.append("")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("")
    prompt_parts.append(
        "Answer the question based on the provided information. "
        "If the conversation context helps understand the question, use it. "
        "If the information is not in the documents, say so."
    )

    return "\n".join(prompt_parts)
```

**Structure**:
1. Conversation context (for follow-up questions)
2. Retrieved chunks (primary source)
3. Current question
4. Instructions (system message)

---

## Evaluation & Monitoring

### Faithfulness Evaluation

**LLM-as-Judge** (`section1_rag_solution.py:834-873`)
```python
class FaithfulnessEvaluator:
    def evaluate(self, question: str, answer: str, sources: List[str]) -> float:
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

        response = self.llm(prompt).strip()
        score = float(response)
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
```

**Why Faithfulness Matters**:
- Detects hallucinations
- Measures answer quality
- Helps tune retrieval parameters

### Query Caching

**Cache Implementation** (`section1_rag_solution.py:876-942`)
```python
class QueryCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}  # query_hash -> (response, expiry)
        self.ttl = ttl_seconds

    def _get_cache_key(self, query: str, doc_id: str, version: int) -> str:
        key_str = f"{query}:{doc_id}:v{version}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, doc_id: str, version: int) -> Optional[Dict]:
        cache_key = self._get_cache_key(query, doc_id, version)

        if cache_key in self.cache:
            response, expiry = self.cache[cache_key]

            if datetime.now() < expiry:
                return response  # Cache hit
            else:
                del self.cache[cache_key]  # Expired

        return None  # Cache miss
```

**Cache Key Design**:
- Include query text (what)
- Include doc_id (where)
- Include version (when)
- Hash with MD5 (consistent length)

### Metrics Tracking

**Performance Monitoring** (`section1_rag_solution.py:945-1005`)
```python
class MetricsTracker:
    def record_query(self, latency_ms: float, cached: bool, chunks: int, faithfulness: float):
        self.metrics["total_queries"] += 1

        if cached:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

        self.metrics["total_latency_ms"] += latency_ms
        self.metrics["total_chunks_retrieved"] += chunks
        self.metrics["total_faithfulness_score"] += faithfulness

    def get_stats(self) -> Dict[str, Any]:
        total_queries = self.metrics["total_queries"]

        return {
            "avg_latency_ms": self.metrics["total_latency_ms"] / total_queries,
            "avg_chunks_per_query": self.metrics["total_chunks_retrieved"] / total_queries,
            "avg_faithfulness_score": self.metrics["total_faithfulness_score"] / total_queries,
            "cache_hit_rate": self.metrics["cache_hits"] / total_queries
        }
```

**Key Metrics**:
- **Latency**: Response time (aim <500ms)
- **Cache hit rate**: % cached responses (aim >30%)
- **Faithfulness**: Answer quality (aim >0.8)
- **Chunks per query**: Context efficiency (aim 3-5)

---

## Production Patterns

### 1. Error Handling

**HTTP Exceptions** (`section1_rag_solution.py:182`)
```python
except Exception as e:
    logger.error(f"Error processing PDF: {str(e)}")
    raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")
```

**Global Handler** (`section1_rag_solution.py:1306-1314`)
```python
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }
```

### 2. Logging Strategy

**Structured Logging** (`section1_rag_solution.py:52-57`)
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

**Log Levels**:
- `INFO`: High-level operations (upload, query)
- `DEBUG`: Detailed operations (cache hits, score calculations)
- `WARNING`: Recoverable issues
- `ERROR`: Failures with stack traces

### 3. API Design

**RESTful Endpoints**:
- `POST /upload` - Create resource
- `POST /query` - Query resource
- `GET /documents` - List resources
- `GET /documents/{doc_id}/metadata` - Get resource
- `DELETE /documents/{doc_id}` - Delete resource
- `GET /health` - Health check

**Response Models** (`section1_rag_solution.py:108-124`)
```python
class UploadResponse(BaseModel):
    doc_id: str
    version: int
    message: str
    chunks_created: int
    format: DocumentFormat

class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[str]
    confidence: float
    faithfulness_score: float
    cached: bool
    latency_ms: float
    metadata: Dict[str, Any] = {}
```

**Why Pydantic?**
- Automatic validation
- Type safety
- API documentation (FastAPI integration)
- JSON serialization

---

## Common Pitfalls

### 1. Chunk Size Issues

❌ **Too Large** (>1000 chars):
- Dilutes relevance
- Exceeds embedding model limits
- Wastes LLM context window

❌ **Too Small** (<200 chars):
- Loses context
- Too many chunks to manage
- Increased retrieval noise

✅ **Sweet Spot**: 400-600 characters with 10% overlap

### 2. Embedding Model Mismatch

❌ **Problem**:
```python
# Index with OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Query with different model
query_embeddings = HuggingFaceEmbeddings()  # Wrong!
```

✅ **Solution**: Use same embedding model for indexing and querying

### 3. Ignoring Metadata

❌ **Poor Metadata**:
```python
metadatas = [{"chunk_index": i}]  # Minimal
```

✅ **Rich Metadata**:
```python
metadatas = [{
    "doc_id": doc_id,
    "version": version,
    "chunk_index": i,
    "total_chunks": len(chunks),
    "format": extension,
    "page_number": page_num,  # If PDF
    "section": section_name    # If structured
}]
```

### 4. No Hybrid Search

❌ **Semantic Only**:
- Misses exact keyword matches
- Poor for names, IDs, acronyms

✅ **Hybrid Approach**:
- Combines semantic understanding with exact matching
- Alpha parameter lets you tune balance

### 5. Forgetting Conversation Context

❌ **Stateless Queries**:
```
User: "What is RAG?"
Bot: "RAG is..."
User: "How does it work?"  # Bot has no context
```

✅ **Session-Based Memory**:
```python
# Store conversation in session
memory.add_message(session_id, "user", question)
memory.add_message(session_id, "assistant", answer)

# Retrieve for next query
context = memory.get_context(session_id)
```

---

## Code Templates

### Template 1: Basic RAG Pipeline

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import chromadb

# 1. Load document
with open("document.txt", "r") as f:
    text = f.read()

# 2. Chunk text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)

# 3. Create embeddings and store
client = chromadb.Client()
collection = client.create_collection(name="my_docs")

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# 4. Query
query = "What is the main topic?"
results = collection.query(
    query_texts=[query],
    n_results=3
)

print(results['documents'])
```

### Template 2: Hybrid Search

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearch:
    def __init__(self, documents, alpha=0.6):
        self.documents = documents
        self.alpha = alpha

        # BM25 index
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Vector index (pseudo-code)
        self.vector_index = create_vector_index(documents)

    def search(self, query, k=5):
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores = self._normalize(bm25_scores)

        # Semantic scores
        semantic_scores = self.vector_index.search(query)
        semantic_scores = self._normalize(semantic_scores)

        # Combine
        hybrid_scores = (
            self.alpha * semantic_scores +
            (1 - self.alpha) * bm25_scores
        )

        # Top k
        top_indices = np.argsort(hybrid_scores)[::-1][:k]
        return [self.documents[i] for i in top_indices]

    def _normalize(self, scores):
        return (scores - scores.min()) / (scores.max() - scores.min())
```

### Template 3: FastAPI RAG Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str
    doc_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

@app.post("/query", response_model=QueryResponse)
async def query_document(query: Query):
    try:
        # 1. Retrieve chunks
        chunks = retriever.retrieve(query.question, query.doc_id, k=5)

        # 2. Re-rank
        reranked = reranker.rerank(query.question, chunks, top_k=3)

        # 3. Generate answer
        answer = llm.generate(query.question, reranked)

        # 4. Evaluate
        confidence = evaluator.score(answer, reranked)

        return QueryResponse(
            answer=answer,
            sources=[chunk['text'] for chunk in reranked],
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Quick Reference: section1_rag_solution.py

### Classes & Responsibilities

| Class | Lines | Purpose |
|-------|-------|---------|
| `MultiFormatDocumentProcessor` | 138-352 | Process txt/pdf/json/csv files |
| `DocumentVersionManager` | 354-450 | Track document versions |
| `HybridRetriever` | 456-603 | Semantic + BM25 search |
| `Reranker` | 605-654 | Cross-encoder re-ranking |
| `ConversationMemory` | 660-740 | Session-based chat history |
| `ContextAwareQueryEngine` | 742-820 | Answer generation with context |
| `FaithfulnessEvaluator` | 826-874 | LLM-based answer evaluation |
| `QueryCache` | 876-942 | TTL-based response caching |
| `MetricsTracker` | 945-1005 | Performance monitoring |

### Key Algorithms

| Algorithm | Location | Complexity |
|-----------|----------|------------|
| Recursive text splitting | 149-154 | O(n) |
| BM25 scoring | 516-517 | O(n·m) |
| Vector similarity search | 520-524 | O(log n) |
| Score normalization | 591-602 | O(n) |
| Cross-encoder re-ranking | 638-642 | O(k) |
| Token-aware context window | 714-723 | O(h) |

Where:
- n = number of documents
- m = query length
- k = candidates to re-rank
- h = history size

---

## Interview Tips

### When Asked About RAG:

1. **Start Simple**: Explain basic retrieval → generation flow
2. **Add Complexity**: Mention hybrid search, re-ranking if relevant
3. **Show Trade-offs**: Fast retrieval vs. accurate re-ranking
4. **Mention Evaluation**: Faithfulness, relevance metrics
5. **Think Production**: Caching, monitoring, error handling

### Common Questions:

**Q: How do you handle large documents?**
A: Chunking with overlap + hierarchical retrieval (chunk → section → document)

**Q: What if semantic search misses exact matches?**
A: Use hybrid search (BM25 + semantic) with tunable alpha parameter

**Q: How do you prevent hallucinations?**
A: 1) Faithfulness evaluation, 2) Constrain to sources, 3) Confidence scoring

**Q: How do you scale to millions of documents?**
A: 1) Vector DB sharding, 2) Hierarchical retrieval, 3) Caching layer

---

**Ready to Practice?**
Check out the progressive challenges in `/tracks/01_rag_system/`:
- Beginner: Simple RAG with single format
- Intermediate: Multi-format with versioning
- Advanced: Full production system with all features

Good luck with your ELGO AI interview! 🚀
