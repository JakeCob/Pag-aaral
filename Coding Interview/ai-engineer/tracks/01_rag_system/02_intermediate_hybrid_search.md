# Challenge 02: Hybrid Search RAG (Intermediate)

**Difficulty**: Intermediate
**Time Estimate**: 45-60 minutes
**Interview Section**: Section 1 - Part B

---

## 📋 Challenge Description

Build a RAG system that combines **semantic search** (vector similarity) and **BM25 search** (keyword matching) to retrieve the most relevant documents. This is called **hybrid search** and is crucial for production RAG systems.

### Why Hybrid Search?

- **Semantic search alone**: Misses exact keyword matches (e.g., product codes, names)
- **BM25 alone**: Misses conceptual similarity (e.g., "ML" vs "machine learning")
- **Hybrid**: Gets the best of both worlds!

---

## 🎯 Requirements

### Part A: Implement BM25 Search (20 min)

1. **BM25Retriever class** with:
   - `add_documents(documents: List[str])` - Build BM25 index
   - `search(query: str, top_k: int) -> List[Tuple[str, float]]` - Return (doc, score) pairs
   - Use parameters: `k1=1.2`, `b=0.75`

2. **Scoring formula**:
   ```
   BM25(q, d) = Σ IDF(qi) × (f(qi, d) × (k1 + 1)) / (f(qi, d) + k1 × (1 - b + b × |d| / avgdl))
   ```
   - Where:
     - `qi` = query term i
     - `f(qi, d)` = frequency of qi in document d
     - `|d|` = length of document d
     - `avgdl` = average document length

### Part B: Implement Semantic Search (15 min)

1. **SemanticRetriever class** with:
   - `add_documents(documents: List[str])` - Create embeddings + store in vector DB
   - `search(query: str, top_k: int) -> List[Tuple[str, float]]` - Cosine similarity search
   - Use ChromaDB or FAISS

2. **Embedding model**:
   - Use `sentence-transformers/all-MiniLM-L6-v2` (or similar)
   - Normalize vectors for cosine similarity

### Part C: Combine with Hybrid Search (20 min)

1. **HybridRetriever class** with:
   - Initialize both BM25 and Semantic retrievers
   - `search(query: str, top_k: int, alpha: float = 0.6) -> List[str]`
   - **Scoring formula**:
     ```python
     hybrid_score = alpha * semantic_score + (1 - alpha) * bm25_score
     ```
   - Return top_k documents sorted by hybrid score

2. **Score normalization**:
   - Normalize BM25 scores to [0, 1] range
   - Normalize semantic scores to [0, 1] range
   - Then apply alpha weighting

---

## 📊 Example Usage

```python
# Sample documents
documents = [
    "Python is a high-level programming language for general-purpose programming.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "FastAPI is a modern web framework for building APIs with Python.",
    "LangChain helps build applications with large language models."
]

# Build hybrid retriever
retriever = HybridRetriever(alpha=0.6)
retriever.add_documents(documents)

# Test query 1: Exact keyword match
results = retriever.search("FastAPI Python", top_k=3)
# Expected: "FastAPI is a modern..." should rank high (keyword match)

# Test query 2: Conceptual similarity
results = retriever.search("AI and neural nets", top_k=3)
# Expected: "Machine learning..." and "Deep learning..." should rank high

# Test query 3: Mixed
results = retriever.search("building ML applications", top_k=3)
# Expected: Balanced results from both retrievers
```

---

## ✅ Expected Output

```
Query: "FastAPI Python"

Results (alpha=0.6):
1. FastAPI is a modern web framework... (score: 0.85)
   - BM25: 0.92 (exact keyword match "FastAPI" + "Python")
   - Semantic: 0.74 (moderate semantic match)
   - Hybrid: 0.6*0.74 + 0.4*0.92 = 0.812

2. Python is a high-level programming... (score: 0.72)
   - BM25: 0.78 (keyword "Python")
   - Semantic: 0.63
   - Hybrid: 0.690

3. LangChain helps build applications... (score: 0.58)
   - BM25: 0.45
   - Semantic: 0.68 (semantic match with "building")
   - Hybrid: 0.588
```

---

## 🧪 Test Cases

### Test 1: BM25 Favors Exact Matches
```python
query = "Python FastAPI"
results = retriever.search(query, top_k=1, alpha=0.3)  # Low alpha = favor BM25
assert "FastAPI" in results[0]
```

### Test 2: Semantic Favors Concepts
```python
query = "artificial intelligence neural networks"
results = retriever.search(query, top_k=2, alpha=0.8)  # High alpha = favor semantic
assert any("machine learning" in r.lower() for r in results)
assert any("deep learning" in r.lower() for r in results)
```

### Test 3: Hybrid Balances Both
```python
query = "ML frameworks"
results = retriever.search(query, top_k=3, alpha=0.5)
# Should include both keyword matches and semantic matches
```

---

## 💡 Implementation Tips

### BM25 Implementation
```python
from collections import Counter
import math

class BM25Retriever:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avgdl = 0
        self.idf_scores = {}

    def _compute_idf(self):
        """Calculate IDF for each term"""
        N = len(self.documents)
        df = Counter()  # Document frequency

        for doc in self.documents:
            unique_terms = set(doc.lower().split())
            df.update(unique_terms)

        for term, doc_freq in df.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf_scores[term] = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
```

### Semantic Search with ChromaDB
```python
import chromadb
from sentence_transformers import SentenceTransformer

class SemanticRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("docs")

    def add_documents(self, documents: List[str]):
        embeddings = self.model.encode(documents)
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
```

### Hybrid Combination
```python
def _normalize_scores(self, scores: List[float]) -> List[float]:
    """Min-max normalization to [0, 1]"""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]
```

---

## 🎓 Key Concepts to Demonstrate

1. **BM25 Algorithm**: Understanding of term frequency saturation and length normalization
2. **Vector Embeddings**: How to use sentence transformers for semantic search
3. **Score Normalization**: Min-max scaling for combining different score ranges
4. **Alpha Parameter**: Tuning the balance between semantic and keyword search
5. **Performance**: Hybrid search is ~2x slower than single method but much more accurate

---

## 🚀 Extensions (If Time Permits)

1. **Custom tokenization**: Use spaCy or NLTK for better text processing
2. **Re-ranking**: Add cross-encoder for final re-ranking step
3. **Caching**: Cache embeddings to avoid recomputation
4. **Batching**: Process multiple queries in parallel

---

## 📚 Related Concepts

- **Reciprocal Rank Fusion (RRF)**: Alternative to weighted averaging
- **Ensemble Retrieval**: Combining 3+ retrieval methods
- **Query Expansion**: Adding synonyms before searching

---

**Time Allocation**:
- BM25 Implementation: 20 min
- Semantic Search: 15 min
- Hybrid Combination: 15 min
- Testing: 10 min
- **Total**: 60 min

**Good luck!** 🎯
