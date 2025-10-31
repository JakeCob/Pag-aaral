"""
Challenge 02: Hybrid Search RAG (Intermediate)
COMPLETE SOLUTION

This demonstrates a production-quality hybrid search implementation.
"""

from typing import List, Tuple, Dict
from collections import Counter
import math
import chromadb
from sentence_transformers import SentenceTransformer


class BM25Retriever:
    """BM25 keyword-based retriever"""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter (typical: 1.2)
            b: Length normalization parameter (typical: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avgdl = 0
        self.idf_scores = {}

    def add_documents(self, documents: List[str]):
        """
        Add documents and build BM25 index.

        Args:
            documents: List of document strings
        """
        self.documents = documents

        # Calculate document lengths
        self.doc_lengths = [len(doc.split()) for doc in documents]

        # Calculate average document length
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

        # Compute IDF scores for all unique terms
        self._compute_idf()

    def _compute_idf(self):
        """Calculate IDF (Inverse Document Frequency) for each term"""
        N = len(self.documents)
        df = Counter()  # Document frequency counter

        # Count how many documents each term appears in
        for doc in self.documents:
            unique_terms = set(doc.lower().split())
            df.update(unique_terms)

        # Calculate IDF for each term
        # Formula: IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        for term, doc_freq in df.items():
            self.idf_scores[term] = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def _bm25_score(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a document given query terms.

        Args:
            query_terms: List of query terms (after tokenization)
            doc_idx: Index of document to score

        Returns:
            BM25 score
        """
        score = 0.0
        doc = self.documents[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]

        # Calculate term frequency for query terms in this document
        term_freqs = Counter(doc)

        for term in query_terms:
            if term not in self.idf_scores:
                continue  # Skip terms not in corpus

            tf = term_freqs[term]  # Term frequency in document
            idf = self.idf_scores[term]  # IDF score for term

            # BM25 formula component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search documents using BM25.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of (document, score) tuples sorted by score
        """
        query_terms = query.lower().split()

        # Score all documents
        scores = []
        for doc_idx in range(len(self.documents)):
            score = self._bm25_score(query_terms, doc_idx)
            scores.append((self.documents[doc_idx], score))

        # Sort by score (descending) and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class SemanticRetriever:
    """Semantic search using sentence embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic retriever.

        Args:
            model_name: SentenceTransformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client()
        # Create collection with unique name to avoid conflicts
        try:
            self.collection = self.client.create_collection("semantic_docs")
        except:
            # If collection exists, delete and recreate
            self.client.delete_collection("semantic_docs")
            self.collection = self.client.create_collection("semantic_docs")

    def add_documents(self, documents: List[str]):
        """
        Add documents to vector database.

        Args:
            documents: List of document strings
        """
        # Generate embeddings for all documents
        embeddings = self.model.encode(documents, convert_to_numpy=True)

        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search documents using semantic similarity.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of (document, score) tuples sorted by similarity
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Query ChromaDB collection
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, self.collection.count())
        )

        # Extract documents and distances, convert to similarity scores
        # ChromaDB returns L2 distances, convert to similarity
        # similarity = 1 / (1 + distance)
        doc_score_pairs = []
        for doc, distance in zip(results['documents'][0], results['distances'][0]):
            similarity = 1.0 / (1.0 + distance)
            doc_score_pairs.append((doc, similarity))

        return doc_score_pairs


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic search"""

    def __init__(self, alpha: float = 0.6):
        """
        Initialize hybrid retriever.

        Args:
            alpha: Weight for semantic search (0-1)
                  hybrid_score = alpha * semantic + (1-alpha) * bm25
        """
        self.alpha = alpha
        self.bm25_retriever = BM25Retriever()
        self.semantic_retriever = SemanticRetriever()

    def add_documents(self, documents: List[str]):
        """Add documents to both retrievers"""
        self.bm25_retriever.add_documents(documents)
        self.semantic_retriever.add_documents(documents)

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: List of scores to normalize

        Returns:
            Normalized scores
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        # Handle edge case: if all scores are the same
        if max_score == min_score:
            return [1.0] * len(scores)

        # Min-max normalization
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search using hybrid approach.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of top_k documents sorted by hybrid score
        """
        # Get results from both retrievers (fetch more for better fusion)
        fetch_k = top_k * 2
        bm25_results = self.bm25_retriever.search(query, top_k=fetch_k)
        semantic_results = self.semantic_retriever.search(query, top_k=fetch_k)

        # Create document -> score mappings
        bm25_scores = {doc: score for doc, score in bm25_results}
        semantic_scores = {doc: score for doc, score in semantic_results}

        # Normalize scores separately
        if bm25_scores:
            bm25_score_list = list(bm25_scores.values())
            bm25_norm_list = self._normalize_scores(bm25_score_list)
            bm25_norm = {doc: norm_score for doc, norm_score in
                        zip(bm25_scores.keys(), bm25_norm_list)}
        else:
            bm25_norm = {}

        if semantic_scores:
            semantic_score_list = list(semantic_scores.values())
            semantic_norm_list = self._normalize_scores(semantic_score_list)
            semantic_norm = {doc: norm_score for doc, norm_score in
                           zip(semantic_scores.keys(), semantic_norm_list)}
        else:
            semantic_norm = {}

        # Compute hybrid scores
        hybrid_scores = {}
        all_docs = set(bm25_scores.keys()) | set(semantic_scores.keys())

        for doc in all_docs:
            # Get normalized scores (default to 0 if not present)
            bm25_s = bm25_norm.get(doc, 0.0)
            semantic_s = semantic_norm.get(doc, 0.0)

            # Hybrid score: alpha * semantic + (1-alpha) * bm25
            hybrid_scores[doc] = self.alpha * semantic_s + (1 - self.alpha) * bm25_s

        # Sort by hybrid score (descending) and return top_k documents
        sorted_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:top_k]]


# =============================================================================
# TESTING CODE
# =============================================================================

def test_bm25_retriever():
    """Test BM25 retriever independently"""
    print("\n" + "="*70)
    print("Testing BM25 Retriever")
    print("="*70)

    documents = [
        "Python is a high-level programming language for general-purpose programming.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "FastAPI is a modern web framework for building APIs with Python.",
        "LangChain helps build applications with large language models."
    ]

    retriever = BM25Retriever()
    retriever.add_documents(documents)

    query = "FastAPI Python"
    results = retriever.search(query, top_k=3)

    print(f"\nQuery: '{query}'")
    print("\nResults:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [{score:.4f}] {doc}")


def test_semantic_retriever():
    """Test semantic retriever independently"""
    print("\n" + "="*70)
    print("Testing Semantic Retriever")
    print("="*70)

    documents = [
        "Python is a high-level programming language for general-purpose programming.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "FastAPI is a modern web framework for building APIs with Python.",
        "LangChain helps build applications with large language models."
    ]

    retriever = SemanticRetriever()
    retriever.add_documents(documents)

    query = "AI and neural nets"
    results = retriever.search(query, top_k=3)

    print(f"\nQuery: '{query}'")
    print("\nResults:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [{score:.4f}] {doc}")


def test_hybrid_retriever():
    """Test the hybrid retriever with sample documents"""
    print("\n" + "="*70)
    print("Testing Hybrid Retriever")
    print("="*70)

    # Sample documents
    documents = [
        "Python is a high-level programming language for general-purpose programming.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "FastAPI is a modern web framework for building APIs with Python.",
        "LangChain helps build applications with large language models."
    ]

    print("\nInitializing hybrid retriever (alpha=0.6)...")
    retriever = HybridRetriever(alpha=0.6)
    retriever.add_documents(documents)

    # Test 1: Keyword-heavy query (should favor BM25)
    print("\n" + "="*70)
    print("Test 1: Keyword Query - 'FastAPI Python'")
    print("(Expected: FastAPI doc should rank #1 due to exact keyword match)")
    print("="*70)
    results = retriever.search("FastAPI Python", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    # Test 2: Conceptual query (should favor semantic)
    print("\n" + "="*70)
    print("Test 2: Conceptual Query - 'AI and neural nets'")
    print("(Expected: ML and Deep Learning docs should rank high)")
    print("="*70)
    results = retriever.search("AI and neural nets", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    # Test 3: Mixed query
    print("\n" + "="*70)
    print("Test 3: Mixed Query - 'building ML applications'")
    print("(Expected: Balanced results from both retrievers)")
    print("="*70)
    results = retriever.search("building ML applications", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    # Test 4: Different alpha values
    print("\n" + "="*70)
    print("Test 4: Alpha Comparison - 'Python framework'")
    print("="*70)

    for alpha in [0.2, 0.5, 0.8]:
        print(f"\nAlpha = {alpha} ({'BM25-heavy' if alpha < 0.5 else 'Semantic-heavy' if alpha > 0.5 else 'Balanced'}):")
        retriever_alpha = HybridRetriever(alpha=alpha)
        retriever_alpha.add_documents(documents)
        results = retriever_alpha.search("Python framework", top_k=2)
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc[:60]}...")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    # Run all tests
    test_bm25_retriever()
    test_semantic_retriever()
    test_hybrid_retriever()
