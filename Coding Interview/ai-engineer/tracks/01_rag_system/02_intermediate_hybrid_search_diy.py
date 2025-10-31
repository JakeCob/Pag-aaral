"""
Challenge 02: Hybrid Search RAG (Intermediate)
DIY Starter Template

Complete the TODOs to build a hybrid search system combining BM25 and semantic search.
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

        # TODO 1: Calculate document lengths
        # Hint: Use len(doc.split()) for each document
        self.doc_lengths = []  # YOUR CODE HERE

        # TODO 2: Calculate average document length
        self.avgdl = 0  # YOUR CODE HERE

        # TODO 3: Compute IDF scores for all unique terms
        # Hint: Use _compute_idf() method
        pass  # YOUR CODE HERE

    def _compute_idf(self):
        """Calculate IDF (Inverse Document Frequency) for each term"""
        N = len(self.documents)
        df = Counter()  # Document frequency counter

        # TODO 4: Count how many documents each term appears in
        # Hint: For each doc, get unique terms and update df counter
        # YOUR CODE HERE

        # TODO 5: Calculate IDF for each term
        # Formula: IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        for term, doc_freq in df.items():
            self.idf_scores[term] = 0  # YOUR CODE HERE (use math.log)

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

        # TODO 6: Implement BM25 scoring formula
        # For each query term:
        #   1. Get term frequency in document: tf = doc.count(term)
        #   2. Get IDF score: idf = self.idf_scores.get(term, 0)
        #   3. Calculate BM25 component:
        #      score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))

        # YOUR CODE HERE

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

        # TODO 7: Score all documents
        scores = []
        for doc_idx in range(len(self.documents)):
            # YOUR CODE HERE: Calculate score for each document
            score = 0  # Replace with actual score
            scores.append((self.documents[doc_idx], score))

        # TODO 8: Sort by score (descending) and return top_k
        # YOUR CODE HERE
        return []  # Replace with sorted results


class SemanticRetriever:
    """Semantic search using sentence embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic retriever.

        Args:
            model_name: SentenceTransformer model name
        """
        # TODO 9: Initialize sentence transformer model
        self.model = None  # YOUR CODE HERE

        # TODO 10: Initialize ChromaDB client and collection
        self.client = None  # YOUR CODE HERE
        self.collection = None  # YOUR CODE HERE

    def add_documents(self, documents: List[str]):
        """
        Add documents to vector database.

        Args:
            documents: List of document strings
        """
        # TODO 11: Generate embeddings for all documents
        embeddings = []  # YOUR CODE HERE (use self.model.encode)

        # TODO 12: Add to ChromaDB collection
        # Hint: collection.add(embeddings=..., documents=..., ids=...)
        # YOUR CODE HERE

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search documents using semantic similarity.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of (document, score) tuples sorted by similarity
        """
        # TODO 13: Generate query embedding
        query_embedding = None  # YOUR CODE HERE

        # TODO 14: Query ChromaDB collection
        # Hint: collection.query(query_embeddings=..., n_results=top_k)
        results = None  # YOUR CODE HERE

        # TODO 15: Extract documents and distances, convert to scores
        # Note: ChromaDB returns distances, convert to similarity scores
        # Hint: similarity = 1 / (1 + distance)
        doc_score_pairs = []  # YOUR CODE HERE

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

        # TODO 16: Initialize both retrievers
        self.bm25_retriever = None  # YOUR CODE HERE
        self.semantic_retriever = None  # YOUR CODE HERE

    def add_documents(self, documents: List[str]):
        """Add documents to both retrievers"""
        # TODO 17: Add documents to both BM25 and semantic retrievers
        # YOUR CODE HERE
        pass

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

        # TODO 18: Implement min-max normalization
        # Formula: (score - min) / (max - min)
        # Handle edge case: if max == min, return all 1.0s

        # YOUR CODE HERE
        return []

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search using hybrid approach.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of top_k documents sorted by hybrid score
        """
        # TODO 19: Get results from both retrievers (get more than top_k for better fusion)
        bm25_results = []  # YOUR CODE HERE (get top_k * 2 results)
        semantic_results = []  # YOUR CODE HERE (get top_k * 2 results)

        # TODO 20: Create document -> score mappings
        bm25_scores = {}  # Map document -> BM25 score
        semantic_scores = {}  # Map document -> semantic score
        # YOUR CODE HERE

        # TODO 21: Normalize scores separately
        bm25_norm = {}  # Normalized BM25 scores
        semantic_norm = {}  # Normalized semantic scores
        # YOUR CODE HERE

        # TODO 22: Compute hybrid scores
        # Formula: hybrid = alpha * semantic_norm + (1-alpha) * bm25_norm
        hybrid_scores = {}
        all_docs = set(bm25_scores.keys()) | set(semantic_scores.keys())

        for doc in all_docs:
            # YOUR CODE HERE: Calculate hybrid score
            # Handle case where doc might not be in both retrievers
            hybrid_scores[doc] = 0  # Replace with actual calculation

        # TODO 23: Sort by hybrid score and return top_k
        # YOUR CODE HERE
        return []


# =============================================================================
# TESTING CODE
# =============================================================================

def test_hybrid_retriever():
    """Test the hybrid retriever with sample documents"""

    # Sample documents
    documents = [
        "Python is a high-level programming language for general-purpose programming.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "FastAPI is a modern web framework for building APIs with Python.",
        "LangChain helps build applications with large language models."
    ]

    print("Initializing hybrid retriever...")
    retriever = HybridRetriever(alpha=0.6)
    retriever.add_documents(documents)

    # Test 1: Keyword-heavy query (should favor BM25)
    print("\n" + "="*70)
    print("Test 1: Keyword Query - 'FastAPI Python'")
    print("="*70)
    results = retriever.search("FastAPI Python", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    # Test 2: Conceptual query (should favor semantic)
    print("\n" + "="*70)
    print("Test 2: Conceptual Query - 'AI and neural nets'")
    print("="*70)
    results = retriever.search("AI and neural nets", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    # Test 3: Mixed query
    print("\n" + "="*70)
    print("Test 3: Mixed Query - 'building ML applications'")
    print("="*70)
    results = retriever.search("building ML applications", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    # TODO 24: Complete all TODOs above, then run this test
    test_hybrid_retriever()
