"""
Hybrid Search RAG - VANILLA PYTHON (No LangChain)
Build BM25 + Semantic Search from scratch

This demonstrates deep understanding of retrieval algorithms.
Perfect for companies that build custom systems!

Key Concepts:
- BM25 algorithm from scratch (no libraries)
- Semantic search with raw vector operations
- Score fusion and normalization
"""

from typing import List, Tuple, Dict
from collections import Counter
import math
import numpy as np
from sentence_transformers import SentenceTransformer


class BM25FromScratch:
    """
    BM25 implementation from first principles.
    No external BM25 libraries - shows you understand the algorithm!

    BM25 Formula:
    score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))

    Where:
    - qi: query term i
    - f(qi, D): frequency of qi in document D
    - |D|: length of document D
    - avgdl: average document length
    - k1: term frequency saturation (default 1.2)
    - b: length normalization (default 0.75)
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avgdl = 0
        self.idf_scores = {}
        self.doc_term_freqs = []  # Store term frequencies for each doc

    def fit(self, documents: List[str]):
        """
        Build BM25 index from documents.

        Steps:
        1. Tokenize documents
        2. Calculate document lengths
        3. Calculate IDF for each term
        4. Store term frequencies
        """
        self.documents = documents
        N = len(documents)

        # Step 1 & 2: Tokenize and get lengths
        tokenized_docs = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))

        # Calculate average document length
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

        # Step 3: Calculate IDF
        # IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        df = Counter()  # Document frequency: how many docs contain each term

        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            df.update(unique_tokens)

        for term, doc_freq in df.items():
            # IDF formula
            idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            self.idf_scores[term] = idf

        # Step 4: Store term frequencies for each document
        for tokens in tokenized_docs:
            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split.
        In production, you might use more sophisticated tokenization.
        """
        return text.lower().split()

    def score_document(self, query: str, doc_idx: int) -> float:
        """
        Calculate BM25 score for a specific document.

        Args:
            query: Search query
            doc_idx: Index of document to score

        Returns:
            BM25 score
        """
        query_terms = self._tokenize(query)
        score = 0.0

        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]

        for term in query_terms:
            if term not in self.idf_scores:
                continue  # Term not in corpus

            # Get term frequency in this document
            tf = term_freqs[term]

            # Get IDF
            idf = self.idf_scores[term]

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search documents using BM25.

        Returns:
            List of (document, score) tuples sorted by score
        """
        scores = []

        for doc_idx in range(len(self.documents)):
            score = self.score_document(query, doc_idx)
            scores.append((self.documents[doc_idx], score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


class SemanticSearchFromScratch:
    """
    Semantic search using embeddings.
    Uses sentence-transformers directly (no LangChain).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.doc_embeddings = None

    def fit(self, documents: List[str]):
        """Create embeddings for all documents"""
        self.documents = documents

        # Create embeddings
        self.doc_embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity manually.
        Shows you understand the math!

        cosine_sim = (A · B) / (||A|| × ||B||)
        """
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search documents using semantic similarity.

        Returns:
            List of (document, score) tuples sorted by similarity
        """
        # Create query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]

        # Calculate similarity with each document
        scores = []
        for doc_idx, doc_embedding in enumerate(self.doc_embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            scores.append((self.documents[doc_idx], similarity))

        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


class HybridSearchFromScratch:
    """
    Combine BM25 (keyword) and Semantic (meaning) search.
    Built from scratch - no frameworks!

    This is what production RAG systems use.
    """

    def __init__(self, alpha: float = 0.6):
        """
        Initialize hybrid search.

        Args:
            alpha: Weight for semantic search (0-1)
                   score = alpha * semantic + (1-alpha) * bm25
                   Default 0.6 means 60% semantic, 40% BM25
        """
        self.alpha = alpha
        self.bm25 = BM25FromScratch()
        self.semantic = SemanticSearchFromScratch()

    def fit(self, documents: List[str]):
        """Index documents in both retrievers"""
        print(f"📚 Indexing {len(documents)} documents...")

        print("  Building BM25 index...")
        self.bm25.fit(documents)

        print("  Creating semantic embeddings...")
        self.semantic.fit(documents)

        print("✓ Indexing complete!")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Min-max normalization to [0, 1] range.

        normalized = (score - min) / (max - min)
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        # Handle edge case: all scores are the same
        if max_score == min_score:
            return [1.0] * len(scores)

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Tuple[str, float]] | List[str]:
        """
        Hybrid search: combine BM25 and semantic.

        Algorithm:
        1. Get results from both retrievers (fetch more for better fusion)
        2. Normalize scores to [0, 1] range
        3. Combine: hybrid = alpha * semantic + (1-alpha) * bm25
        4. Return top_k results

        Args:
            query: Search query
            top_k: Number of results to return
            return_scores: If True, return (doc, score) tuples

        Returns:
            List of documents or (document, score) tuples
        """
        # Fetch more candidates for better fusion
        fetch_k = top_k * 3

        # Get results from both retrievers
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        semantic_results = self.semantic.search(query, top_k=fetch_k)

        # Create doc -> score mappings
        bm25_scores = {doc: score for doc, score in bm25_results}
        semantic_scores = {doc: score for doc, score in semantic_results}

        # Normalize scores separately
        if bm25_scores:
            bm25_score_list = list(bm25_scores.values())
            bm25_norm_list = self._normalize_scores(bm25_score_list)
            bm25_norm = dict(zip(bm25_scores.keys(), bm25_norm_list))
        else:
            bm25_norm = {}

        if semantic_scores:
            semantic_score_list = list(semantic_scores.values())
            semantic_norm_list = self._normalize_scores(semantic_score_list)
            semantic_norm = dict(zip(semantic_scores.keys(), semantic_norm_list))
        else:
            semantic_norm = {}

        # Combine scores
        hybrid_scores = {}
        all_docs = set(bm25_scores.keys()) | set(semantic_scores.keys())

        for doc in all_docs:
            bm25_s = bm25_norm.get(doc, 0.0)
            semantic_s = semantic_norm.get(doc, 0.0)

            # Hybrid formula
            hybrid_scores[doc] = self.alpha * semantic_s + (1 - self.alpha) * bm25_s

        # Sort and return top_k
        sorted_results = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        if return_scores:
            return sorted_results
        else:
            return [doc for doc, score in sorted_results]


# =============================================================================
# EXAMPLE USAGE & TESTING
# =============================================================================

def test_bm25():
    """Test BM25 implementation"""
    print("\n" + "="*70)
    print("Testing BM25 (Keyword Search)")
    print("="*70)

    documents = [
        "Python is a high-level programming language.",
        "FastAPI is a modern Python web framework.",
        "Machine learning uses Python extensively.",
        "JavaScript is used for web development.",
        "Python has great data science libraries."
    ]

    bm25 = BM25FromScratch()
    bm25.fit(documents)

    query = "Python programming"
    results = bm25.search(query, top_k=3)

    print(f"\nQuery: '{query}'")
    print("\nTop 3 Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [Score: {score:.4f}] {doc}")


def test_semantic():
    """Test semantic search"""
    print("\n" + "="*70)
    print("Testing Semantic Search")
    print("="*70)

    documents = [
        "Python is a high-level programming language.",
        "FastAPI is a modern Python web framework.",
        "Machine learning uses Python extensively.",
        "JavaScript is used for web development.",
        "Python has great data science libraries."
    ]

    semantic = SemanticSearchFromScratch()
    semantic.fit(documents)

    query = "data science tools"  # No exact keyword match
    results = semantic.search(query, top_k=3)

    print(f"\nQuery: '{query}'")
    print("\nTop 3 Results (by semantic similarity):")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [Score: {score:.4f}] {doc}")


def test_hybrid():
    """Test hybrid search"""
    print("\n" + "="*70)
    print("Testing Hybrid Search (BM25 + Semantic)")
    print("="*70)

    documents = [
        "LangChain is a framework for building LLM applications.",
        "LangChain supports multiple LLMs including OpenAI and Anthropic.",
        "LCEL is LangChain Expression Language for building chains.",
        "FastAPI is a modern web framework for Python.",
        "Python is excellent for machine learning and AI."
    ]

    hybrid = HybridSearchFromScratch(alpha=0.6)
    hybrid.fit(documents)

    # Test 1: Keyword-heavy query
    print("\n--- Test 1: Keyword Query ---")
    query1 = "LangChain LLM"
    results1 = hybrid.search(query1, top_k=3, return_scores=True)

    print(f"Query: '{query1}'")
    print("(Should favor documents with exact 'LangChain' match)")
    for i, (doc, score) in enumerate(results1, 1):
        print(f"{i}. [Score: {score:.4f}] {doc}")

    # Test 2: Semantic query
    print("\n--- Test 2: Semantic Query ---")
    query2 = "building applications with AI models"
    results2 = hybrid.search(query2, top_k=3, return_scores=True)

    print(f"Query: '{query2}'")
    print("(Should find conceptually similar docs)")
    for i, (doc, score) in enumerate(results2, 1):
        print(f"{i}. [Score: {score:.4f}] {doc}")

    # Test 3: Different alpha values
    print("\n--- Test 3: Alpha Comparison ---")
    query3 = "framework"

    for alpha in [0.2, 0.5, 0.8]:
        hybrid_alpha = HybridSearchFromScratch(alpha=alpha)
        hybrid_alpha.fit(documents)
        results = hybrid_alpha.search(query3, top_k=2, return_scores=True)

        emphasis = "BM25-heavy" if alpha < 0.5 else "Balanced" if alpha == 0.5 else "Semantic-heavy"
        print(f"\nAlpha = {alpha} ({emphasis}):")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.4f}] {doc[:60]}...")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("VANILLA HYBRID SEARCH - Built from Scratch!")
    print("No LangChain, just algorithms and math")
    print("="*70)

    test_bm25()
    test_semantic()
    test_hybrid()

    print("\n" + "="*70)
    print("✅ All tests complete!")
    print("\nKey Takeaways:")
    print("1. BM25 = Keyword matching with TF-IDF + length normalization")
    print("2. Semantic = Embedding similarity (cosine)")
    print("3. Hybrid = Combine both for best results")
    print("4. Alpha parameter tunes BM25 vs Semantic weight")
    print("="*70)


if __name__ == "__main__":
    main()
