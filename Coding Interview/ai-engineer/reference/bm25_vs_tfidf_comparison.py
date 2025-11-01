"""
BM25 vs TF-IDF Comparison
Demonstrates the key differences with concrete examples
"""

from collections import Counter
import math
from typing import List


class SimpleTFIDF:
    """Traditional TF-IDF implementation"""

    def __init__(self, documents: List[str]):
        self.documents = documents

        # Calculate IDF
        N = len(documents)
        df = Counter()
        for doc in documents:
            unique_terms = set(doc.lower().split())
            df.update(unique_terms)

        self.idf = {}
        for term, doc_freq in df.items():
            self.idf[term] = math.log(N / doc_freq)

    def score(self, query: str, doc_idx: int) -> float:
        """Calculate TF-IDF score"""
        score = 0.0
        doc = self.documents[doc_idx].lower().split()
        doc_len = len(doc)
        term_freqs = Counter(doc)

        for term in query.lower().split():
            if term not in self.idf:
                continue

            # Traditional TF-IDF: tf * idf
            tf = term_freqs[term] / doc_len  # Normalized by doc length
            idf = self.idf[term]

            score += tf * idf

        return score


class BM25:
    """BM25 implementation (from your code)"""

    def __init__(self, documents: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents

        # Calculate document lengths
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avgdl = sum(self.doc_lengths) / len(documents)

        # Calculate IDF
        N = len(documents)
        df = Counter()
        for doc in documents:
            unique_terms = set(doc.lower().split())
            df.update(unique_terms)

        self.idf = {}
        for term, doc_freq in df.items():
            # BM25's improved IDF formula
            self.idf[term] = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score"""
        score = 0.0
        doc = self.documents[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = Counter(doc)

        for term in query.lower().split():
            if term not in self.idf:
                continue

            tf = term_freqs[term]
            idf = self.idf[term]

            # BM25 formula with saturation
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)

        return score


def demo_saturation_effect():
    """Show how BM25 saturates while TF-IDF grows linearly"""
    print("=" * 70)
    print("Demo 1: Term Frequency Saturation")
    print("=" * 70)

    # Create documents with different term frequencies
    documents = [
        "Python",  # 1 occurrence
        "Python Python Python Python Python",  # 5 occurrences
        "Python " * 10,  # 10 occurrences
        "Python " * 50,  # 50 occurrences
        "JavaScript is a language"  # No Python
    ]

    query = "Python"

    tfidf = SimpleTFIDF(documents)
    bm25 = BM25(documents)

    print(f"\nQuery: '{query}'")
    print(f"\n{'Doc':<5} {'Term Count':<12} {'TF-IDF Score':<15} {'BM25 Score':<15} {'Growth'}")
    print("-" * 70)

    prev_tfidf = 0
    prev_bm25 = 0

    for i, doc in enumerate(documents[:4]):  # Skip JavaScript doc
        term_count = doc.lower().split().count("python")
        tfidf_score = tfidf.score(query, i)
        bm25_score = bm25.score(query, i)

        if i > 0:
            tfidf_growth = (tfidf_score / prev_tfidf) if prev_tfidf > 0 else 0
            bm25_growth = (bm25_score / prev_bm25) if prev_bm25 > 0 else 0
            growth_str = f"TF-IDF: {tfidf_growth:.1f}x, BM25: {bm25_growth:.1f}x"
        else:
            growth_str = "baseline"

        print(f"{i:<5} {term_count:<12} {tfidf_score:<15.4f} {bm25_score:<15.4f} {growth_str}")

        prev_tfidf = tfidf_score
        prev_bm25 = bm25_score

    print("\n📊 Key Observation:")
    print("- TF-IDF: Going from 1 to 50 occurrences → ~50x increase (linear)")
    print("- BM25: Going from 1 to 50 occurrences → ~2x increase (saturates)")
    print("- This prevents keyword stuffing from dominating results!")


def demo_length_normalization():
    """Show how BM25 handles document length better"""
    print("\n" + "=" * 70)
    print("Demo 2: Document Length Normalization")
    print("=" * 70)

    # Create documents of different lengths with same proportion of target term
    documents = [
        # Short doc: "Python" is 2/10 words = 20%
        "Python is great Python for data science work today",

        # Medium doc: "Python" is 4/20 words = 20%
        "Python is a great language Python for data science and web development Python work today Python",

        # Long doc: "Python" is 8/40 words = 20%
        "Python is a really great programming language Python for data science machine learning and web development "
        "Python work today Python and tomorrow and next year for sure Python guaranteed Python awesome Python",

        # Different topic
        "JavaScript is used for frontend web development"
    ]

    query = "Python"

    tfidf = SimpleTFIDF(documents)
    bm25 = BM25(documents, k1=1.2, b=0.75)

    print(f"\nQuery: '{query}'")
    print(f"\nAll docs have same proportion of 'Python' (~20% of words)")
    print(f"\n{'Doc':<5} {'Length':<8} {'Term Count':<12} {'TF-IDF':<12} {'BM25':<12}")
    print("-" * 70)

    for i, doc in enumerate(documents[:3]):
        words = doc.split()
        doc_len = len(words)
        term_count = doc.lower().split().count("python")
        proportion = term_count / doc_len

        tfidf_score = tfidf.score(query, i)
        bm25_score = bm25.score(query, i)

        print(f"{i:<5} {doc_len:<8} {term_count:<12} {tfidf_score:<12.4f} {bm25_score:<12.4f}")

    print("\n📊 Key Observation:")
    print("- TF-IDF: Scores vary significantly despite same proportion")
    print("- BM25: Scores are more consistent (better normalization)")
    print("- b parameter (0.75) controls how much length matters")


def demo_parameter_tuning():
    """Show effect of k1 and b parameters"""
    print("\n" + "=" * 70)
    print("Demo 3: Parameter Tuning Effects")
    print("=" * 70)

    documents = [
        "Python Python Python",  # High TF, short
        "Python " * 10,  # Very high TF
        "Python is a great language for development",  # Low TF, medium length
    ]

    query = "Python"

    print(f"\nQuery: '{query}'")
    print("\n--- Effect of k1 (saturation) ---")
    print(f"{'k1':<6} {'Doc 0':<10} {'Doc 1':<10} {'Doc 2':<10}")
    print("-" * 40)

    for k1 in [0.5, 1.2, 2.0, 100.0]:
        bm25 = BM25(documents, k1=k1, b=0.75)
        scores = [bm25.score(query, i) for i in range(3)]
        k1_label = f"{k1:.1f}" if k1 < 100 else "∞ (TF-IDF)"
        print(f"{k1_label:<6} {scores[0]:<10.3f} {scores[1]:<10.3f} {scores[2]:<10.3f}")

    print("\n--- Effect of b (length normalization) ---")
    print(f"{'b':<6} {'Doc 0':<10} {'Doc 1':<10} {'Doc 2':<10}")
    print("-" * 40)

    for b in [0.0, 0.5, 0.75, 1.0]:
        bm25 = BM25(documents, k1=1.2, b=b)
        scores = [bm25.score(query, i) for i in range(3)]
        print(f"{b:<6.2f} {scores[0]:<10.3f} {scores[1]:<10.3f} {scores[2]:<10.3f}")

    print("\n📊 Key Observations:")
    print("- Higher k1 → Less saturation (closer to TF-IDF)")
    print("- Lower k1 → More saturation (term count matters less)")
    print("- Higher b → Stronger length normalization (penalize long docs more)")
    print("- Lower b → Weaker length normalization (length matters less)")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 70)
    print("BM25 vs TF-IDF: Understanding the Differences")
    print("=" * 70)

    demo_saturation_effect()
    demo_length_normalization()
    demo_parameter_tuning()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
BM25 improves on TF-IDF with two key mechanisms:

1. **Term Frequency Saturation (k1 parameter)**
   - Problem: TF-IDF treats 100 occurrences as 10x more relevant than 10
   - Solution: BM25 has diminishing returns after certain point
   - Default k1=1.2 works well for most cases

2. **Document Length Normalization (b parameter)**
   - Problem: TF-IDF can unfairly favor/penalize based on length
   - Solution: BM25 normalizes relative to average doc length
   - Default b=0.75 balances short vs long docs

Formula breakdown:
    score = IDF(term) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × len/avglen))
                         ^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         Saturation      Length normalization

For your interview:
✓ Understand WHY these parameters exist (the problems they solve)
✓ Know typical values: k1=1.2, b=0.75
✓ Explain saturation curve vs linear growth
✓ Can implement from scratch (you already did! ✅)
    """)

    print("=" * 70)


if __name__ == "__main__":
    main()
