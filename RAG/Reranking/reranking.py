"""
RAG Reranking Example
Demonstrates how to improve retrieval precision using reranking
"""

import numpy as np
from typing import List, Dict
import os

# Install these packages:
# pip install cohere sentence-transformers openai chromadb

# ============================================================================
# Mock Data and Initial Retrieval Setup
# ============================================================================

# Sample documents for our knowledge base
DOCUMENTS = [
    "To reset your password, click 'Forgot Password' on the login page and follow the email instructions.",
    "Our company was founded in 2010 and has grown to 500 employees.",
    "Password recovery requires access to your registered email address.",
    "The latest product update includes dark mode and improved performance.",
    "If you can't access your email, contact support at support@example.com for manual password reset.",
    "Our office hours are Monday to Friday, 9 AM to 5 PM EST.",
    "Two-factor authentication adds an extra layer of security to your account.",
    "Premium plans start at $29.99 per month with annual discounts available.",
    "Account security settings can be found in the Settings > Security menu.",
    "We use industry-standard encryption to protect your data.",
]

def simple_embedding_search(query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
    """
    Simulates initial retrieval using embeddings.
    In production, you'd use a real vector database (Pinecone, Weaviate, ChromaDB, etc.)
    """
    # For demo purposes, we'll use simple keyword matching scores
    # In reality, you'd use actual embeddings (OpenAI, Sentence-Transformers, etc.)
    
    results = []
    query_lower = query.lower()
    
    for idx, doc in enumerate(documents):
        # Simple scoring based on keyword overlap (replace with real embeddings)
        doc_lower = doc.lower()
        score = sum(word in doc_lower for word in query_lower.split()) / len(query_lower.split())
        
        results.append({
            "id": idx,
            "text": doc,
            "score": score + np.random.uniform(0, 0.2)  # Add noise to simulate embedding similarity
        })
    
    # Sort by score and return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ============================================================================
# Method 1: Reranking with Cohere
# ============================================================================

def rerank_with_cohere(query: str, documents: List[str], top_n: int = 3) -> List[Dict]:
    """
    Rerank documents using Cohere's rerank API
    """
    import cohere
    
    # Initialize Cohere client
    co = cohere.Client(os.getenv("COHERE_API_KEY"))  # Set your API key
    
    # Call rerank API
    results = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model="rerank-english-v3.0"  # or "rerank-multilingual-v3.0"
    )
    
    # Format results
    reranked = []
    for result in results.results:
        reranked.append({
            "index": result.index,
            "text": documents[result.index],
            "relevance_score": result.relevance_score
        })
    
    return reranked


# ============================================================================
# Method 2: Reranking with Cross-Encoder
# ============================================================================

def rerank_with_cross_encoder(query: str, documents: List[str], top_n: int = 3) -> List[Dict]:
    """
    Rerank documents using a cross-encoder model (local, no API needed)
    """
    from sentence_transformers import CrossEncoder
    
    # Load cross-encoder model (this will download on first run)
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Create query-document pairs
    pairs = [[query, doc] for doc in documents]
    
    # Get relevance scores
    scores = model.predict(pairs)
    
    # Combine documents with scores
    results = [
        {"index": idx, "text": doc, "relevance_score": float(score)}
        for idx, (doc, score) in enumerate(zip(documents, scores))
    ]
    
    # Sort by score and return top_n
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_n]


# ============================================================================
# Complete RAG Pipeline with Reranking
# ============================================================================

def rag_with_reranking(query: str, rerank_method: str = "cross-encoder"):
    """
    Full RAG pipeline: Initial retrieval → Reranking → LLM
    """
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    # Step 1: Initial retrieval (get more candidates)
    print("Step 1: Initial Retrieval (Top 8 candidates)")
    print("-" * 70)
    initial_results = simple_embedding_search(query, DOCUMENTS, top_k=8)
    
    for i, result in enumerate(initial_results, 1):
        print(f"{i}. [Score: {result['score']:.3f}] {result['text'][:80]}...")
    
    # Extract documents for reranking
    candidate_docs = [r["text"] for r in initial_results]
    
    # Step 2: Reranking
    print(f"\nStep 2: Reranking with {rerank_method} (Top 3)")
    print("-" * 70)
    
    if rerank_method == "cohere":
        reranked = rerank_with_cohere(query, candidate_docs, top_n=3)
    else:  # cross-encoder
        reranked = rerank_with_cross_encoder(query, candidate_docs, top_n=3)
    
    for i, result in enumerate(reranked, 1):
        print(f"{i}. [Relevance: {result['relevance_score']:.3f}] {result['text'][:80]}...")
    
    # Step 3: Send to LLM (simulated)
    print(f"\nStep 3: Send top {len(reranked)} documents to LLM")
    print("-" * 70)
    print("These refined results would now be used as context for the LLM to generate an answer.")
    
    return reranked


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Example query
    query = "How can I reset my password if I lost access to my email?"
    
    # Run with cross-encoder (no API key needed)
    print("\n🔍 Using Cross-Encoder Reranking (Local)")
    rag_with_reranking(query, rerank_method="cross-encoder")
    
    # Uncomment to run with Cohere (requires API key)
    # print("\n\n🔍 Using Cohere Reranking (API)")
    # rag_with_reranking(query, rerank_method="cohere")
    
    # Compare: without reranking vs with reranking
    print("\n\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print("\nWithout reranking: You might get irrelevant docs in top results")
    print("With reranking: More precise, contextually relevant documents\n")