"""
Challenge 03: Conversational RAG with Re-ranking (Advanced)
DIY Starter Template

Complete the TODOs to build a production-grade conversational RAG system.
"""

from typing import List, Dict, Tuple, Any
import asyncio
import tiktoken
from sentence_transformers import CrossEncoder
import re

# Import from previous challenge
# Assume HybridRetriever is available
from intermediate_hybrid_search_solution import HybridRetriever


class ConversationMemory:
    """Manages conversation history with token limiting"""

    def __init__(self, max_tokens: int = 2000, max_turns: int = 10):
        """
        Initialize conversation memory.

        Args:
            max_tokens: Maximum tokens to keep in history
            max_turns: Maximum number of turns to keep per session
        """
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.sessions = {}  # session_id -> List[{"user": str, "assistant": str}]
        # TODO 1: Initialize tiktoken tokenizer
        self.tokenizer = None  # YOUR CODE HERE

    def add_turn(self, user_msg: str, assistant_msg: str, session_id: str):
        """
        Add a conversation turn to session history.

        Args:
            user_msg: User message
            assistant_msg: Assistant response
            session_id: Unique session identifier
        """
        # TODO 2: Initialize session if doesn't exist
        # YOUR CODE HERE

        # TODO 3: Append new turn
        # YOUR CODE HERE

        # TODO 4: Limit to max_turns
        # YOUR CODE HERE

        # TODO 5: Prune old messages if exceeding token limit
        # YOUR CODE HERE

    def _prune_history(self, session_id: str):
        """Remove oldest messages until under token limit"""
        # TODO 6: Implement token-based pruning
        # Hint: Use format_for_prompt() to get full history text
        # Count tokens with self.tokenizer.encode()
        # Remove oldest turns (pop from front) until under limit
        pass  # YOUR CODE HERE

    def get_history(self, session_id: str, last_k: int = 5) -> List[Dict]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            last_k: Number of recent turns to return

        Returns:
            List of conversation turns
        """
        # TODO 7: Return last_k turns for session
        # YOUR CODE HERE
        return []

    def format_for_prompt(self, session_id: str) -> str:
        """
        Format conversation history for LLM prompt.

        Args:
            session_id: Session identifier

        Returns:
            Formatted conversation string
        """
        # TODO 8: Format history as "User: ...\nAssistant: ...\n" format
        # YOUR CODE HERE
        return ""

    def clear_session(self, session_id: str):
        """Clear history for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


class CrossEncoderReranker:
    """Re-ranks documents using cross-encoder for better precision"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
        """
        # TODO 9: Initialize CrossEncoder model
        self.model = None  # YOUR CODE HERE

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Re-rank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of candidate documents
            top_k: Number of top results to return

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []

        # TODO 10: Create query-document pairs
        # Format: [[query, doc1], [query, doc2], ...]
        pairs = []  # YOUR CODE HERE

        # TODO 11: Score all pairs using cross-encoder
        scores = []  # YOUR CODE HERE (use self.model.predict)

        # TODO 12: Combine documents and scores, sort by score (descending)
        doc_score_pairs = []  # YOUR CODE HERE

        # TODO 13: Return top_k results
        # YOUR CODE HERE
        return []


class MockLLM:
    """Mock LLM for testing (replace with real OpenAI/Anthropic in production)"""

    async def generate(self, prompt: str) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # TODO 14: For DIY version, return a mock response
        # In production, call actual LLM API (OpenAI, Anthropic, etc.)

        # Simple mock: extract question and create basic response
        if "standalone question" in prompt.lower():
            # For contextualization
            lines = prompt.split("\n")
            for line in lines:
                if line.startswith("Follow-up Question:"):
                    return line.replace("Follow-up Question:", "").strip()
            return "What is LangChain?"

        else:
            # For answer generation
            return "Based on the provided context, [1] describes the main features."


class ConversationalRAG:
    """Complete conversational RAG pipeline"""

    def __init__(self, alpha: float = 0.6):
        """
        Initialize conversational RAG system.

        Args:
            alpha: Weight for semantic vs BM25 search
        """
        # TODO 15: Initialize components
        self.retriever = None  # YOUR CODE HERE (HybridRetriever)
        self.memory = None  # YOUR CODE HERE (ConversationMemory)
        self.reranker = None  # YOUR CODE HERE (CrossEncoderReranker)
        self.llm = None  # YOUR CODE HERE (MockLLM)

    def add_documents(self, documents: List[str]):
        """Add documents to retriever"""
        # TODO 16: Add documents to hybrid retriever
        pass  # YOUR CODE HERE

    async def contextualize_query(self, question: str, session_id: str) -> str:
        """
        Rewrite follow-up questions to be standalone using conversation history.

        Args:
            question: User's question
            session_id: Session identifier

        Returns:
            Contextualized standalone question
        """
        # TODO 17: Get conversation history
        history = []  # YOUR CODE HERE

        # TODO 18: If no history, return question as-is
        # YOUR CODE HERE

        # TODO 19: Build contextualization prompt
        history_text = self.memory.format_for_prompt(session_id)

        prompt = f"""Given the conversation history, rewrite the follow-up question to be standalone.

Conversation History:
{history_text}

Follow-up Question: {question}

Standalone Question (keep it concise):"""

        # TODO 20: Call LLM to contextualize
        contextualized = ""  # YOUR CODE HERE (await self.llm.generate)

        return contextualized.strip()

    async def generate_answer(
        self,
        question: str,
        context_docs: List[str],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Generate answer with citations from context documents.

        Args:
            question: User's question
            context_docs: Retrieved context documents
            session_id: Session identifier

        Returns:
            Dictionary with answer, sources, and confidence
        """
        # TODO 21: Format context with source numbers [1], [2], etc.
        context = ""  # YOUR CODE HERE

        # TODO 22: Get conversation history
        history = ""  # YOUR CODE HERE (self.memory.format_for_prompt)

        # TODO 23: Build answer generation prompt
        prompt = f"""You are a helpful assistant. Answer the question based on the context provided.
Include source numbers [1], [2], etc. in your answer.

Conversation History:
{history}

Context Documents:
{context}

Question: {question}

Answer (cite sources with [1], [2], etc.):"""

        # TODO 24: Generate answer
        answer = ""  # YOUR CODE HERE (await self.llm.generate)

        # TODO 25: Extract cited sources using regex
        # Pattern: \[(\d+)\] matches [1], [2], etc.
        cited_indices = []  # YOUR CODE HERE (use re.findall)
        sources = []  # YOUR CODE HERE (map indices to actual documents)

        # TODO 26: Calculate confidence score
        # Simple approach: confidence = min(len(sources) / len(context_docs), 1.0)
        confidence = 0.0  # YOUR CODE HERE

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }

    async def query(
        self,
        question: str,
        session_id: str,
        top_k: int = 5,
        use_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline with conversation context.

        Args:
            question: User's question
            session_id: Unique session identifier
            top_k: Number of documents to retrieve
            use_reranking: Whether to use cross-encoder reranking

        Returns:
            Dictionary with answer, sources, confidence, and conversation_history
        """
        # TODO 27: Contextualize query if follow-up question
        contextualized_question = ""  # YOUR CODE HERE

        # TODO 28: Retrieve documents (get more for reranking)
        retrieve_k = top_k * 3 if use_reranking else top_k
        retrieved_docs = []  # YOUR CODE HERE (self.retriever.search)

        # TODO 29: Re-rank if enabled
        if use_reranking:
            # YOUR CODE HERE: Use self.reranker.rerank()
            # Extract just documents from (doc, score) pairs
            context_docs = []
        else:
            context_docs = retrieved_docs[:top_k]

        # TODO 30: Generate answer
        result = {}  # YOUR CODE HERE (await self.generate_answer)

        # TODO 31: Update conversation memory
        # YOUR CODE HERE (self.memory.add_turn)

        # TODO 32: Add conversation history to result
        result["conversation_history"] = []  # YOUR CODE HERE

        return result


# =============================================================================
# TESTING CODE
# =============================================================================

async def test_conversation_memory():
    """Test conversation memory component"""
    print("\n" + "="*70)
    print("Test 1: Conversation Memory")
    print("="*70)

    memory = ConversationMemory(max_tokens=500, max_turns=5)

    # Add some turns
    memory.add_turn("What is Python?", "Python is a programming language.", "session1")
    memory.add_turn("What about Java?", "Java is also a programming language.", "session1")

    history = memory.get_history("session1")
    print(f"\nSession history ({len(history)} turns):")
    for i, turn in enumerate(history, 1):
        print(f"  Turn {i}:")
        print(f"    User: {turn['user']}")
        print(f"    Assistant: {turn['assistant']}")

    formatted = memory.format_for_prompt("session1")
    print(f"\nFormatted for prompt:\n{formatted}")


async def test_reranking():
    """Test cross-encoder reranking"""
    print("\n" + "="*70)
    print("Test 2: Cross-Encoder Re-ranking")
    print("="*70)

    reranker = CrossEncoderReranker()

    documents = [
        "Python is a high-level programming language.",
        "The sky is blue on a clear day.",
        "FastAPI is a Python web framework.",
        "Machine learning is a subset of AI."
    ]

    query = "Python programming"

    results = reranker.rerank(query, documents, top_k=3)

    print(f"\nQuery: '{query}'")
    print("\nRe-ranked results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [{score:.4f}] {doc}")


async def test_conversational_rag():
    """Test full conversational RAG system"""
    print("\n" + "="*70)
    print("Test 3: Conversational RAG")
    print("="*70)

    # Initialize system
    rag = ConversationalRAG(alpha=0.6)

    # Add documents
    documents = [
        "LangChain is a framework for building LLM applications.",
        "LangChain supports multiple LLMs including OpenAI, Anthropic, and HuggingFace.",
        "LCEL (LangChain Expression Language) allows chaining components.",
        "LangChain provides retrievers, agents, and memory components.",
        "FastAPI is a modern web framework for building APIs with Python.",
        "Python is a high-level programming language."
    ]

    rag.add_documents(documents)

    # Turn 1
    print("\n--- Turn 1 ---")
    response1 = await rag.query(
        question="What is LangChain?",
        session_id="test_session",
        use_reranking=True
    )

    print(f"Question: What is LangChain?")
    print(f"Answer: {response1['answer']}")
    print(f"Sources: {len(response1['sources'])} documents")
    print(f"Confidence: {response1['confidence']:.2f}")

    # Turn 2 (follow-up)
    print("\n--- Turn 2 (Follow-up) ---")
    response2 = await rag.query(
        question="What LLMs does it support?",
        session_id="test_session",
        use_reranking=True
    )

    print(f"Question: What LLMs does it support?")
    print(f"Answer: {response2['answer']}")
    print(f"Sources: {len(response2['sources'])} documents")
    print(f"Confidence: {response2['confidence']:.2f}")
    print(f"Conversation history: {len(response2['conversation_history'])} turns")

    print("\n✅ All tests completed!")


async def main():
    """Run all tests"""
    # TODO 33: Complete all TODOs above, then run these tests
    await test_conversation_memory()
    await test_reranking()
    await test_conversational_rag()


if __name__ == "__main__":
    asyncio.run(main())
