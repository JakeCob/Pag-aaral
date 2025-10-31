"""
Challenge 03: Conversational RAG with Re-ranking (Advanced)
COMPLETE SOLUTION

This demonstrates a production-quality conversational RAG implementation.
"""

from typing import List, Dict, Tuple, Any
import asyncio
import tiktoken
from sentence_transformers import CrossEncoder
import re

# Import from previous challenge
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
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def add_turn(self, user_msg: str, assistant_msg: str, session_id: str):
        """
        Add a conversation turn to session history.

        Args:
            user_msg: User message
            assistant_msg: Assistant response
            session_id: Unique session identifier
        """
        # Initialize session if doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        # Append new turn
        self.sessions[session_id].append({
            "user": user_msg,
            "assistant": assistant_msg
        })

        # Limit to max_turns (keep most recent)
        if len(self.sessions[session_id]) > self.max_turns:
            self.sessions[session_id] = self.sessions[session_id][-self.max_turns:]

        # Prune old messages if exceeding token limit
        self._prune_history(session_id)

    def _prune_history(self, session_id: str):
        """Remove oldest messages until under token limit"""
        while len(self.sessions[session_id]) > 1:
            # Get full history text
            history_text = self.format_for_prompt(session_id)

            # Count tokens
            token_count = len(self.tokenizer.encode(history_text))

            # If under limit, we're done
            if token_count <= self.max_tokens:
                break

            # Remove oldest turn
            self.sessions[session_id].pop(0)

    def get_history(self, session_id: str, last_k: int = 5) -> List[Dict]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            last_k: Number of recent turns to return

        Returns:
            List of conversation turns
        """
        if session_id not in self.sessions:
            return []

        # Return last_k turns
        return self.sessions[session_id][-last_k:]

    def format_for_prompt(self, session_id: str) -> str:
        """
        Format conversation history for LLM prompt.

        Args:
            session_id: Session identifier

        Returns:
            Formatted conversation string
        """
        if session_id not in self.sessions:
            return ""

        formatted_lines = []
        for turn in self.sessions[session_id]:
            formatted_lines.append(f"User: {turn['user']}")
            formatted_lines.append(f"Assistant: {turn['assistant']}")

        return "\n".join(formatted_lines)

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
        self.model = CrossEncoder(model_name)

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

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs using cross-encoder
        scores = self.model.predict(pairs)

        # Combine documents and scores
        doc_score_pairs = list(zip(documents, scores))

        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return doc_score_pairs[:top_k]


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
        # Check if this is a contextualization request
        if "standalone question" in prompt.lower():
            # Extract the follow-up question
            lines = prompt.split("\n")
            follow_up = ""
            history_lines = []

            capture_history = False
            for line in lines:
                if "conversation history:" in line.lower():
                    capture_history = True
                    continue
                if "follow-up question:" in line:
                    follow_up = line.replace("Follow-up Question:", "").replace("Follow-up question:", "").strip()
                    break
                if capture_history and line.startswith("User:"):
                    # Extract main topic from history
                    history_lines.append(line)

            # Simple contextualization: replace pronouns with entities from history
            if history_lines and follow_up:
                # Extract entity from first user question (simplified)
                first_question = history_lines[0].replace("User:", "").strip()

                # Find main subject (naive approach: first capitalized word that's not common)
                import re
                entities = re.findall(r'\b([A-Z][a-z]+(?:Chain|API|[A-Z]+))\b', first_question)

                if entities:
                    entity = entities[0]
                    # Replace pronouns
                    contextualized = follow_up.replace(" it ", f" {entity} ")
                    contextualized = contextualized.replace(" its ", f" {entity}'s ")
                    contextualized = contextualized.replace(" It ", f" {entity} ")
                    return contextualized

            return follow_up if follow_up else "What are the main features?"

        else:
            # Answer generation
            # Extract context documents
            context_match = re.search(r'Context Documents:\s*(.*?)\n\nQuestion:', prompt, re.DOTALL)
            question_match = re.search(r'Question: (.*?)\n', prompt, re.DOTALL)

            if context_match and question_match:
                context = context_match.group(1)
                question = question_match.group(1).strip()

                # Extract document text (simple extraction)
                docs = re.findall(r'\[(\d+)\]\s+(.*?)(?=\n\[|\Z)', context, re.DOTALL)

                if docs:
                    # Use first relevant document
                    doc_num, doc_text = docs[0]
                    doc_text = doc_text.strip()

                    # Simple answer generation
                    return f"Based on the provided context, {doc_text} [{doc_num}]"

            return "Based on the provided context, the answer can be found in the documents."


class ConversationalRAG:
    """Complete conversational RAG pipeline"""

    def __init__(self, alpha: float = 0.6):
        """
        Initialize conversational RAG system.

        Args:
            alpha: Weight for semantic vs BM25 search
        """
        self.retriever = HybridRetriever(alpha=alpha)
        self.memory = ConversationMemory(max_tokens=2000, max_turns=10)
        self.reranker = CrossEncoderReranker()
        self.llm = MockLLM()

    def add_documents(self, documents: List[str]):
        """Add documents to retriever"""
        self.retriever.add_documents(documents)

    async def contextualize_query(self, question: str, session_id: str) -> str:
        """
        Rewrite follow-up questions to be standalone using conversation history.

        Args:
            question: User's question
            session_id: Session identifier

        Returns:
            Contextualized standalone question
        """
        # Get conversation history
        history = self.memory.get_history(session_id, last_k=3)

        # If no history, return question as-is
        if not history:
            return question

        # Build contextualization prompt
        history_text = self.memory.format_for_prompt(session_id)

        prompt = f"""Given the conversation history, rewrite the follow-up question to be standalone.

Conversation History:
{history_text}

Follow-up Question: {question}

Standalone Question (keep it concise):"""

        # Call LLM to contextualize
        contextualized = await self.llm.generate(prompt)

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
        # Format context with source numbers
        context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)])

        # Get conversation history
        history = self.memory.format_for_prompt(session_id)

        # Build answer generation prompt
        prompt = f"""You are a helpful assistant. Answer the question based on the context provided.
Include source numbers [1], [2], etc. in your answer.

Conversation History:
{history}

Context Documents:
{context}

Question: {question}

Answer (cite sources with [1], [2], etc.):"""

        # Generate answer
        answer = await self.llm.generate(prompt)

        # Extract cited sources using regex
        cited_indices = re.findall(r'\[(\d+)\]', answer)

        # Map indices to actual documents
        sources = []
        for idx_str in cited_indices:
            idx = int(idx_str) - 1  # Convert to 0-indexed
            if 0 <= idx < len(context_docs):
                sources.append(context_docs[idx])

        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)

        # Calculate confidence score
        if context_docs:
            confidence = min(len(unique_sources) / len(context_docs), 1.0)
        else:
            confidence = 0.0

        return {
            "answer": answer,
            "sources": unique_sources,
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
        # Step 1: Contextualize query if follow-up question
        contextualized_question = await self.contextualize_query(question, session_id)

        # Step 2: Retrieve documents (get more for reranking)
        retrieve_k = top_k * 3 if use_reranking else top_k
        retrieved_docs = self.retriever.search(contextualized_question, top_k=retrieve_k)

        # Step 3: Re-rank if enabled
        if use_reranking and retrieved_docs:
            # Re-rank using cross-encoder
            reranked = self.reranker.rerank(contextualized_question, retrieved_docs, top_k=top_k)
            # Extract just documents from (doc, score) pairs
            context_docs = [doc for doc, score in reranked]
        else:
            context_docs = retrieved_docs[:top_k]

        # Step 4: Generate answer
        result = await self.generate_answer(question, context_docs, session_id)

        # Step 5: Update conversation memory
        self.memory.add_turn(question, result["answer"], session_id)

        # Step 6: Add conversation history to result
        result["conversation_history"] = self.memory.get_history(session_id)

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
    memory.add_turn("Tell me about C++", "C++ is a compiled programming language.", "session1")

    history = memory.get_history("session1")
    print(f"\nSession history ({len(history)} turns):")
    for i, turn in enumerate(history, 1):
        print(f"  Turn {i}:")
        print(f"    User: {turn['user']}")
        print(f"    Assistant: {turn['assistant']}")

    formatted = memory.format_for_prompt("session1")
    print(f"\nFormatted for prompt:\n{formatted}")

    # Test token limiting
    print("\n--- Testing Token Limiting ---")
    for i in range(10):
        memory.add_turn(
            f"Question {i+4} about programming languages?",
            f"Answer {i+4}: This is a detailed response with many words to test token limiting.",
            "session1"
        )

    print(f"After adding 10 more turns, history size: {len(memory.get_history('session1'))}")
    print(f"Max turns setting: {memory.max_turns}")


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
        "Machine learning is a subset of AI.",
        "Python was created by Guido van Rossum."
    ]

    query = "Python programming"

    print(f"Query: '{query}'")
    print(f"Candidate documents: {len(documents)}")

    results = reranker.rerank(query, documents, top_k=3)

    print("\nRe-ranked results (top 3):")
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
        "Python is a high-level programming language.",
        "ChromaDB is a vector database for AI applications.",
        "Vector embeddings represent text as numerical arrays."
    ]

    print(f"Added {len(documents)} documents to knowledge base.")
    rag.add_documents(documents)

    # Turn 1
    print("\n" + "-"*70)
    print("Turn 1: Initial Question")
    print("-"*70)

    response1 = await rag.query(
        question="What is LangChain?",
        session_id="demo_session",
        use_reranking=True
    )

    print(f"User: What is LangChain?")
    print(f"Assistant: {response1['answer']}")
    print(f"Sources: {len(response1['sources'])} document(s) cited")
    print(f"Confidence: {response1['confidence']:.2f}")

    # Turn 2 (follow-up with pronoun)
    print("\n" + "-"*70)
    print("Turn 2: Follow-up Question (with pronoun 'it')")
    print("-"*70)

    response2 = await rag.query(
        question="What LLMs does it support?",
        session_id="demo_session",
        use_reranking=True
    )

    print(f"User: What LLMs does it support?")
    print(f"Assistant: {response2['answer']}")
    print(f"Sources: {len(response2['sources'])} document(s) cited")
    print(f"Confidence: {response2['confidence']:.2f}")
    print(f"Total conversation turns: {len(response2['conversation_history'])}")

    # Turn 3 (another follow-up)
    print("\n" + "-"*70)
    print("Turn 3: Another Follow-up")
    print("-"*70)

    response3 = await rag.query(
        question="What are its main components?",
        session_id="demo_session",
        use_reranking=True
    )

    print(f"User: What are its main components?")
    print(f"Assistant: {response3['answer']}")
    print(f"Sources: {len(response3['sources'])} document(s) cited")
    print(f"Confidence: {response3['confidence']:.2f}")

    # Test session isolation
    print("\n" + "-"*70)
    print("Test 4: Session Isolation")
    print("-"*70)

    response4 = await rag.query(
        question="What is FastAPI?",
        session_id="different_session",
        use_reranking=True
    )

    print(f"User (new session): What is FastAPI?")
    print(f"Assistant: {response4['answer']}")

    response5 = await rag.query(
        question="What is it used for?",
        session_id="different_session",
        use_reranking=True
    )

    print(f"User: What is it used for?")
    print(f"Assistant: {response5['answer']}")
    print("(Should refer to FastAPI, not LangChain)")

    print("\n✅ All tests completed!")


async def main():
    """Run all tests"""
    await test_conversation_memory()
    await test_reranking()
    await test_conversational_rag()


if __name__ == "__main__":
    asyncio.run(main())
