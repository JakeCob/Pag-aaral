# Challenge 03: Conversational RAG with Re-ranking (Advanced)

**Difficulty**: Advanced
**Time Estimate**: 60-75 minutes
**Interview Section**: Section 1 - Part C + Extensions

---

## 📋 Challenge Description

Build a **production-grade conversational RAG system** that:
1. Maintains conversation history and context
2. Uses hybrid search (from Challenge 02)
3. Re-ranks results using a cross-encoder
4. Generates answers with proper citations
5. Handles multi-turn conversations intelligently

This represents what you'd build in a real production environment.

---

## 🎯 Requirements

### Part A: Conversation Memory (20 min)

1. **ConversationMemory class** with:
   - `add_turn(user_msg: str, assistant_msg: str, session_id: str)`
   - `get_history(session_id: str, last_k: int = 5) -> List[Dict]`
   - `format_for_prompt(session_id: str) -> str` - Format history for LLM context
   - Token limiting (max 2000 tokens per session)

2. **Session Management**:
   - Each conversation has unique `session_id`
   - Store up to last 10 turns per session
   - Automatically prune old messages when token limit exceeded

### Part B: Cross-Encoder Re-ranking (15 min)

1. **CrossEncoderReranker class** with:
   - Initialize with `cross-encoder/ms-marco-MiniLM-L-6-v2` model
   - `rerank(query: str, documents: List[str], top_k: int) -> List[Tuple[str, float]]`
   - Returns documents sorted by relevance score

2. **Why Re-ranking?**
   - Initial retrieval (BM25/semantic) is fast but imprecise (top 20-50 docs)
   - Cross-encoder is slow but very accurate (re-rank top 20 → pick best 5)
   - Two-stage approach: Fast retrieval + Precise re-ranking

### Part C: Conversational RAG Pipeline (25 min)

1. **ConversationalRAG class** with complete workflow:

```python
async def query(
    question: str,
    session_id: str,
    top_k: int = 5,
    use_reranking: bool = True
) -> Dict[str, Any]:
    """
    Full RAG pipeline with conversation context.

    Returns:
        {
            "answer": str,
            "sources": List[str],
            "confidence": float,
            "conversation_history": List[Dict]
        }
    """
```

2. **Pipeline Steps**:
   1. Retrieve conversation history
   2. Contextualize current question (if follow-up)
   3. Hybrid search (retrieve top 20)
   4. Re-rank to top 5 (if enabled)
   5. Generate answer with LLM
   6. Extract citations/sources
   7. Update conversation memory
   8. Return structured response

---

## 📊 Example Workflow

```python
# Initialize system
rag = ConversationalRAG()

# Add documents
documents = [
    "LangChain is a framework for building LLM applications.",
    "LangChain supports multiple LLMs including OpenAI, Anthropic, and HuggingFace.",
    "LCEL (LangChain Expression Language) allows chaining components.",
    "LangChain provides retrievers, agents, and memory components."
]
rag.add_documents(documents)

# Turn 1
response1 = await rag.query(
    question="What is LangChain?",
    session_id="user_123"
)
print(response1["answer"])
# "LangChain is a framework for building LLM applications..."

# Turn 2 (follow-up question)
response2 = await rag.query(
    question="What LLMs does it support?",  # "it" refers to LangChain
    session_id="user_123"
)
print(response2["answer"])
# "LangChain supports multiple LLMs including OpenAI, Anthropic, and HuggingFace."

# The system knows "it" = "LangChain" from conversation history!
```

---

## ✅ Expected Output

```
=== Turn 1 ===
User: What is LangChain?

Retrieved Documents (before re-ranking):
1. [BM25: 8.5, Semantic: 0.82] LangChain is a framework for building LLM applications.
2. [BM25: 6.2, Semantic: 0.75] LangChain supports multiple LLMs...
3. [BM25: 5.8, Semantic: 0.71] LCEL allows chaining components...

Re-ranked Documents (top 3):
1. [Score: 0.94] LangChain is a framework for building LLM applications.
2. [Score: 0.87] LangChain provides retrievers, agents, and memory components.
3. [Score: 0.82] LangChain supports multiple LLMs...

Answer:
LangChain is a framework for building applications with large language models (LLMs).
It provides components like retrievers, agents, and memory to simplify LLM app development.

Sources: [Doc 1, Doc 4]
Confidence: 0.91

---

=== Turn 2 ===
User: What LLMs does it support?

Contextualized Query (with history):
"What LLMs does LangChain support?"

Retrieved Documents:
1. [Score: 0.96] LangChain supports multiple LLMs including OpenAI, Anthropic, and HuggingFace.
2. [Score: 0.68] LangChain is a framework...

Answer:
LangChain supports multiple LLM providers including OpenAI, Anthropic, and HuggingFace models.

Sources: [Doc 2]
Confidence: 0.95
```

---

## 🧪 Test Cases

### Test 1: Single-Turn Query
```python
response = await rag.query("What is LCEL?", session_id="test_1")
assert "LangChain Expression Language" in response["answer"].lower() or "lcel" in response["answer"].lower()
assert len(response["sources"]) > 0
assert response["confidence"] > 0.5
```

### Test 2: Multi-Turn Conversation
```python
# Turn 1
r1 = await rag.query("Tell me about LangChain", session_id="test_2")

# Turn 2 (pronoun reference)
r2 = await rag.query("What are its main components?", session_id="test_2")
assert "retriever" in r2["answer"].lower() or "agent" in r2["answer"].lower()

# Verify history is maintained
history = rag.memory.get_history("test_2")
assert len(history) == 2
```

### Test 3: Re-ranking Improves Results
```python
# Without re-ranking
r1 = await rag.query("LLM support", session_id="test_3", use_reranking=False)

# With re-ranking
r2 = await rag.query("LLM support", session_id="test_4", use_reranking=True)

# Re-ranking should give higher confidence
assert r2["confidence"] >= r1["confidence"]
```

### Test 4: Session Isolation
```python
# Session A
await rag.query("What is LangChain?", session_id="session_a")

# Session B
await rag.query("What is FastAPI?", session_id="session_b")

# Follow-up in session A should reference LangChain, not FastAPI
response = await rag.query("Tell me more about it", session_id="session_a")
assert "langchain" in response["answer"].lower()
assert "fastapi" not in response["answer"].lower()
```

---

## 💡 Implementation Tips

### Conversation Memory with Token Limiting
```python
from typing import List, Dict
import tiktoken

class ConversationMemory:
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.sessions = {}  # session_id -> List[{user, assistant}]
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def add_turn(self, user_msg: str, assistant_msg: str, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "user": user_msg,
            "assistant": assistant_msg
        })

        # Prune old messages if exceeding token limit
        self._prune_history(session_id)

    def _prune_history(self, session_id: str):
        """Remove oldest messages until under token limit"""
        while True:
            history_text = self.format_for_prompt(session_id)
            token_count = len(self.tokenizer.encode(history_text))

            if token_count <= self.max_tokens or len(self.sessions[session_id]) <= 1:
                break

            # Remove oldest turn
            self.sessions[session_id].pop(0)
```

### Cross-Encoder Re-ranking
```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
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
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Combine and sort
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return doc_score_pairs[:top_k]
```

### Query Contextualization
```python
async def contextualize_query(self, question: str, session_id: str) -> str:
    """
    Rewrite follow-up questions to be standalone using conversation history.

    Example:
        History: "What is LangChain?" -> "LangChain is a framework..."
        Follow-up: "What LLMs does it support?"
        Contextualized: "What LLMs does LangChain support?"
    """
    history = self.memory.get_history(session_id, last_k=3)

    if not history:
        return question  # No context needed

    # Use LLM to rewrite question with context
    history_text = self.memory.format_for_prompt(session_id)

    prompt = f"""Given the conversation history, rewrite the follow-up question to be standalone.

Conversation History:
{history_text}

Follow-up Question: {question}

Standalone Question (keep it concise):"""

    # Call LLM (OpenAI, Anthropic, etc.)
    contextualized = await self.llm.generate(prompt)

    return contextualized.strip()
```

### Answer Generation with Citations
```python
async def generate_answer(self, question: str, context_docs: List[str], session_id: str) -> Dict:
    """Generate answer with citations from context documents"""

    # Format context
    context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)])

    # Get conversation history
    history = self.memory.format_for_prompt(session_id)

    # Build prompt
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

    # Extract cited sources
    import re
    cited = re.findall(r'\[(\d+)\]', answer)
    sources = [context_docs[int(i)-1] for i in cited if 0 < int(i) <= len(context_docs)]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": min(len(sources) / max(len(context_docs), 1), 1.0)
    }
```

---

## 🎓 Key Concepts to Demonstrate

1. **Conversation State Management**: Session-based memory with token limits
2. **Query Contextualization**: Rewriting follow-up questions using history
3. **Two-Stage Retrieval**: Fast initial retrieval + slow precise re-ranking
4. **Cross-Encoder vs Bi-Encoder**:
   - Bi-encoder: Encodes query and doc separately, fast but less accurate
   - Cross-encoder: Encodes query+doc together, slow but very accurate
5. **Citation Extraction**: Parsing LLM output for source references
6. **Confidence Scoring**: Based on number of sources and relevance scores

---

## 🚀 Extensions (If Time Permits)

1. **Streaming Responses**: Stream answer tokens as they're generated
2. **Conversation Summarization**: Compress long histories into summaries
3. **Multi-Modal Context**: Include images, tables, code snippets
4. **Fact Verification**: Check if answer is faithful to sources
5. **Conversation Analytics**: Track topics, sentiment, user satisfaction

---

## 📚 Related Concepts

- **Contextual Compression**: Remove irrelevant parts of retrieved docs
- **Hypothetical Document Embeddings (HyDE)**: Generate hypothetical answer first, then search
- **Parent Document Retrieval**: Retrieve small chunks, return larger parent documents
- **Query Decomposition**: Break complex questions into sub-questions

---

**Time Allocation**:
- Conversation Memory: 20 min
- Cross-Encoder Re-ranking: 15 min
- Full RAG Pipeline: 25 min
- Testing & Debugging: 10 min
- **Total**: 70 min

**Good luck!** 🎯
