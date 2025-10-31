# Test Documents for RAG Practice

This folder contains sample documents for testing your RAG system implementations.

## Available Documents

1. **langchain_overview.txt** (3,800 words)
   - Topic: LangChain framework overview
   - Good for testing: RAG chains, vector stores, document retrieval
   - Sample queries:
     - "What is LangChain?"
     - "How does LCEL work?"
     - "What are the core components?"

2. **machine_learning_basics.txt** (4,200 words)
   - Topic: Machine learning fundamentals
   - Good for testing: Classification vs regression, overfitting, metrics
   - Sample queries:
     - "What is supervised learning?"
     - "How do you prevent overfitting?"
     - "What evaluation metrics should I use?"

3. **python_fastapi.txt** (4,500 words)
   - Topic: Python and FastAPI web development
   - Good for testing: FastAPI features, async support, dependency injection
   - Sample queries:
     - "How do I create a FastAPI endpoint?"
     - "What is dependency injection in FastAPI?"
     - "How do I test FastAPI applications?"

## Usage in Challenges

### Beginner Challenge
Use **one** document to practice basic RAG:
```python
file_path = "data/test_documents/langchain_overview.txt"
```

### Intermediate Challenge
Use **all three** documents to test hybrid search:
```python
docs = [
    "data/test_documents/langchain_overview.txt",
    "data/test_documents/machine_learning_basics.txt",
    "data/test_documents/python_fastapi.txt"
]
```

### Advanced Challenge
Use all documents for conversational RAG:
- Test follow-up questions
- Test cross-document queries
- Test conversation memory

## Testing Tips

**Good Test Queries:**
- "What is [specific concept]?" - Tests basic retrieval
- "How do I [perform task]?" - Tests procedural knowledge
- "Compare [A] and [B]" - Tests multi-chunk retrieval
- "What are the benefits of [X]?" - Tests semantic search

**Edge Cases to Test:**
- Query with typos: "What is machne lerning?"
- Query with no matches: "Tell me about quantum computing"
- Very short query: "FastAPI"
- Very long query: Multiple sentences

**Multi-Turn Conversations:**
```
Turn 1: "What is LangChain?"
Turn 2: "What are its main components?" (tests context understanding)
Turn 3: "How do I use agents?" (tests continued context)
```

## Document Statistics

| Document | Words | Chunks (500 chars) | Topics Covered |
|----------|-------|-------------------|----------------|
| LangChain | 3,800 | ~30 | RAG, chains, agents, LCEL |
| ML Basics | 4,200 | ~34 | Supervised/unsupervised, metrics |
| Python/FastAPI | 4,500 | ~36 | FastAPI, async, testing |

**Total**: ~12,500 words, ~100 chunks

Perfect for testing retrieval systems! 🚀
