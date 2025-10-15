# ELGO AI - AI Engineer Coding Interview

Complete interview package for AI Engineer position at ELGO AI, including improved prompt, reference solutions, comprehensive tests, and grading rubrics.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup Instructions](#setup-instructions)
- [Interview Sections](#interview-sections)
- [Running Solutions](#running-solutions)
- [Testing](#testing)
- [Grading](#grading)
- [For Interviewers](#for-interviewers)
- [For Candidates](#for-candidates)

---

## Overview

This interview package evaluates AI Engineering skills across three critical dimensions:

1. **RAG System Implementation** - Building production-ready retrieval-augmented generation systems
2. **Algorithmic Problem Solving** - Data structures, algorithms, and system optimization
3. **System Design** - Multi-agent architectures and workflow orchestration

**Total Time**: ~3 hours
**Format**: Take-home coding challenge
**Tech Stack**: Python 3.9+, FastAPI, LangChain, ChromaDB, OpenAI

---

## Repository Structure

```
Coding Interview/
├── README.md                          # This file
├── INTERVIEW_PROMPT.md                # Improved interview questions
├── GRADING_RUBRIC.md                  # Detailed grading criteria
├── requirements.txt                   # Python dependencies
│
├── section1_rag_solution.py           # Section 1: Advanced RAG system
├── section2_cache_solution.py         # Section 2: Distributed cache
├── section3_multiagent_solution.py    # Section 3: Multi-agent workflow
│
└── test_solutions.py                  # Comprehensive test suite
```

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher (3.10/3.11 recommended)
- pip package manager
- Git
- OpenAI API key (for Section 1)
- 8GB+ RAM recommended
- Linux/macOS/WSL2 (Windows Subsystem for Linux)

### Installation

#### 1. Clone Repository (if applicable)

```bash
git clone <repository-url>
cd "Coding Interview"
```

#### 2. Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n elgo-interview python=3.10
conda activate elgo-interview
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (web framework)
- LangChain (RAG framework)
- ChromaDB (vector database)
- OpenAI SDK (LLM access)
- pytest (testing)
- And many more...

#### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_org_id  # Optional

# Database (if testing SQL agent)
DATABASE_URL=postgresql://user:password@localhost/dbname

# Optional: For production deployment
ENVIRONMENT=development
LOG_LEVEL=INFO
```

**Important**: Never commit `.env` files to version control!

#### 5. Verify Installation

```bash
# Test imports
python -c "import fastapi, langchain, chromadb; print('All imports successful!')"

# Run basic tests
python section2_cache_solution.py  # Should run cache tests
python section3_multiagent_solution.py  # Should run agent demo
```

---

## Interview Sections

### Section 1: Advanced RAG System (90 minutes)

**File**: `section1_rag_solution.py`

**Implements**:
- Multi-format document processing (txt, pdf, json, csv)
- Document versioning system
- Hybrid search (semantic + BM25)
- Cross-encoder re-ranking
- Conversation memory with context
- Faithfulness evaluation
- Query caching
- Metrics tracking

**Key Features**:
- FastAPI REST API
- ChromaDB vector storage
- LangChain integration
- Production-ready error handling
- Comprehensive logging

### Section 2: Distributed Cache (45 minutes)

**File**: `section2_cache_solution.py`

**Implements**:
- LRU (Least Recently Used) eviction policy
- TTL (Time To Live) support
- Thread-safe operations with locks
- Complex object serialization (numpy arrays, dicts)
- Statistics tracking (hits, misses, evictions)
- O(1) get and put operations

**Key Features**:
- Doubly-linked list + hashmap
- Exponential time complexity
- Read-write lock optimization
- Cache warming utilities
- Metrics collection

### Section 3: Multi-Agent Workflow System (60 minutes)

**File**: `section3_multiagent_solution.py`

**Implements**:
- Multiple specialized agents (RAG, SQL, Code, Analysis, Web Search)
- Intelligent query routing
- DAG-based workflow orchestration
- Parallel execution with asyncio
- Error handling with exponential backoff retry
- Circuit breaker pattern
- Execution tracing

**Key Features**:
- Async/await architecture
- Topological sort for DAG execution
- Result aggregation
- Agent metrics tracking
- Production-grade observability

---

## Running Solutions

### Section 1: RAG System

#### Start the Server

```bash
# Development mode with auto-reload
uvicorn section1_rag_solution:app --reload --port 8000

# Production mode
uvicorn section1_rag_solution:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Test the API

```bash
# Upload a document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@sample.txt"

# Query a document
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic?",
    "doc_id": "abc123",
    "session_id": "user1",
    "use_hybrid_search": true,
    "use_reranking": true
  }'

# List documents
curl "http://localhost:8000/documents"

# Get metrics
curl "http://localhost:8000/metrics"

# Health check
curl "http://localhost:8000/health"
```

#### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI

### Section 2: Cache System

#### Run Tests

```bash
# Run built-in tests
python section2_cache_solution.py
```

#### Use in Code

```python
from section2_cache_solution import DistributedCache
import numpy as np

# Create cache
cache = DistributedCache(max_size=100, default_ttl=3600)

# Store simple values
cache.put("user:123", {"name": "John", "age": 30})

# Store numpy arrays (embeddings)
embedding = np.array([0.1, 0.2, 0.3])
cache.put("embedding:doc1", embedding, ttl=7200)

# Retrieve
user_data = cache.get("user:123")
emb = cache.get("embedding:doc1")

# Check statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['current_size']}/{stats['max_size']}")
```

### Section 3: Multi-Agent System

#### Run Demo

```bash
# Run the built-in demo
python section3_multiagent_solution.py
```

#### Use in Code

```python
import asyncio
from section3_multiagent_solution import (
    RAGAgent, SQLAgent, CodeAgent,
    QueryRouter, WorkflowOrchestrator
)

async def main():
    # Initialize agents
    agents = [
        RAGAgent(),
        SQLAgent(),
        CodeAgent()
    ]

    # Create router and orchestrator
    router = QueryRouter(agents, confidence_threshold=0.3)
    orchestrator = WorkflowOrchestrator(max_retries=3)

    # Execute workflow
    result = await orchestrator.execute_workflow(
        "Explain machine learning and show me code examples",
        router
    )

    print(result)

asyncio.run(main())
```

---

## Testing

### Run All Tests

```bash
# Run with pytest (verbose)
pytest test_solutions.py -v

# Run with coverage
pytest test_solutions.py --cov --cov-report=html

# Run specific test class
pytest test_solutions.py::TestDistributedCache -v

# Run specific test
pytest test_solutions.py::TestDistributedCache::test_lru_eviction -v
```

### Test Categories

1. **Unit Tests**
   - Individual component functionality
   - Edge cases and error conditions
   - Input validation

2. **Integration Tests**
   - Multi-component workflows
   - End-to-end scenarios
   - API endpoint testing

3. **Performance Tests**
   - Benchmark cache operations
   - Parallel agent execution
   - Throughput testing

4. **Concurrency Tests**
   - Thread safety verification
   - Race condition detection
   - Lock contention testing

### Expected Test Results

```
================================ test session starts =================================
collected 30 items

test_solutions.py::TestDistributedCache::test_basic_put_get PASSED            [  3%]
test_solutions.py::TestDistributedCache::test_lru_eviction PASSED             [  6%]
test_solutions.py::TestDistributedCache::test_lru_ordering PASSED             [ 10%]
...
test_solutions.py::TestPerformance::test_cache_performance PASSED             [100%]

============================== 30 passed in 15.23s ================================
```

---

## Grading

### Evaluation Criteria

Solutions are evaluated on four dimensions:

1. **Correctness (40%)** - Does it work?
2. **Code Quality (30%)** - Is it well-written?
3. **Performance (15%)** - Is it efficient?
4. **Production Readiness (15%)** - Can it be deployed?

### Scoring

- **90-100**: Excellent - Hire strongly
- **75-89**: Good - Hire
- **60-74**: Satisfactory - Consider
- **<60**: Needs Improvement - Do not hire

See `GRADING_RUBRIC.md` for detailed scoring criteria.

### Self-Assessment Checklist

Before submitting, verify:

- [ ] All required features implemented
- [ ] Code runs without errors
- [ ] Tests pass
- [ ] Docstrings and comments added
- [ ] Type hints throughout
- [ ] Error handling implemented
- [ ] No hardcoded secrets
- [ ] README updated with usage instructions
- [ ] Edge cases handled
- [ ] Code follows PEP 8

---

## For Interviewers

### Conducting the Interview

#### Before the Interview

1. Send candidate:
   - `INTERVIEW_PROMPT.md`
   - `requirements.txt`
   - Setup instructions from this README

2. Set expectations:
   - Time limit: 3 hours total
   - Can use Google/docs/Stack Overflow
   - Should ask questions if unclear
   - Focus on working solution first, optimize later

#### During the Interview (if live coding)

1. **Observe**:
   - Problem-solving approach
   - Communication clarity
   - Time management
   - Response to hints

2. **Provide hints if stuck**:
   - "Have you considered...?"
   - "What about this edge case...?"
   - "How would you handle errors here?"

3. **Don't**:
   - Give direct solutions
   - Rush the candidate
   - Interrupt unnecessarily

#### After Submission

1. **Review Code**:
   - Run their solution
   - Execute test cases
   - Check for bugs
   - Review code quality

2. **Use Grading Rubric**:
   - Score each section
   - Fill out interviewer notes
   - Provide specific feedback

3. **Make Decision**:
   - Compare to rubric thresholds
   - Consider overall performance
   - Discuss with team if borderline

### Common Red Flags

- ❌ Code doesn't run at all
- ❌ Hardcoded API keys/secrets
- ❌ No error handling
- ❌ Copy-pasted code without understanding
- ❌ Missing core functionality
- ❌ No tests or documentation
- ❌ Asks to extend time significantly

### Green Flags

- ✅ Clean, readable code
- ✅ Comprehensive error handling
- ✅ Good test coverage
- ✅ Clear documentation
- ✅ Considers edge cases
- ✅ Production-minded (logging, monitoring)
- ✅ Asks clarifying questions

---

## For Candidates

### Preparation Tips

#### Before You Start

1. **Read the prompt carefully**
   - Understand all requirements
   - Note the evaluation criteria
   - Identify must-haves vs nice-to-haves

2. **Set up your environment**
   - Install all dependencies
   - Test that everything works
   - Prepare your IDE/editor

3. **Plan your approach**
   - Allocate time per section
   - Identify potential challenges
   - Plan testing strategy

#### During the Interview

1. **Start with the basics**
   - Get a working solution first
   - Add features incrementally
   - Test frequently

2. **Manage your time**
   - Section 1: 90 minutes
   - Section 2: 45 minutes
   - Section 3: 60 minutes
   - Leave time for testing and documentation

3. **Write clean code**
   - Use descriptive names
   - Add docstrings
   - Include type hints
   - Handle errors gracefully

4. **Test as you go**
   - Don't wait until the end
   - Test edge cases
   - Fix bugs immediately

5. **Document your work**
   - Add comments for complex logic
   - Update README with usage
   - Include examples

#### If You Get Stuck

1. **Break down the problem**
   - Simplify to minimal viable solution
   - Add complexity gradually

2. **Use resources**
   - Google is allowed
   - Check documentation
   - Review examples

3. **Ask questions**
   - Clarify requirements if unclear
   - Ask for hints if really stuck
   - Don't waste time guessing

### Time Management Strategy

**Recommended allocation**:

```
Section 1: RAG System (90 min)
  - Part A (Multi-format): 25 min
  - Part B (Hybrid search): 25 min
  - Part C (Memory): 20 min
  - Part D (Evaluation): 15 min
  - Testing & polish: 5 min

Section 2: Cache (45 min)
  - LRU implementation: 20 min
  - TTL support: 10 min
  - Thread safety: 8 min
  - Testing: 7 min

Section 3: Multi-Agent (60 min)
  - Agent classes: 15 min
  - Routing: 15 min
  - Orchestration: 20 min
  - Testing & polish: 10 min
```

### What We're Looking For

**Must Have**:
- Working solution that solves the problem
- Clean, readable code
- Basic error handling
- Some documentation

**Should Have**:
- Comprehensive error handling
- Good test coverage
- Type hints
- Performance considerations

**Nice to Have**:
- Advanced optimizations
- Monitoring/observability
- Production deployment considerations
- Bonus features

---

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'fastapi'
# Solution: Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### ChromaDB Issues

```bash
# Error: ChromaDB connection failed
# Solution: Clear ChromaDB directory
rm -rf ./chroma_db_advanced
```

#### OpenAI API Errors

```bash
# Error: openai.error.AuthenticationError
# Solution: Check API key
echo $OPENAI_API_KEY  # Should show your key
export OPENAI_API_KEY=sk-...  # Set if missing
```

#### Port Already in Use

```bash
# Error: Address already in use
# Solution: Use different port
uvicorn section1_rag_solution:app --port 8001
```

### Getting Help

- **Documentation**: See `INTERVIEW_PROMPT.md` for detailed requirements
- **Examples**: Check reference solutions for implementation patterns
- **Testing**: Run `test_solutions.py` to verify your code
- **Questions**: Contact the interviewer if requirements are unclear

---

## Additional Resources

### Recommended Reading

- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Python Async/Await Guide](https://realpython.com/async-io-python/)
- [LRU Cache Algorithm Explained](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU)

### Code Examples

Check the reference solutions for:
- RAG system architecture
- Cache implementation patterns
- Multi-agent workflow design
- Error handling strategies
- Testing approaches

---

## License

This interview package is proprietary to ELGO AI. Not for distribution.

---

## Contact

For questions about this interview package:
- **Technical Questions**: Contact engineering team
- **Process Questions**: Contact HR/recruiting

---

**Good luck with the interview!**

Remember: We're not looking for perfect code. We want to see:
- How you approach problems
- How you write production code
- How you handle complexity
- How you communicate through code

Focus on creating a solid, working solution that demonstrates your skills. If you run short on time, prioritize correctness over perfection.
