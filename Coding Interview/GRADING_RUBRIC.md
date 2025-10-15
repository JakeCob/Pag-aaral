# ELGO AI - AI Engineer Interview Grading Rubric

## Overview

This rubric provides detailed scoring criteria for evaluating candidate solutions across all three interview sections. Each section is worth 100 points, and the overall score is the weighted average.

**Weighting**:
- Section 1 (RAG System): 40%
- Section 2 (Cache Implementation): 30%
- Section 3 (Multi-Agent System): 30%

**Overall Grading Scale**:
- **Excellent (90-100)**: Hire strongly
- **Good (75-89)**: Hire
- **Satisfactory (60-74)**: Consider with reservations
- **Needs Improvement (<60)**: Do not hire

---

## Section 1: Advanced RAG System (100 points)

### Part A: Multi-Format Document Management (25 points)

#### File Format Support (10 points)
- **10 pts**: All 4 formats (txt, pdf, json, csv) correctly implemented
  - PDF: Proper text extraction with page tracking
  - JSON: Structured flattening preserves hierarchy
  - CSV: Tabular data formatted readably
  - TXT: Multiple encoding support
- **7 pts**: 3 formats working correctly
- **4 pts**: 2 formats working
- **0 pts**: <2 formats or critical bugs

#### Document Versioning (8 points)
- **8 pts**: Complete versioning system
  - Tracks multiple versions per document
  - Version metadata (timestamp, size, chunks)
  - Query-specific version support
  - No data loss between versions
- **5 pts**: Basic versioning, some features missing
- **2 pts**: Minimal versioning attempt
- **0 pts**: No versioning

#### Lifecycle Management (7 points)
- **7 pts**: All endpoints implemented correctly
  - DELETE removes all versions and embeddings
  - GET versions returns complete history
  - GET metadata returns accurate info
  - Proper error handling
- **4 pts**: Most endpoints work, minor issues
- **2 pts**: Some endpoints missing/broken
- **0 pts**: Lifecycle management not implemented

### Part B: Hybrid Search with Re-ranking (25 points)

#### Hybrid Search Implementation (15 points)
- **15 pts**: Production-quality hybrid search
  - BM25 correctly implemented and indexed
  - Semantic search via embeddings
  - Proper score normalization
  - Configurable alpha weighting
  - Handles edge cases (empty docs, no matches)
- **10 pts**: Both methods work, minor normalization issues
- **5 pts**: Only one method working
- **0 pts**: Neither method implemented

#### Re-ranking (10 points)
- **10 pts**: Cross-encoder re-ranking implemented
  - Model loaded correctly
  - Query-document pairs formatted properly
  - Scores integrated with base retrieval
  - Top-k selection works
- **6 pts**: Re-ranking works with minor issues
- **3 pts**: Attempted but not functional
- **0 pts**: Not implemented

### Part C: Conversation Memory & Context (20 points)

#### Conversation Memory (10 points)
- **10 pts**: Complete memory management
  - Stores messages per session
  - Enforces max_history limit
  - Token counting and limiting
  - Session isolation
  - Clear session functionality
- **6 pts**: Basic memory, some limits missing
- **3 pts**: Stores messages but no limiting
- **0 pts**: No memory implementation

#### Context-Aware Querying (10 points)
- **10 pts**: LLM uses conversation context
  - Context properly formatted in prompt
  - Pronouns/references resolved
  - Follow-up questions handled
  - Context doesn't pollute unrelated queries
- **6 pts**: Context used but not optimally
- **3 pts**: Minimal context awareness
- **0 pts**: No context integration

### Part D: Evaluation & Monitoring (20 points)

#### Faithfulness Evaluation (8 points)
- **8 pts**: LLM-based faithfulness scoring
  - Proper prompt engineering
  - Returns 0.0-1.0 score
  - Compares answer to sources
  - Handles edge cases
- **5 pts**: Basic evaluation, some issues
- **2 pts**: Placeholder implementation
- **0 pts**: Not implemented

#### Query Caching (6 points)
- **6 pts**: Complete caching system
  - Hashes query + doc_id for keys
  - TTL expiration works
  - Returns cached responses correctly
  - Clear cache endpoint
- **4 pts**: Caching works with minor issues
- **2 pts**: Basic caching attempt
- **0 pts**: No caching

#### Metrics Tracking (6 points)
- **6 pts**: Comprehensive metrics
  - Tracks hits, misses, latency, chunks
  - Calculates averages and rates
  - GET /metrics endpoint
  - Thread-safe updates
- **4 pts**: Basic metrics tracking
- **2 pts**: Some metrics missing
- **0 pts**: No metrics

### Code Quality (10 points)

- **10 pts**: Excellent code quality
  - Clear structure and organization
  - Comprehensive docstrings
  - Type hints throughout
  - Proper error handling
  - Follows PEP 8
  - No code smells
- **7 pts**: Good code quality, minor issues
- **4 pts**: Functional but messy
- **0 pts**: Poor code quality

---

## Section 2: Distributed Cache (100 points)

### LRU Implementation (30 points)

#### Doubly-Linked List (15 points)
- **15 pts**: Perfect DLL implementation
  - Head/tail pointers maintained correctly
  - All operations preserve list integrity
  - No memory leaks
  - Edge cases handled (empty list, single item)
- **10 pts**: Working DLL with minor issues
- **5 pts**: DLL attempted but buggy
- **0 pts**: No DLL or fundamentally broken

#### HashMap Integration (10 points)
- **10 pts**: O(1) key lookup
  - HashMap correctly maps keys to nodes
  - Updates on get/put operations
  - No orphaned references
- **6 pts**: Works but not optimal
- **3 pts**: Has performance issues
- **0 pts**: Not implemented

#### LRU Ordering (5 points)
- **5 pts**: Perfect LRU ordering
  - Get moves item to front
  - Put adds to front
  - Evicts from tail correctly
- **3 pts**: Mostly correct, minor bugs
- **1 pt**: Eviction wrong
- **0 pts**: No LRU ordering

### TTL Support (20 points)

#### TTL Storage & Checking (12 points)
- **12 pts**: Complete TTL implementation
  - Stores expiry timestamp with each node
  - Lazy deletion on get()
  - Custom TTL per entry supported
  - Default TTL used when not specified
- **8 pts**: TTL works with minor issues
- **4 pts**: Basic TTL, no custom support
- **0 pts**: No TTL

#### Expiration Handling (8 points)
- **8 pts**: Proper expiration handling
  - Expired items return None
  - Stats updated (expirations counter)
  - cleanup_expired() method works
  - No stale data returned
- **5 pts**: Mostly works, edge cases missed
- **2 pts**: Incomplete handling
- **0 pts**: No expiration handling

### Thread Safety (20 points)

#### Locking Mechanism (12 points)
- **12 pts**: Perfect thread safety
  - Lock protects all critical sections
  - No race conditions
  - No deadlocks
  - Minimal lock contention
- **8 pts**: Thread-safe with inefficiencies
- **4 pts**: Some race conditions possible
- **0 pts**: Not thread-safe

#### Concurrent Access Tests (8 points)
- **8 pts**: Comprehensive concurrency testing
  - Multiple threads verified
  - No data corruption
  - Stats remain consistent
- **5 pts**: Basic testing done
- **2 pts**: Minimal testing
- **0 pts**: No concurrency tests

### Serialization (15 points)

#### Complex Object Support (10 points)
- **10 pts**: Handles all object types
  - Numpy arrays serialized correctly
  - Dicts and lists preserved
  - Custom objects supported
  - Proper error handling
- **6 pts**: Most types work
- **3 pts**: Basic types only
- **0 pts**: No serialization

#### Performance (5 points)
- **5 pts**: Efficient serialization
  - Uses HIGHEST_PROTOCOL
  - Minimal overhead
  - No unnecessary copies
- **3 pts**: Works but slow
- **0 pts**: Very inefficient

### Statistics & Features (10 points)

- **10 pts**: Complete statistics
  - Hit/miss tracking
  - Eviction counting
  - Hit rate calculation
  - Additional utilities (contains, remove, get_keys)
- **6 pts**: Basic stats present
- **3 pts**: Minimal stats
- **0 pts**: No statistics

### Code Quality (5 points)

- **5 pts**: Excellent code quality
- **3 pts**: Good code quality
- **1 pt**: Acceptable
- **0 pts**: Poor

---

## Section 3: Multi-Agent Workflow System (100 points)

### Agent Architecture (25 points)

#### Base Agent Class (8 points)
- **8 pts**: Well-designed base class
  - Abstract methods properly defined
  - Common functionality in base (timing, metrics)
  - Clear interface (execute, can_handle)
  - Proper inheritance
- **5 pts**: Functional base class, some issues
- **2 pts**: Minimal abstraction
- **0 pts**: No base class

#### Specialized Agents (12 points)
- **12 pts**: All 3+ agents implemented
  - RAG, SQL, and Code agents functional
  - Each returns proper AgentOutput
  - Confidence scores meaningful
  - Error handling in each
- **8 pts**: 3 agents, some incomplete
- **4 pts**: 2 agents working
- **0 pts**: <2 agents

#### Agent Capabilities (5 points)
- **5 pts**: Agents demonstrate realistic capabilities
  - can_handle() uses intelligent logic
  - Execution simulates real behavior
  - Confidence scoring makes sense
- **3 pts**: Basic capabilities
- **0 pts**: Placeholder agents

### Query Routing (20 points)

#### Classification Logic (10 points)
- **10 pts**: Intelligent routing
  - Keyword matching implemented
  - Pattern recognition used
  - Confidence scores meaningful
  - Handles ambiguous queries
- **6 pts**: Basic routing works
- **3 pts**: Simple keyword matching only
- **0 pts**: Random/no routing

#### Multi-Agent Selection (10 points)
- **10 pts**: Selects multiple agents correctly
  - Confidence threshold respected
  - max_agents limit enforced
  - Returns sorted by confidence
  - Handles no-match case
- **6 pts**: Selection works with issues
- **3 pts**: Basic selection
- **0 pts**: No multi-agent support

### Workflow Orchestration (35 points)

#### DAG Construction (10 points)
- **10 pts**: Correct DAG implementation
  - build_dependency_graph() works
  - topological_sort() implemented (Kahn's or DFS)
  - Returns execution levels
  - Detects/handles cycles
- **6 pts**: DAG works for simple cases
- **3 pts**: Linear execution only
- **0 pts**: No DAG support

#### Parallel Execution (12 points)
- **12 pts**: True parallel execution
  - Uses asyncio.gather()
  - Independent steps run concurrently
  - Context properly shared
  - Results aggregated correctly
- **8 pts**: Parallel attempted, some issues
- **4 pts**: Sequential with async
- **0 pts**: No parallelism

#### Error Handling & Retries (8 points)
- **8 pts**: Robust error handling
  - Exponential backoff retry
  - Max retries enforced
  - Timeout handling
  - Failed steps don't crash workflow
- **5 pts**: Basic retry logic
- **2 pts**: Minimal error handling
- **0 pts**: No error handling

#### Result Aggregation (5 points)
- **5 pts**: Intelligent aggregation
  - Combines multiple outputs
  - Uses confidence scores
  - Handles failures gracefully
- **3 pts**: Basic aggregation
- **0 pts**: No aggregation

### Observability (10 points)

#### Execution Tracing (6 points)
- **6 pts**: Complete tracing
  - WorkflowExecutionTracer implemented
  - Logs all step executions
  - Timing information captured
  - Trace retrieval works
- **4 pts**: Basic tracing
- **0 pts**: No tracing

#### Logging & Metrics (4 points)
- **4 pts**: Comprehensive logging
  - Structured logging throughout
  - Agent metrics collected
  - Meaningful log messages
- **2 pts**: Basic logging
- **0 pts**: No logging

### Code Quality & Design (10 points)

- **10 pts**: Excellent design
  - Clean architecture
  - SOLID principles
  - Well-documented
  - Extensible design
- **6 pts**: Good design
- **3 pts**: Acceptable
- **0 pts**: Poor design

---

## Bonus Points (up to +15 points)

### Section 1 Bonuses (+10 max)
- **+3**: Unit tests for RAG components
- **+2**: Docker containerization
- **+2**: Rate limiting middleware
- **+2**: Authentication/API keys
- **+1**: Swagger documentation examples

### Section 2 Bonuses (+5 max)
- **+2**: Read-write lock optimization
- **+2**: Cache warming utility
- **+1**: Metrics export/monitoring

### Section 3 Bonuses (+5 max)
- **+2**: Circuit breaker pattern
- **+2**: Comprehensive integration tests
- **+1**: Performance benchmarks

---

## Evaluation Guidelines

### How to Grade

1. **Test Each Feature Systematically**
   - Run the code with provided test cases
   - Try edge cases and error conditions
   - Verify outputs match specifications

2. **Review Code Quality**
   - Read through implementation
   - Check for best practices
   - Look for code smells

3. **Assess Production Readiness**
   - Could this code go to production?
   - How much work to make it production-ready?
   - Are there security concerns?

### Common Deductions

**Correctness Issues (-5 to -20 per issue)**
- Crashes on valid input
- Returns incorrect results
- Doesn't handle edge cases
- Memory leaks

**Code Quality Issues (-2 to -10 per issue)**
- No type hints
- Missing docstrings
- Poor variable names
- Inconsistent formatting
- No error handling
- Hard-coded values (API keys, paths)

**Performance Issues (-5 to -15)**
- Incorrect time complexity
- Inefficient algorithms
- Unnecessary operations
- Memory waste

**Missing Features (-points as specified in rubric)**
- Each unimplemented required feature loses allocated points

### Partial Credit

Award partial credit for:
- **Attempted but incomplete** (~40-60% of points)
- **Working with bugs** (~60-80% of points)
- **Works but inefficient** (~70-85% of points)

### Red Flags (Automatic Fail)

- **Plagiarism**: Code copied without understanding
- **Non-functional**: Doesn't run at all
- **Security vulnerabilities**: SQL injection, hardcoded secrets
- **No attempt**: Blank or template code only

---

## Scoring Examples

### Section 1 Example: Good Solution (82/100)

- **Part A (22/25)**: All formats work, versioning complete, one endpoint has minor bug
- **Part B (20/25)**: Hybrid search works, re-ranking not implemented
- **Part C (18/20)**: Memory works well, context integration has minor issues
- **Part D (15/20)**: Basic faithfulness, caching works, missing some metrics
- **Code Quality (7/10)**: Good code, some docstrings missing

**Overall**: 82% = Good, recommend hire

### Section 2 Example: Excellent Solution (95/100)

- **LRU (29/30)**: Perfect except one edge case
- **TTL (20/20)**: Complete implementation
- **Thread Safety (19/20)**: Minor lock contention issue
- **Serialization (15/15)**: All types supported
- **Statistics (10/10)**: Complete
- **Code Quality (5/5)**: Excellent
- **Bonus (+3)**: RW locks implemented

**Overall**: 95% = Excellent, hire strongly

### Section 3 Example: Needs Improvement (55/100)

- **Agent Architecture (15/25)**: Only 2 agents, basic implementation
- **Query Routing (10/20)**: Simple keyword matching only
- **Workflow Orchestration (15/35)**: Sequential execution, basic retry
- **Observability (5/10)**: Minimal logging
- **Code Quality (4/10)**: Messy code, no docstrings

**Overall**: 55% = Do not hire

---

## Final Recommendation Matrix

| Overall Score | Section 1 | Section 2 | Section 3 | Recommendation |
|---------------|-----------|-----------|-----------|----------------|
| 90-100 | 85+ | 85+ | 85+ | **Hire Strongly** - Excellent candidate |
| 85-89 | 80+ | 80+ | 75+ | **Hire** - Strong candidate |
| 75-84 | 70+ | 70+ | 70+ | **Hire** - Solid candidate |
| 65-74 | 60+ | 60+ | 60+ | **Consider** - Has potential, needs growth |
| 60-64 | 55+ | 55+ | 55+ | **Borderline** - Discuss with team |
| <60 | <55 | <55 | <55 | **Do Not Hire** - Not ready |

### Special Cases

- **Uneven Performance**: If candidate excels in one section but fails others, discuss:
  - Section 1 strong: Good for RAG-focused role
  - Section 2 strong: Good for infrastructure/backend
  - Section 3 strong: Good for architecture/design

- **Time Management**: If candidate completes 2/3 sections excellently but runs out of time:
  - Consider for hire with note about time management
  - May indicate perfectionism over pragmatism

---

## Interviewer Notes Template

```markdown
## Candidate: [Name]
## Date: [Date]
## Interviewer: [Name]

### Section 1: RAG System (__ /100)
- Part A: __ /25
- Part B: __ /25
- Part C: __ /20
- Part D: __ /20
- Code Quality: __ /10

**Strengths**:
-

**Weaknesses**:
-

**Notable Observations**:
-

---

### Section 2: Cache (__ /100)
- LRU: __ /30
- TTL: __ /20
- Thread Safety: __ /20
- Serialization: __ /15
- Statistics: __ /10
- Code Quality: __ /5

**Strengths**:
-

**Weaknesses**:
-

---

### Section 3: Multi-Agent (__ /100)
- Agents: __ /25
- Routing: __ /20
- Orchestration: __ /35
- Observability: __ /10
- Code Quality: __ /10

**Strengths**:
-

**Weaknesses**:
-

---

### Overall Assessment

**Total Score**: __ / 100
**Weighted Score**: __

**Recommendation**: [ ] Hire Strongly [ ] Hire [ ] Consider [ ] Do Not Hire

**Justification**:


**Additional Comments**:


**Interview Behavior**:
- Communication:
- Problem-solving approach:
- Time management:
- Response to hints:
```

---

**Note to Graders**: Be fair but rigorous. Remember that this is for a startup position where engineers need to ship production code with minimal oversight. Code quality and completeness matter as much as correctness.
