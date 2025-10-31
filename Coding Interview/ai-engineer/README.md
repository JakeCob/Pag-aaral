# ELGO AI Engineer Interview Preparation

**Interview Date**: November 3, 2025
**Position**: AI Engineer
**Company**: ELGO AI (Singapore)
**Focus**: RAG Systems, Multi-Agent Workflows, LangChain

---

## 📁 Folder Structure

```
ai-engineer/
├── solutions/                                    # ⭐ YOUR PRODUCTION SOLUTIONS
│   ├── section1_rag_solution.py                 # 40KB - Complete RAG system
│   ├── section2_cache_solution.py               # 23KB - LRU cache with TTL
│   ├── section3_multiagent_solution.py          # 37KB - Multi-agent orchestrator
│   ├── INTERVIEW_PROMPT.md                      # 53KB - Interview questions
│   └── GRADING_RUBRIC.md                        # 17KB - Grading criteria
│
├── reference/                                    # 🎯 STUDY GUIDES (extract from solutions/)
│   ├── rag_architecture_guide.md                # 400+ lines on RAG systems
│   ├── caching_strategies_guide.md              # 600+ lines on LRU cache + TTL
│   └── multiagent_patterns_guide.md             # 500+ lines on multi-agent systems
│
├── tracks/                                       # Practice challenges
│   ├── 01_rag_system/                           # RAG challenges (beginner → advanced)
│   ├── 02_workflow_multiagent/                  # Multi-agent challenges
│   ├── 03_algorithms_cache/                     # Cache algorithm challenges
│   └── 04_production_api/                       # Production API challenges
│
├── practice/                                     # Your solutions
│   ├── beginner/
│   ├── intermediate/
│   └── advanced/
│
├── mock_interviews/                              # Full 3-hour mocks
│   ├── day5_mock1/
│   └── day7_mock2/
│
├── resources/                                    # Templates and checklists
│   ├── code_templates/
│   ├── daily_checklist.md
│   ├── interview_day_guide.md
│   └── time_management.md
│
└── README.md                                     # This file
```

---

## 🚀 Quick Start (5-Minute Setup)

### Step 1: Review Reference Guides

**Day 1-2**: RAG Architecture
```bash
cd ai-engineer/reference
cat rag_architecture_guide.md
```

**Day 3-4**: Caching + Multi-Agent
```bash
cat caching_strategies_guide.md
cat multiagent_patterns_guide.md
```

### Step 2: Your Production Solutions (NOW IN ai-engineer/solutions/)

You have **production-quality solutions** right here:
- `solutions/section1_rag_solution.py` (RAG) - 40KB, 1,320 lines
- `solutions/section2_cache_solution.py` (Cache) - 23KB, 816 lines
- `solutions/section3_multiagent_solution.py` (Multi-Agent) - 37KB, 1,312 lines
- `solutions/INTERVIEW_PROMPT.md` - The actual interview questions
- `solutions/GRADING_RUBRIC.md` - How you'll be evaluated

**These are your best resources!** The reference guides extract the key patterns from these.

```bash
# Access them easily
cd ai-engineer
cat solutions/section1_rag_solution.py
cat solutions/INTERVIEW_PROMPT.md
```

### Step 3: Practice Challenges

```bash
cd tracks/01_rag_system
# Read challenge: 01_beginner_simple_rag.md
# Try DIY version: 01_beginner_simple_rag_diy.py
# Compare with solutions/section1_rag_solution.py
```

---

## 📅 7-Day Study Plan (Jan 1-8, 2025)

### Day 1 (Jan 1) - RAG Foundations
- ✅ Read `rag_architecture_guide.md` (1-2 hours)
- ✅ Review `section1_rag_solution.py` focusing on:
  - Document processing (lines 138-352)
  - Hybrid search (lines 456-603)
  - Re-ranking (lines 605-654)
- ✅ Practice: Build simple RAG from scratch

### Day 2 (Jan 2) - Advanced RAG
- ✅ Study conversation memory pattern (lines 660-740)
- ✅ Study faithfulness evaluation (lines 826-874)
- ✅ Practice: Add conversation context to your RAG
- ✅ Time yourself: Aim for <90 minutes

### Day 3 (Jan 3) - Caching Deep Dive
- ✅ Read `caching_strategies_guide.md` (1-2 hours)
- ✅ Review `section2_cache_solution.py` focusing on:
  - LRU implementation (lines 40-442)
  - Thread safety (lines 114, 465-515)
  - TTL management (lines 148-156, 378-407)
- ✅ Practice: Implement LRU from memory

### Day 4 (Jan 4) - Multi-Agent Systems
- ✅ Read `multiagent_patterns_guide.md` (1-2 hours)
- ✅ Review `section3_multiagent_solution.py` focusing on:
  - Agent pattern (lines 80-184)
  - Query routing (lines 614-687)
  - DAG orchestration (lines 705-1072)
- ✅ Practice: Build 2-agent system

### Day 5 (Jan 5) - Mock Interview #1
- ✅ Full 3-hour timed mock (see `/mock_interviews/day5_mock1/`)
- ✅ Record areas of struggle
- ✅ Review against GRADING_RUBRIC.md in parent folder

### Day 6 (Jan 6) - Improvements + Production
- ✅ Fix weak areas from Mock #1
- ✅ Practice explaining code out loud
- ✅ Review error handling patterns
- ✅ Study production patterns (logging, metrics, error handling)

### Day 7 (Jan 7) - Mock Interview #2 + Final Review
- ✅ Second 3-hour mock with different problems
- ✅ Review all 3 reference guides (quick scan)
- ✅ Prepare questions to ask interviewer
- ✅ Get good sleep!

### Day 8 (Jan 8) - INTERVIEW DAY! 🎯

---

## 🎯 Key Concepts to Master

### RAG Systems (90 min section)
| Concept | Lines in section1 | Quick Ref |
|---------|-------------------|-----------|
| **Document Processing** | 138-352 | Multi-format (txt/pdf/json/csv), chunking with overlap |
| **Hybrid Search** | 456-603 | Semantic (vector) + BM25 (keyword), alpha=0.6 weighting |
| **Re-ranking** | 605-654 | Cross-encoder after initial retrieval |
| **Conversation Memory** | 660-740 | Session-based with token limiting |
| **Evaluation** | 826-874 | Faithfulness scoring with LLM-as-judge |
| **Caching** | 876-942 | Query cache with MD5 keys, TTL support |

### Cache Systems (45 min section)
| Concept | Lines in section2 | Quick Ref |
|---------|-------------------|-----------|
| **LRU Implementation** | 40-442 | Doubly-linked list + HashMap for O(1) |
| **TTL Management** | 148-156, 378-407 | Lazy deletion + manual cleanup |
| **Thread Safety** | 114, 465-515 | Single lock or RW lock |
| **Serialization** | 320-363 | Pickle for complex objects (numpy, dicts) |
| **Statistics** | 293-309 | Hit rate, evictions, expirations |

### Multi-Agent Systems (60 min section)
| Concept | Lines in section3 | Quick Ref |
|---------|-------------------|-----------|
| **Agent Pattern** | 80-184 | ABC with execute() and can_handle() |
| **Query Routing** | 614-687 | Confidence-based selection, top-k agents |
| **DAG Execution** | 705-1072 | Topological sort → parallel execution |
| **Retry Logic** | 853-923 | Exponential backoff (1s, 2s, 4s) |
| **Circuit Breaker** | 1174-1247 | 3 states: closed, open, half-open |
| **Tracing** | 1078-1168 | Execution logs for debugging |

---

## 🔑 Critical Patterns to Remember

### Pattern 1: Error Handling (All Sections)
```python
try:
    result = await operation()
    logger.info(f"Success: {result}")
    return result
except SpecificError as e:
    logger.error(f"Error: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    return default_value
```

### Pattern 2: Async Parallel Execution
```python
# Parallel execution (faster)
tasks = [agent1.execute(input), agent2.execute(input)]
results = await asyncio.gather(*tasks, return_exceptions=True)

# Sequential execution (when order matters)
result1 = await agent1.execute(input)
result2 = await agent2.execute(input)
```

### Pattern 3: Pydantic Models (FastAPI)
```python
class QueryRequest(BaseModel):
    question: str
    max_results: int = Field(default=5, ge=1, le=20)
    use_cache: bool = True

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    cached: bool
    latency_ms: float
```

### Pattern 4: Context Manager (Thread Safety)
```python
with self.lock:
    # All operations here are atomic
    if key in self.cache:
        node = self.cache[key]
        self._move_to_front(node)
        return node.value
```

---

## 💡 Interview Day Checklist

### Before Interview (Morning)
- [ ] Review all 3 reference guides (30 min quick scan)
- [ ] Have `section1/2/3_solution.py` files open for reference
- [ ] Test your environment (Python, libraries installed)
- [ ] Have OpenAI API key ready (if needed)
- [ ] Water bottle, comfortable setup

### During Interview
- [ ] **Read requirements TWICE** before coding
- [ ] Ask clarifying questions:
  - "Should I handle errors?"
  - "What file formats to support?"
  - "Is conversation memory needed?"
  - "Should I optimize for reads or writes?"
- [ ] **Start simple**, then add features
- [ ] **Explain as you code** (thought process)
- [ ] **Test with examples** before saying "done"

### Communication Tips
- ✅ "I'll start with the core functionality, then add error handling"
- ✅ "This uses O(1) operations because..."
- ✅ "In production, I'd add logging here"
- ✅ "I built a similar system at [your company]"
- ❌ Silent coding for 20 minutes
- ❌ "I don't know" (try: "Let me think through this...")

---

## 🎓 Resources

### Your Best Resources (Already Have!)
1. **`section1_rag_solution.py`** - Production RAG system
2. **`section2_cache_solution.py`** - LRU cache with TTL
3. **`section3_multiagent_solution.py`** - Multi-agent orchestrator
4. **`INTERVIEW_PROMPT.md`** - Actual interview questions
5. **`GRADING_RUBRIC.md`** - How you'll be evaluated

### Reference Guides (Created Today)
1. **`reference/rag_architecture_guide.md`** - RAG deep dive
2. **`reference/caching_strategies_guide.md`** - Cache implementation
3. **`reference/multiagent_patterns_guide.md`** - Agent patterns

### External Links
- LangChain: https://python.langchain.com/docs/
- ChromaDB: https://docs.trychroma.com/
- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/

---

## 📊 Progress Tracker

### Week 1 (Jan 1-4) - Study & Practice
- [ ] Day 1: RAG foundations
- [ ] Day 2: Advanced RAG
- [ ] Day 3: Caching
- [ ] Day 4: Multi-agent

### Week 2 (Jan 5-7) - Mocks & Polish
- [ ] Day 5: Mock interview #1
- [ ] Day 6: Improvements
- [ ] Day 7: Mock interview #2
- [ ] Day 8: **INTERVIEW** 🎯

---

## 🔥 Quick Tips

### Time Management (3-hour interview)
- **Section 1 (RAG)**: 90 minutes
  - 30 min: Part A (multi-format processing)
  - 25 min: Part B (hybrid search)
  - 25 min: Part C (conversation memory)
  - 10 min: Buffer
- **Section 2 (Cache)**: 45 minutes
  - 25 min: LRU implementation
  - 15 min: TTL + thread safety
  - 5 min: Testing
- **Section 3 (Multi-Agent)**: 60 minutes
  - 20 min: Agent base class + 3 agents
  - 15 min: Query routing
  - 25 min: DAG orchestration

### Code Quality Priorities
1. **Type hints** (always include)
2. **Docstrings** (for public methods)
3. **Error handling** (try-except at boundaries)
4. **Logging** (INFO for actions, ERROR for failures)
5. **Comments** (for complex logic only)

### What NOT to Worry About
- ❌ Perfect variable names (good enough is fine)
- ❌ 100% test coverage (basic tests sufficient)
- ❌ Premature optimization
- ❌ Fancy design patterns (KISS principle)

---

## 🎯 Your Competitive Advantages

Based on your existing solutions, you already know:
1. ✅ Production-grade RAG (multi-format, hybrid search, re-ranking)
2. ✅ Thread-safe LRU cache with TTL
3. ✅ Multi-agent DAG orchestration
4. ✅ Error handling patterns
5. ✅ FastAPI + Pydantic
6. ✅ Logging and metrics

**You're well-prepared!** Focus on:
- Explaining your thought process
- Writing clean, readable code
- Handling edge cases
- Time management

---

## 📞 Need Help?

**During Prep**:
- Re-read reference guides
- Review your section1/2/3_solution.py files
- Practice explaining code out loud

**During Interview**:
- Ask clarifying questions
- Think out loud
- Reference your past work ("I implemented similar caching in...")

---

## 🚀 Final Thoughts

You have:
- ✅ 7 days to prepare
- ✅ Production-quality code to learn from
- ✅ Comprehensive reference guides
- ✅ Clear study plan

**Focus on**:
1. Understanding patterns (not memorizing code)
2. Practicing explanations
3. Time management
4. Staying calm and thinking clearly

**You've got this!** 💪

---

**Last Updated**: October 30, 2024
**Created by**: Claude Code Interview Prep System

Good luck on January 8th! 🎯🚀
