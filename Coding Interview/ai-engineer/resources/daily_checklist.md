# Daily Practice Checklist

Track your daily progress during the 7-day prep period.

---

## Day 1 (Jan 1) - RAG Foundations

### Study Tasks
- [ ] Read `rag_architecture_guide.md` (pages 1-5: Overview through Document Processing)
- [ ] Review `section1_rag_solution.py` lines 138-352 (Document Processing)
- [ ] Understand chunking strategy (RecursiveCharacterTextSplitter)
- [ ] Learn multi-format support (txt, pdf, json, csv)

### Practice Tasks
- [ ] Implement simple text chunking from scratch
- [ ] Test with sample document (create 10+ chunks)
- [ ] Verify chunk overlap is working

### Review Questions
- [ ] Can you explain why chunk overlap is needed?
- [ ] What's the difference between fixed-size and recursive chunking?
- [ ] How would you handle a 100MB PDF?

**Time Log**: ______ hours

---

## Day 2 (Jan 2) - Advanced RAG

### Study Tasks
- [ ] Read `rag_architecture_guide.md` (pages 6-10: Hybrid Search & Re-ranking)
- [ ] Review `section1_rag_solution.py` lines 456-654 (Hybrid + Re-ranking)
- [ ] Understand BM25 + semantic fusion (alpha parameter)
- [ ] Learn cross-encoder re-ranking pattern

### Practice Tasks
- [ ] Implement hybrid search (combine BM25 + vector scores)
- [ ] Test with queries that need keyword matching
- [ ] Time yourself: Build RAG system in <90 minutes

### Review Questions
- [ ] Why hybrid search instead of just semantic?
- [ ] When would you use alpha=0.8 vs alpha=0.4?
- [ ] What's the trade-off of re-ranking?

**Time Log**: ______ hours

---

## Day 3 (Jan 3) - Caching Deep Dive

### Study Tasks
- [ ] Read `caching_strategies_guide.md` (complete)
- [ ] Review `section2_cache_solution.py` lines 40-442 (LRU implementation)
- [ ] Understand doubly-linked list + HashMap pattern
- [ ] Learn TTL management (lazy vs active deletion)

### Practice Tasks
- [ ] Draw LRU data structure on paper
- [ ] Implement `get()` and `put()` from memory
- [ ] Add TTL support to your implementation
- [ ] Test thread safety with concurrent access

### Review Questions
- [ ] Why O(1) for both get and put?
- [ ] How does lazy deletion work?
- [ ] When would you use RW locks vs single lock?

**Time Log**: ______ hours

---

## Day 4 (Jan 4) - Multi-Agent Systems

### Study Tasks
- [ ] Read `multiagent_patterns_guide.md` (complete)
- [ ] Review `section3_multiagent_solution.py` lines 80-687 (Agents + Router)
- [ ] Understand agent abstraction (execute, can_handle)
- [ ] Learn DAG orchestration with topological sort

### Practice Tasks
- [ ] Implement base Agent class
- [ ] Create 2 specialized agents (RAG + SQL)
- [ ] Build simple query router
- [ ] Test parallel execution with asyncio.gather

### Review Questions
- [ ] How does query routing work?
- [ ] What's the difference between sequential and parallel execution?
- [ ] How does topological sort enable parallelism?

**Time Log**: ______ hours

---

## Day 5 (Jan 5) - Mock Interview #1

### Morning Preparation (1 hour)
- [ ] Quick review of all 3 reference guides (scan key sections)
- [ ] Review `INTERVIEW_PROMPT.md` in parent folder
- [ ] Set up environment (Python, libraries, API keys)
- [ ] Prepare note-taking setup

### Mock Interview (3 hours)
- [ ] **Section 1 (90 min)**: RAG System Implementation
  - [ ] Part A: Multi-format document processing
  - [ ] Part B: Hybrid search implementation
  - [ ] Part C: Conversation memory
  - [ ] Part D: Evaluation & caching
- [ ] **Section 2 (45 min)**: LRU Cache with TTL
  - [ ] LRU implementation
  - [ ] TTL support
  - [ ] Thread safety
- [ ] **Section 3 (60 min)**: Multi-Agent System
  - [ ] Agent base class + specialized agents
  - [ ] Query routing
  - [ ] DAG orchestration
  - [ ] Error handling

### Post-Interview Review (1 hour)
- [ ] Compare your code to reference solutions
- [ ] Note areas of struggle
- [ ] Grade yourself using `GRADING_RUBRIC.md`
- [ ] Create improvement plan for Day 6

**Struggles / Areas to Improve**:
```
1. ________________________________
2. ________________________________
3. ________________________________
```

**Score**: ______ / 100

---

## Day 6 (Jan 6) - Improvements & Production Patterns

### Fix Weak Areas
- [ ] Re-implement areas from Mock #1 that struggled
- [ ] Focus on: ________________________
- [ ] Practice: ________________________

### Production Patterns Review
- [ ] Error handling: Try-except at all boundaries
- [ ] Logging: INFO for actions, ERROR for failures
- [ ] Type hints: All function parameters and returns
- [ ] Docstrings: Public methods only (save time)
- [ ] Pydantic models: Request/response validation

### Code Quality Practice
- [ ] Review your Mock #1 code for improvements
- [ ] Add missing error handling
- [ ] Add type hints where missing
- [ ] Practice explaining code out loud (record yourself!)

### Interview Communication Practice
- [ ] Practice saying: "Let me start with the core functionality..."
- [ ] Practice asking: "Should I optimize for X or Y?"
- [ ] Practice explaining: "I'm using this pattern because..."

**Time Log**: ______ hours

---

## Day 7 (Jan 7) - Mock Interview #2 & Final Review

### Mock Interview #2 (3 hours)
Use different variants of problems:

- [ ] **Section 1**: RAG with streaming responses (different from Mock #1)
- [ ] **Section 2**: Write-through cache with Redis (different approach)
- [ ] **Section 3**: Multi-agent with conditional workflows

### Post-Interview Review
- [ ] Compare Mock #2 vs Mock #1 performance
- [ ] Note improvements
- [ ] Final list of concepts to review tomorrow morning

**Score**: ______ / 100
**Improvement vs Mock #1**: ______ points

### Final Review (Evening)
- [ ] Quick scan of all 3 reference guides (30 min each)
- [ ] Review key code snippets from section1/2/3_solution.py
- [ ] Prepare questions to ask interviewer
- [ ] Pack: Water, snacks, comfortable clothes
- [ ] Set alarm for interview day
- [ ] **Get good sleep!** 😴

---

## Day 8 (Jan 8) - INTERVIEW DAY! 🎯

### Morning Routine (2 hours before)
- [ ] Light breakfast
- [ ] Quick review of README.md key patterns (15 min)
- [ ] Review your notes from Mock #2
- [ ] Test environment one last time
- [ ] Breathe and stay calm! 🧘

### Pre-Interview (30 min before)
- [ ] Water bottle ready
- [ ] `section1/2/3_solution.py` open for quick reference
- [ ] Reference guides open in browser tabs
- [ ] Comfortable setup
- [ ] Deep breaths!

### During Interview
- [ ] Read requirements TWICE
- [ ] Ask clarifying questions
- [ ] Think out loud
- [ ] Start simple, then iterate
- [ ] Test your code with examples

### After Interview
- [ ] Note: How did it go? _________________________
- [ ] Note: What went well? _________________________
- [ ] Note: What to improve for next time? _________________________

---

## Progress Summary

| Day | Focus | Time Spent | Confidence (1-10) |
|-----|-------|------------|-------------------|
| 1 | RAG Foundations | _____ hrs | _____ |
| 2 | Advanced RAG | _____ hrs | _____ |
| 3 | Caching | _____ hrs | _____ |
| 4 | Multi-Agent | _____ hrs | _____ |
| 5 | Mock #1 | _____ hrs | _____ |
| 6 | Improvements | _____ hrs | _____ |
| 7 | Mock #2 + Review | _____ hrs | _____ |

**Total Study Time**: ______ hours

---

## Notes & Reflections

### What I Learned
```
Day 1: _________________________________
Day 2: _________________________________
Day 3: _________________________________
Day 4: _________________________________
Day 5: _________________________________
Day 6: _________________________________
Day 7: _________________________________
```

### Key Takeaways
```
1. _____________________________________
2. _____________________________________
3. _____________________________________
```

---

**Remember**: You have production-quality code to learn from. Focus on understanding patterns, not memorizing code!

Good luck! 🚀
