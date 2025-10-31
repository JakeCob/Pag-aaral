# Interview Day Guide - January 8, 2025

**Position**: AI Engineer at ELGO AI (Singapore)
**Format**: 3-hour technical interview (likely remote)
**Focus**: RAG Systems, Caching, Multi-Agent Orchestration

---

## ⏰ Timeline for Interview Day

### 2 Hours Before Interview
- [ ] Light breakfast (avoid heavy meal)
- [ ] Coffee/tea if that's your routine (not too much!)
- [ ] Review README.md key patterns section (15 min)
- [ ] Quick scan of your Mock #2 feedback

### 1 Hour Before Interview
- [ ] Test your setup:
  - [ ] Python environment working
  - [ ] Required libraries installed (langchain, chromadb, fastapi, etc.)
  - [ ] OpenAI API key loaded (if needed)
  - [ ] IDE/editor ready
- [ ] Open reference materials:
  - [ ] `section1/2/3_solution.py` files
  - [ ] 3 reference guides in browser tabs
- [ ] Bathroom break
- [ ] Deep breathing exercises (5 min)

### 30 Minutes Before
- [ ] Close all distractions (Slack, email, phone on silent)
- [ ] Water bottle within reach
- [ ] Comfortable clothes
- [ ] Good lighting for camera (if video call)
- [ ] Join meeting link 5 min early

---

## 🎯 Interview Structure (Expected)

### Introduction (10 min)
- Interviewer introduces themselves and ELGO AI
- You introduce yourself
- Overview of interview format

**Your Introduction Template**:
> "Hi, I'm [name]. I have [X] years of experience in AI/ML engineering. Most recently, I've worked on [RAG systems / LLM applications / production ML]. I'm excited about ELGO AI because [reason related to job description]. I'm looking forward to today's interview!"

### Section 1: RAG System (90 minutes)

**Expected Tasks**:
- Part A: Multi-format document processing
- Part B: Hybrid search (semantic + BM25)
- Part C: Conversation memory
- Part D: Evaluation and caching

**Time Allocation**:
- 30 min: Part A
- 25 min: Part B
- 25 min: Part C
- 10 min: Buffer for questions/debugging

**Quick Reference**:
- Document formats: txt, pdf, json, csv (see `section1_rag_solution.py:158-277`)
- Chunking: RecursiveCharacterTextSplitter with overlap
- Hybrid search: `alpha * semantic + (1-alpha) * bm25` (alpha=0.6)
- Re-ranking: Cross-encoder after retrieval
- Memory: Deque with token limiting

### Section 2: Caching (45 minutes)

**Expected Tasks**:
- LRU cache implementation
- TTL support
- Thread safety
- Statistics tracking

**Time Allocation**:
- 25 min: Core LRU (doubly-linked list + HashMap)
- 15 min: TTL + thread safety
- 5 min: Testing

**Quick Reference**:
- Data structure: DLL + HashMap for O(1)
- Operations: `_move_to_front()`, `_evict_lru()`, `_remove_node()`
- TTL: Lazy deletion on get() + optional cleanup
- Thread safety: `with self.lock:` context manager

### Section 3: Multi-Agent System (60 minutes)

**Expected Tasks**:
- Part A: Agent base class + specialized agents
- Part B: Query routing
- Part C: DAG workflow orchestration
- Part D: Error handling and tracing

**Time Allocation**:
- 20 min: Base class + 3 agents
- 15 min: Router with confidence scoring
- 25 min: Orchestrator with topological sort

**Quick Reference**:
- Pattern: ABC with `execute()` and `can_handle()`
- Routing: Keyword/regex matching → confidence scores
- DAG: Topological sort (Kahn's algorithm)
- Parallel: `asyncio.gather(*tasks, return_exceptions=True)`
- Retry: Exponential backoff (1s, 2s, 4s)

### Wrap-up (10 minutes)
- Questions for interviewer
- Next steps in process
- Thank you and follow-up

---

## 💬 Communication Templates

### Starting Each Section
> "Let me make sure I understand the requirements correctly..."
> (Repeat back what you heard)
> "Should I prioritize [X] or [Y]?"
> "Any specific edge cases you want me to handle?"

### While Coding
> "I'm going to start with the core functionality, then add error handling..."
> "I'm using a doubly-linked list here because it gives O(1) removal..."
> "This is similar to a system I built at [company] where..."

### When Stuck
> "Let me think through this step by step..." (think out loud)
> "I'm considering two approaches: [A] and [B]. [A] is simpler but [B] is more scalable..."
> "Can I ask a clarifying question about [specific requirement]?"

### DON'T Say
- ❌ "I don't know" (try: "Let me think through this...")
- ❌ "This is hard" (try: "This is interesting, let me break it down...")
- ❌ Silent coding for 10+ minutes
- ❌ "I've never done this before"

---

## 📋 Pre-Coding Checklist

Before writing any code, DO THIS:

1. **Read Requirements Twice**
   - First read: Get overall picture
   - Second read: Note specific requirements

2. **Ask Clarifying Questions** (Write these down beforehand!)
   - "What file formats should I support?"
   - "Should I handle errors or focus on happy path?"
   - "Is performance critical here?"
   - "Should this be production-ready or MVP?"
   - "Any specific libraries I should use?"

3. **Explain Your Approach** (30 seconds)
   - "I'll create 3 classes: DocumentProcessor, VectorStore, and RAG..."
   - "First I'll implement the core logic, then add error handling..."

4. **Start Simple**
   - Core functionality first
   - Error handling second
   - Optimization third (if time)

---

## ⚡ Quick Reference: Core Patterns

### Pattern 1: FastAPI Endpoint
```python
@app.post("/query", response_model=QueryResponse)
async def query_document(query: Query):
    try:
        # 1. Validate input
        # 2. Process request
        # 3. Return response
        return QueryResponse(...)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Pattern 2: Pydantic Models
```python
class Query(BaseModel):
    question: str
    doc_id: str
    max_results: int = Field(default=5, ge=1, le=20)
    use_cache: bool = True
```

### Pattern 3: LRU Node Operations
```python
# Move to front (O(1))
def _move_to_front(self, node):
    if node == self.head:
        return
    self._remove_node(node)
    self._add_to_front(node)

# Evict LRU (O(1))
def _evict_lru(self):
    if self.tail:
        self._remove_node(self.tail)
        del self.cache[self.tail.key]
```

### Pattern 4: Async Parallel Execution
```python
# Execute agents in parallel
tasks = [agent.execute(input) for agent in agents]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

---

## 🎓 Questions to Ask Interviewer

### About the Role
- "What does a typical day look like for an AI Engineer at ELGO AI?"
- "What's the current AI stack you're using?"
- "What are the biggest technical challenges the team is facing?"

### About the Team
- "How big is the AI/ML team?"
- "How does the team collaborate (pair programming, code reviews)?"
- "What's the deployment process like?"

### About the Company
- "What's ELGO AI's approach to RAG systems in production?"
- "How do you balance research vs production work?"
- "What's the vision for AI at ELGO in the next 1-2 years?"

### About Next Steps
- "What are the next steps in the interview process?"
- "When can I expect to hear back?"
- "Is there anything else you'd like to know about my background?"

---

## ✅ Final Checks (Morning Of)

### Environment
- [ ] Python 3.9+ installed and working
- [ ] Virtual environment activated
- [ ] Libraries installed:
  ```bash
  pip install fastapi uvicorn langchain chromadb openai pydantic pytest numpy
  ```
- [ ] OpenAI API key in environment: `export OPENAI_API_KEY="sk-..."`

### Reference Materials Ready
- [ ] `section1_rag_solution.py` open
- [ ] `section2_cache_solution.py` open
- [ ] `section3_multiagent_solution.py` open
- [ ] Reference guides in browser tabs
- [ ] `GRADING_RUBRIC.md` open for self-assessment

### Physical Setup
- [ ] Comfortable chair
- [ ] Good lighting
- [ ] Water bottle
- [ ] Phone on silent
- [ ] Clean desk
- [ ] Charger nearby

---

## 🧠 Mindset & Stress Management

### Before Interview
- **Deep breathing**: 4-7-8 technique (inhale 4s, hold 7s, exhale 8s)
- **Positive visualization**: See yourself solving problems calmly
- **Affirmations**: "I've prepared well. I know this material. I can do this."

### During Interview
- **If stuck**: Take a breath, think out loud, break problem down
- **If making mistake**: "Actually, let me reconsider that..." (it's okay!)
- **If behind on time**: "I'll implement the core functionality and mention what I'd add with more time"

### Remember
- ✅ They want you to succeed (they're hiring!)
- ✅ It's a conversation, not an interrogation
- ✅ Asking questions shows engagement
- ✅ Explaining your thought process is more important than perfect code
- ✅ You've built production systems before - you know this!

---

## 📊 Self-Scoring (After Interview)

Rate yourself on each section (1-10):

**Section 1 - RAG System**
- Functionality: _____ /10
- Code Quality: _____ /10
- Communication: _____ /10

**Section 2 - Caching**
- Functionality: _____ /10
- Code Quality: _____ /10
- Communication: _____ /10

**Section 3 - Multi-Agent**
- Functionality: _____ /10
- Code Quality: _____ /10
- Communication: _____ /10

**Overall Performance**: _____ /100

**What went well**:
```
1. _____________________________________
2. _____________________________________
3. _____________________________________
```

**What to improve next time**:
```
1. _____________________________________
2. _____________________________________
3. _____________________________________
```

---

## 🚀 You've Got This!

**Remember**:
- You have 7 days of focused preparation
- You have production-quality code to learn from
- You understand the patterns
- You can explain your thinking clearly

**On interview day**:
- Stay calm
- Think out loud
- Start simple
- Ask questions
- Show your expertise

**You're ready!** 💪

Good luck! 🎯

---

**Emergency Contact**: If technical issues arise, have ELGO AI contact info ready.

**Post-Interview**: Send thank-you email within 24 hours mentioning specific topics discussed.
