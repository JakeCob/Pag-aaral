# Interview Time Management Guide

**Total Duration**: 3 hours (180 minutes)
**Format**: 3 sections with strict time limits

---

## ⏱️ Overall Time Allocation

```
Section 1: RAG System       ████████████████████ 90 min (50%)
Section 2: Cache System     ██████████           45 min (25%)
Section 3: Multi-Agent      ████████████         60 min (33%)
Buffer/Review               █                     5 min
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total                                           180 min (3 hours)
```

---

## 📊 Section 1: RAG System (90 minutes)

### Timeline

```
00:00 - 00:05  │ Read requirements & plan approach
00:05 - 00:35  │ Part A: Multi-format Processing (30 min)
00:35 - 01:00  │ Part B: Hybrid Search (25 min)
01:00 - 01:25  │ Part C: Conversation Memory (25 min)
01:25 - 01:30  │ Testing & Buffer (5 min)
```

### Part A: Multi-format Processing (30 min)

**0-10 min**: Core document loading
- [ ] TXT loader (3 min)
- [ ] PDF loader (5 min)
- [ ] JSON loader (2 min)

**10-20 min**: Chunking & embeddings
- [ ] Implement chunking with overlap (7 min)
- [ ] Create embeddings (3 min)

**20-30 min**: Vector storage & testing
- [ ] Store in ChromaDB (5 min)
- [ ] Test with sample docs (5 min)

### Part B: Hybrid Search (25 min)

**0-12 min**: BM25 implementation
- [ ] BM25 class structure (3 min)
- [ ] IDF calculation (4 min)
- [ ] Scoring function (5 min)

**12-20 min**: Semantic search
- [ ] Embedding-based search (5 min)
- [ ] ChromaDB query (3 min)

**20-25 min**: Hybrid combination
- [ ] Score normalization (3 min)
- [ ] Alpha weighting (2 min)

### Part C: Conversation Memory (25 min)

**0-10 min**: Memory structure
- [ ] Session storage (5 min)
- [ ] Add/retrieve turns (5 min)

**10-20 min**: Token limiting
- [ ] Token counting (5 min)
- [ ] Pruning logic (5 min)

**20-25 min**: Integration & testing
- [ ] Integrate with RAG (3 min)
- [ ] Test multi-turn (2 min)

### ⚠️ Section 1 Warnings

**If running behind at 45 min**:
- Skip conversation memory, focus on hybrid search
- Use simple list instead of token limiting

**If running behind at 60 min**:
- Move to Section 2
- Come back if time permits

---

## 📊 Section 2: Cache System (45 minutes)

### Timeline

```
01:30 - 01:35  │ Read requirements & plan approach (5 min)
01:35 - 02:00  │ Part A: LRU Implementation (25 min)
02:00 - 02:10  │ Part B: TTL Support (10 min)
02:10 - 02:15  │ Testing (5 min)
```

### Part A: LRU Implementation (25 min)

**Choose implementation approach** (1 min decision):
- **OrderedDict** (15 min) - Simpler, faster to code
- **Doubly-linked list** (25 min) - More impressive, shows DS knowledge

**If using OrderedDict** (recommended):
- 0-5 min: Basic structure
- 5-10 min: Get method
- 10-15 min: Put method with eviction
- 15-20 min: Testing
- 20-25 min: Buffer

**If using Doubly-linked list**:
- 0-8 min: Node class + list structure
- 8-15 min: Add/remove helpers
- 15-20 min: Get method
- 20-25 min: Put method

### Part B: TTL Support (10 min)

**0-5 min**: Expiry tracking
- [ ] Timestamp storage (2 min)
- [ ] Expiry check function (3 min)

**5-10 min**: Integration
- [ ] Update get() with expiry check (3 min)
- [ ] Update put() with TTL param (2 min)

### ⚠️ Section 2 Warnings

**If running behind at 25 min**:
- Finish basic LRU first
- TTL is bonus

**If running behind at 35 min**:
- Skip thread safety
- Move to Section 3

---

## 📊 Section 3: Multi-Agent System (60 minutes)

### Timeline

```
02:15 - 02:20  │ Read requirements & plan (5 min)
02:20 - 02:40  │ Part A: Base + 3 Agents (20 min)
02:40 - 02:55  │ Part B: Router (15 min)
02:55 - 03:15  │ Part C: Workflow/DAG (20 min)
03:15 - 03:20  │ Testing & Buffer (5 min)
```

### Part A: Base Agent + 3 Agents (20 min)

**0-5 min**: Base agent class
- [ ] Abstract base with ABC (3 min)
- [ ] can_handle() and execute() signatures (2 min)

**5-20 min**: Implement 3 agents (5 min each)
- [ ] SearchAgent (5 min)
- [ ] CalculatorAgent (5 min)
- [ ] WeatherAgent (5 min)

### Part B: Router (15 min)

**0-8 min**: Simple router
- [ ] Agent registration (3 min)
- [ ] Route to first matching agent (5 min)

**8-15 min**: Testing
- [ ] Test routing (5 min)
- [ ] Test no-match case (2 min)

### Part C: Workflow/DAG (20 min)

**0-10 min**: Task structure
- [ ] Task class with dependencies (3 min)
- [ ] Topological sort (7 min)

**10-18 min**: Workflow executor
- [ ] Execute in order (5 min)
- [ ] Pass data between tasks (3 min)

**18-20 min**: Quick test

### ⚠️ Section 3 Warnings

**If running behind at 45 min (02:45)**:
- Finish router, skip workflow
- Basic routing is more important

**If running behind at 55 min (02:55)**:
- Wrap up and move to review

---

## 🚨 Emergency Time Management

### If You're Running 15+ Minutes Behind

**Priority Order** (do in this sequence):
1. ✅ Section 1 Part A + B (multi-format + hybrid search)
2. ✅ Section 2 Part A (basic LRU)
3. ✅ Section 3 Part A + B (agents + router)
4. ⚠️ Section 1 Part C (conversation memory) - skip if needed
5. ⚠️ Section 2 Part B (TTL) - skip if needed
6. ⚠️ Section 3 Part C (DAG) - skip if needed

### If You're Stuck on Something (>10 min)

**Decision tree**:
```
Stuck > 10 min?
├─ Yes → Skip and move on
│         Mark with TODO comment
│         Come back if time permits
│
└─ No → Keep working
        Set 5-min checkpoint
```

---

## ⏰ Checkpoint System

Set alarms/timers at these points:

```
✓ 00:30 - Should be done with Part A of Section 1
✓ 00:55 - Should be done with Part B of Section 1
✓ 01:20 - Should be finishing Section 1
✓ 01:30 - MUST move to Section 2
✓ 02:00 - Should be done with basic LRU
✓ 02:15 - MUST move to Section 3
✓ 02:40 - Should have base agents done
✓ 03:00 - Should be wrapping up
✓ 03:15 - Final review period
```

---

## 💡 Time-Saving Tips

### Before You Start Coding

1. **Read requirements TWICE** (saves 20+ min of rework)
2. **Sketch data structures** (5 min planning saves 15 min coding)
3. **Identify must-haves vs nice-to-haves**

### While Coding

1. **Copy-paste your own code** - If you wrote a similar function, copy it
2. **Use simple imports** - `from collections import OrderedDict` not custom DS
3. **Skip docstrings initially** - Add them at the end if time
4. **Test as you go** - Small tests prevent big debugging later
5. **Use print statements** - Faster than debugger for quick checks

### Code Quality Shortcuts

**✅ Always do**:
- Type hints on function signatures
- Try-except on external calls
- Return structured responses

**⚠️ Skip if short on time**:
- Extensive comments
- Detailed docstrings
- Edge case handling (unless asked)
- Optimization (get it working first)

---

## 🎯 Per-Section Success Criteria

### Minimum to Pass Section 1
- [ ] Can load at least 2 file formats
- [ ] Semantic search working
- [ ] Basic retrieval returns results

### Minimum to Pass Section 2
- [ ] LRU cache get/put work
- [ ] Eviction happens when capacity exceeded
- [ ] O(1) operations (no loops in get/put)

### Minimum to Pass Section 3
- [ ] 2-3 agents implemented
- [ ] Router can direct queries
- [ ] Basic execution returns results

---

## 📝 Time Tracking Template

Use this during practice:

```
Section 1:
  Part A: Started _____ Ended _____ (Target: 30 min, Actual: ___ min)
  Part B: Started _____ Ended _____ (Target: 25 min, Actual: ___ min)
  Part C: Started _____ Ended _____ (Target: 25 min, Actual: ___ min)
  Section Total: ___ / 90 min

Section 2:
  Part A: Started _____ Ended _____ (Target: 25 min, Actual: ___ min)
  Part B: Started _____ Ended _____ (Target: 10 min, Actual: ___ min)
  Section Total: ___ / 45 min

Section 3:
  Part A: Started _____ Ended _____ (Target: 20 min, Actual: ___ min)
  Part B: Started _____ Ended _____ (Target: 15 min, Actual: ___ min)
  Part C: Started _____ Ended _____ (Target: 20 min, Actual: ___ min)
  Section Total: ___ / 60 min

Overall Total: ___ / 180 min
```

---

## 🔔 Final Reminders

**Before Interview**:
- [ ] Set up 3-hour timer
- [ ] Set up checkpoint alarms
- [ ] Have time allocation chart visible
- [ ] Close all distractions

**During Interview**:
- [ ] Check timer every 15 minutes
- [ ] Move on if stuck >10 min
- [ ] Prioritize breadth over depth
- [ ] Leave 5 min for final review

**Time Mindset**:
- **Good-enough code that works > Perfect code that's incomplete**
- **All sections attempted > One section perfect**
- **Working prototype > Optimized non-functional code**

---

**Remember**: The goal is to demonstrate you can build functional systems in reasonable time, not to write perfect production code. Time management is a key evaluation criteria!

Good luck! ⏱️💪
