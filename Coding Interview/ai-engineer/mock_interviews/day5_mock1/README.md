# Mock Interview #1 - Day 5

**Duration**: 3 hours
**Difficulty**: Simulates actual interview
**Goal**: Identify weak areas and practice time management

---

## 📋 Format (Matches Real Interview)

### Section 1: RAG System (90 minutes)

**Part A (30 min)**: Multi-format Document Processing
- Load documents from: TXT, PDF, JSON
- Implement chunking with overlap
- Create embeddings and store in ChromaDB

**Part B (30 min)**: Hybrid Search Implementation
- Implement BM25 retriever
- Implement semantic search
- Combine with alpha weighting (alpha=0.6)

**Part C (25 min)**: Conversation Memory
- Session-based conversation history
- Token limiting (max 2000 tokens)
- Context integration in queries

**Buffer**: 5 min for testing and debugging

---

### Section 2: Cache System (45 minutes)

**Part A (25 min)**: LRU Cache Implementation
- Doubly-linked list + HashMap
- O(1) get and put operations
- Capacity enforcement

**Part B (15 min)**: TTL Support
- Add expiration times
- Lazy deletion on access
- Thread safety with locks

**Testing**: 5 min

---

### Section 3: Multi-Agent System (60 minutes)

**Part A (20 min)**: Agent Base Class + 3 Agents
- Abstract BaseAgent
- SearchAgent, CalculatorAgent, WeatherAgent
- `can_handle()` and `execute()` methods

**Part B (15 min)**: Query Router
- Route queries to appropriate agent
- Handle queries no agent can process

**Part C (20 min)**: Simple Workflow
- Define task dependencies
- Topological sort
- Sequential execution

**Buffer**: 5 min

---

## 🎯 Success Criteria

### Minimum Viable (Pass)
- [ ] All sections attempted
- [ ] Basic functionality works
- [ ] Code runs without crashes
- [ ] Reasonable time management

### Good Performance
- [ ] All core features implemented
- [ ] Proper error handling
- [ ] Clean, readable code
- [ ] Completed within time limits

### Excellent Performance
- [ ] All features + extensions
- [ ] Production-quality code
- [ ] Comprehensive error handling
- [ ] Finished with time to spare
- [ ] Can explain all design choices

---

## 📝 Preparation Checklist

### Before Starting (5 min)
- [ ] Set up 3-hour timer
- [ ] Have water/snacks ready
- [ ] Close distractions (Slack, email, etc.)
- [ ] Open necessary documentation
- [ ] Create workspace folder

### During Mock (3 hours)
- [ ] Read ALL requirements before coding
- [ ] Time-box each section strictly
- [ ] Write code comments as you go
- [ ] Test each component before moving on
- [ ] Take 2-min breaks between sections

### After Mock (30 min)
- [ ] Note what went well
- [ ] Identify time management issues
- [ ] List topics to review
- [ ] Compare with reference solutions
- [ ] Update study plan

---

## 🔍 Self-Assessment

After completing, rate yourself (1-5) on:

### Technical Implementation
- [ ] RAG system completeness (1-5)
- [ ] Cache correctness (1-5)
- [ ] Multi-agent functionality (1-5)
- [ ] Code quality (1-5)
- [ ] Error handling (1-5)

### Soft Skills
- [ ] Time management (1-5)
- [ ] Code organization (1-5)
- [ ] Communication (explaining approach) (1-5)
- [ ] Problem-solving approach (1-5)

### Areas for Improvement
1. _____________________________________
2. _____________________________________
3. _____________________________________

### What to Focus on Before Next Mock
1. _____________________________________
2. _____________________________________
3. _____________________________________

---

## 💡 Tips

1. **Read requirements twice** - Don't start coding immediately
2. **Start simple** - Get basic version working first
3. **Time management** - If stuck >10 min, move on and come back
4. **Test frequently** - Don't wait until the end
5. **Comment as you go** - Helps with explaining later
6. **Stay calm** - It's practice, mistakes are learning opportunities

---

## 📚 Reference Materials

During the mock, you CAN refer to:
- Your solutions/ folder
- Reference guides
- Official documentation (Python, FastAPI, LangChain)

You should NOT:
- Copy-paste large code blocks
- Look up specific algorithms during the mock
- Seek external help

---

## 🎬 Ready to Start?

1. Set your timer for 3 hours
2. Create a new folder: `mock1_attempt/`
3. Open `INTERVIEW_PROMPT.md` from solutions/ folder
4. Begin Section 1!

**Good luck!** 🚀

---

**Post-Mock**: Review your code against `section1/2/3_solution.py` and note gaps.
