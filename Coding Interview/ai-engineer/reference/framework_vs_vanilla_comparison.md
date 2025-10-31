# Framework vs Vanilla Python - Complete Comparison

**For companies that build custom systems instead of using LangChain**

---

## 🎯 Key Insight

> "We don't use LangChain or LangFlow - we build our own"

This tells you they want engineers who:
- ✅ Understand fundamentals, not just APIs
- ✅ Can build from primitives
- ✅ Make architectural decisions
- ✅ Know WHY things work

---

## 📊 Quick Comparison Table

| Concept | With LangChain | Vanilla Python | Interview Focus |
|---------|----------------|----------------|-----------------|
| **Document Loading** | `TextLoader()` | `open(file).read()` | ✅ Chunking strategies |
| **Chunking** | `CharacterTextSplitter()` | Custom loop with overlap | ✅ Why overlap matters |
| **Embeddings** | `OpenAIEmbeddings()` | `sentence_transformers` | ✅ Model selection |
| **Vector Store** | `Chroma.from_documents()` | `chromadb.Client()` | ✅ Similarity metrics |
| **LLM Calls** | `ChatOpenAI()` | `openai.ChatCompletion.create()` | ✅ Prompt engineering |
| **Chains** | `LCEL: prompt \| llm \| parser` | Custom functions | ✅ Orchestration logic |
| **Agents** | `initialize_agent()` | Custom base class + routing | ✅ Agent patterns |
| **Memory** | `ConversationBufferMemory()` | List/dict + pruning | ✅ State management |

---

## 🔍 Detailed Comparisons

### 1. RAG System

#### With LangChain
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# Split
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Create chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever()
)

# Query
answer = qa.run("What is this about?")
```

**Pros**:
- Fast to prototype
- Lots of built-in features
- Good for standard use cases

**Cons**:
- Black box - don't know what's happening
- Hard to customize
- Framework lock-in

#### Vanilla Python
```python
import openai
import chromadb
from sentence_transformers import SentenceTransformer

# Load and chunk manually
with open("document.txt") as f:
    text = f.read()

chunks = []
for i in range(0, len(text), 450):  # 500-50 for overlap
    chunks.append(text[i:i+500])

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Store in ChromaDB
client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(
    embeddings=embeddings.tolist(),
    documents=chunks,
    ids=[f"doc_{i}" for i in range(len(chunks))]
)

# Retrieve
query_embedding = model.encode(["What is this about?"])[0]
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

# Generate answer
context = "\n".join(results["documents"][0])
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Answer based on context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: What is this about?"}
    ]
)
answer = response.choices[0].message.content
```

**Pros**:
- Full control over every step
- Easy to customize
- Understand exactly what's happening
- No framework dependencies

**Cons**:
- More code to write
- Need to handle edge cases yourself
- Reinvent some wheels

**Interview Advantage**: ✅ Shows deep understanding

---

### 2. BM25 Search

#### With LangChain
```python
from langchain.retrievers import BM25Retriever

retriever = BM25Retriever.from_documents(documents)
results = retriever.get_relevant_documents("query")
```

**What you DON'T learn**: How BM25 actually works!

#### Vanilla Python
```python
from collections import Counter
import math

class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents

        # Calculate document lengths
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avgdl = sum(self.doc_lengths) / len(documents)

        # Calculate IDF
        N = len(documents)
        df = Counter()
        for doc in documents:
            unique_terms = set(doc.lower().split())
            df.update(unique_terms)

        self.idf = {}
        for term, doc_freq in df.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def score(self, query, doc_idx):
        score = 0.0
        doc = self.documents[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = Counter(doc)

        for term in query.lower().split():
            if term not in self.idf:
                continue

            tf = term_freqs[term]
            idf = self.idf[term]

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)

        return score
```

**Interview Advantage**: ✅ Can explain formula, tune parameters, debug issues

---

### 3. Multi-Agent System

#### With LangChain
```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [
    Tool(name="Search", func=search_function, description="Search docs"),
    Tool(name="Calculator", func=calc_function, description="Do math")
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("Search for Python and calculate 10 + 20")
```

**What you DON'T learn**: How agents decide which tool to use!

#### Vanilla Python
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Decide if this agent can handle query"""
        pass

    @abstractmethod
    async def execute(self, query: str) -> dict:
        """Execute and return result"""
        pass

class SearchAgent(BaseAgent):
    def can_handle(self, query: str) -> bool:
        return any(kw in query.lower() for kw in ["search", "find"])

    async def execute(self, query: str) -> dict:
        # Custom search logic
        return {"result": "...", "confidence": 0.9}

class Router:
    def __init__(self):
        self.agents = []

    def register(self, agent):
        self.agents.append(agent)

    async def route(self, query: str):
        for agent in self.agents:
            if agent.can_handle(query):
                return await agent.execute(query)
        return {"error": "No agent available"}
```

**Interview Advantage**: ✅ Can design custom routing, add fallbacks, explain decisions

---

## 🎓 What to Emphasize in Interview

### ✅ DO Talk About:

1. **Fundamentals**
   - "I'd use sentence-transformers for embeddings because..."
   - "BM25 works by combining TF-IDF with length normalization..."
   - "Cosine similarity is better than L2 for normalized vectors because..."

2. **Trade-offs**
   - "I chose chunk size 500 with 50 overlap to balance context and granularity"
   - "ChromaDB is good for prototypes, but Pinecone scales better"
   - "Alpha=0.6 gives 60% weight to semantic, 40% to BM25 - tunable per use case"

3. **System Design**
   - "I'd cache embeddings to avoid recomputing"
   - "Circuit breaker prevents cascading failures"
   - "Two-stage retrieval: fast BM25, then precise re-ranking"

### ❌ DON'T Say:

1. ~~"I'd use LangChain's RetrievalQA chain"~~ (too high-level)
2. ~~"LCEL makes it easy to chain components"~~ (they don't use LCEL)
3. ~~"LangSmith helps debug chains"~~ (they build custom tools)

---

## 💡 Interview Strategy

### When Asked: "How would you build a RAG system?"

**Bad Answer**:
> "I'd use LangChain's document loaders to load files, then CharacterTextSplitter to chunk them, then create a Chroma vectorstore with OpenAI embeddings, and finally use RetrievalQA chain."

**Good Answer**:
> "I'd start by loading the documents and chunking them with overlap - maybe 500 characters per chunk with 50 character overlap to maintain context. Then I'd generate embeddings using sentence-transformers (all-MiniLM-L6-v2 is fast and good quality) and store them in ChromaDB with cosine similarity. For retrieval, I'd implement hybrid search combining BM25 for keyword matching and semantic search for meaning. Finally, I'd use the OpenAI API directly to generate answers from the retrieved context."

**Why it's better**:
- Shows understanding of each component
- Mentions specific design choices
- Explains trade-offs
- Can implement from scratch

---

## 📝 Study Approach Changes

### Original Plan
- ~~Day 1-2: LangChain RAG chains~~
- ~~Day 3: LCEL syntax~~
- ~~Day 4: LangChain agents~~

### Updated Plan
- ✅ Day 1: RAG fundamentals (chunking, embeddings, retrieval)
- ✅ Day 2: BM25 implementation, hybrid search
- ✅ Day 3: Agent patterns, routing logic
- ✅ Day 4: System architecture, caching, fault tolerance
- ✅ Day 5: Mock interview with vanilla implementations

---

## 🔧 Practice Files

### Use These Files:
1. `01_beginner_simple_rag_vanilla.py` ← **Start here**
2. `02_intermediate_hybrid_search_vanilla.py` ← **BM25 from scratch**
3. `01_beginner_simple_agents_vanilla.py` ← **Agent patterns**

### Ignore These (for now):
- ~~Any file mentioning LangChain~~
- ~~LCEL examples~~
- ~~LangSmith debugging~~

You can review them later to understand what frameworks abstract away!

---

## 🎯 Bottom Line

**For This Interview**:
- Build everything from scratch ✅
- Use raw APIs (OpenAI, ChromaDB) ✅
- Explain your decisions ✅
- Know the algorithms ✅
- ~~Use LangChain~~ ❌

**Remember**: They want to see you can build systems, not just use frameworks!

---

## 📚 Quick Reference

**Raw APIs to Know**:
```python
# OpenAI
import openai
response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[...])

# ChromaDB
import chromadb
client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(embeddings=..., documents=..., ids=...)
results = collection.query(query_embeddings=[...], n_results=5)

# Sentence Transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
```

**Algorithms to Implement**:
- Chunking with overlap
- BM25 scoring
- Cosine similarity
- Min-max normalization
- Topological sort (for DAGs)
- LRU cache

Good luck! You've got this! 🚀
