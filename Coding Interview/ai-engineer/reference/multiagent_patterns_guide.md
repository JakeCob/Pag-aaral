# Multi-Agent Patterns Guide
**Reference Guide for ELGO AI Interview Preparation**

Based on: `section3_multiagent_solution.py`

---

## Table of Contents
1. [Overview](#overview)
2. [Agent Design Pattern](#agent-design-pattern)
3. [Query Routing](#query-routing)
4. [DAG Workflow Orchestration](#dag-workflow-orchestration)
5. [Error Handling & Resilience](#error-handling--resilience)
6. [Circuit Breaker Pattern](#circuit-breaker-pattern)
7. [Production Patterns](#production-patterns)
8. [Code Templates](#code-templates)

---

## Overview

### What is Multi-Agent Architecture?

Instead of one monolithic AI system, use **specialized agents** for different tasks:

```
User Query → Router → [RAG Agent, SQL Agent, Code Agent] → Orchestrator → Aggregated Response
```

**Benefits**:
- ✅ Specialization (each agent optimized for its task)
- ✅ Parallel execution (faster responses)
- ✅ Fault tolerance (one agent failing doesn't break everything)
- ✅ Scalability (add new agents easily)

### Architecture Components

```
┌──────────────────────────────────────────────────────┐
│                     User Query                        │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────────────────┐
│              Query Router                             │
│  - Analyzes query                                     │
│  - Scores agents by confidence                        │
│  - Selects best agent(s)                              │
└───────────────────┬───────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────────────────┐
│         Workflow Orchestrator                         │
│  - Builds dependency graph (DAG)                      │
│  - Executes agents in parallel where possible         │
│  - Handles retries and timeouts                       │
└───────────────────┬───────────────────────────────────┘
                    ↓
        ┌──────────┬──────────┬──────────┐
        ↓          ↓          ↓          ↓
    RAG Agent  SQL Agent  Code Agent  Analysis Agent
        │          │          │          │
        └──────────┴──────────┴──────────┘
                    ↓
┌───────────────────────────────────────────────────────┐
│           Result Aggregator                           │
│  - Combines outputs by confidence                     │
│  - Returns best/merged result                         │
└───────────────────────────────────────────────────────┘
```

---

## Agent Design Pattern

### Base Agent Class

**Abstract Interface** (`section3_multiagent_solution.py:80-184`)

```python
class Agent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str, agent_type: AgentType):
        self.name = name
        self.agent_type = agent_type
        self.execution_count = 0
        self.total_execution_time = 0.0

    @abstractmethod
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """Execute agent logic - MUST IMPLEMENT"""
        pass

    @abstractmethod
    def can_handle(self, query: str) -> float:
        """Return confidence score (0.0-1.0) - MUST IMPLEMENT"""
        pass

    async def execute_with_timing(self, input_data: AgentInput) -> AgentOutput:
        """Wrapper with timing and error handling"""
        start_time = time.time()

        try:
            output = await self.execute(input_data)
            execution_time_ms = (time.time() - start_time) * 1000

            output.execution_time_ms = execution_time_ms
            self.execution_count += 1
            self.total_execution_time += execution_time_ms

            return output

        except Exception as e:
            return AgentOutput(
                result=None,
                confidence=0.0,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
```

**Key Patterns**:
1. **ABC (Abstract Base Class)** - Enforces interface
2. **Standardized I/O** - AgentInput/AgentOutput Pydantic models
3. **Built-in Timing** - Automatic performance tracking
4. **Error Handling** - Never throws, returns error in output

### Specialized Agents

#### RAG Agent (`section3_multiagent_solution.py:190-269`)

**Use Case**: Document Q&A, knowledge retrieval

```python
class RAGAgent(Agent):
    def __init__(self):
        super().__init__("RAG Agent", AgentType.RAG)
        self.keywords = [
            "document", "explain", "what is", "tell me about",
            "information", "details", "describe"
        ]

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        # 1. Embed query
        # 2. Retrieve relevant chunks
        # 3. Generate answer with LLM
        return AgentOutput(
            result={"answer": "...", "sources": [...]},
            confidence=0.85
        )

    def can_handle(self, query: str) -> float:
        query_lower = query.lower()

        # Keyword matching
        keyword_score = sum(
            1 for kw in self.keywords if kw in query_lower
        ) / len(self.keywords)

        # Pattern matching
        patterns = [r"what (is|are)", r"explain", r"describe"]
        pattern_score = sum(
            1 for p in patterns if re.search(p, query_lower)
        ) / len(patterns)

        # Combine (max ensures OR logic)
        return min(max(keyword_score, pattern_score) * 2, 1.0)
```

**Routing Strategy**: Keyword + regex pattern matching

#### SQL Agent (`section3_multiagent_solution.py:271-356`)

**Use Case**: Database queries, analytics

```python
class SQLAgent(Agent):
    def __init__(self, db_connection_string: Optional[str] = None):
        super().__init__("SQL Agent", AgentType.SQL)
        self.db_connection = db_connection_string
        self.keywords = [
            "database", "query", "select", "table",
            "count", "sum", "average"
        ]

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        # 1. Generate SQL with LLM
        # 2. Validate SQL (prevent injection)
        # 3. Execute against database
        # 4. Format results
        return AgentOutput(
            result={"sql_query": "...", "results": [...]},
            confidence=0.90
        )

    def can_handle(self, query: str) -> float:
        # Check for database-related keywords and patterns
        sql_patterns = [
            r"(get|fetch|retrieve).*(data|records)",
            r"how many",
            r"from (the )?(database|table)"
        ]
        # Similar scoring logic
```

**Key Insight**: Higher confidence (0.90) because SQL queries are more deterministic

#### Code Agent (`section3_multiagent_solution.py:358-451`)

**Use Case**: Code generation, implementation help

```python
class CodeAgent(Agent):
    def can_handle(self, query: str) -> float:
        code_patterns = [
            r"(write|create|implement).*(code|function)",
            r"how (do|can) (i|we) (code|implement)",
            r"show me (the )?code"
        ]
        # Pattern matching for code requests
```

### Standard Models

**Input** (`section3_multiagent_solution.py:60-64`):
```python
class AgentInput(BaseModel):
    query: str                          # User question
    context: Dict[str, Any] = {}        # Results from previous agents
    parameters: Dict[str, Any] = {}     # Additional params
```

**Output** (`section3_multiagent_solution.py:67-73`):
```python
class AgentOutput(BaseModel):
    result: Any                         # Agent result (dict, list, str)
    confidence: float                   # 0.0-1.0 confidence score
    metadata: Dict[str, Any] = {}       # Extra info
    error: Optional[str] = None         # Error message if failed
    execution_time_ms: float = 0.0      # Performance tracking
```

---

## Query Routing

### Router Implementation

**QueryRouter** (`section3_multiagent_solution.py:614-687`)

```python
class QueryRouter:
    """Route queries to appropriate agents"""

    def __init__(
        self,
        agents: List[Agent],
        confidence_threshold: float = 0.3,  # Minimum to consider
        max_agents: int = 3                 # Limit parallel execution
    ):
        self.agents = agents
        self.confidence_threshold = confidence_threshold
        self.max_agents = max_agents

    def route(self, query: str) -> List[Tuple[Agent, float]]:
        """Return top agents sorted by confidence"""
        scores = []

        # Score all agents
        for agent in self.agents:
            confidence = agent.can_handle(query)
            if confidence >= self.confidence_threshold:
                scores.append((agent, confidence))

        # Sort by confidence descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top max_agents
        return scores[:self.max_agents]
```

### Routing Strategies

**1. Keyword Matching** (Simple but effective)
```python
def can_handle(self, query: str) -> float:
    query_lower = query.lower()
    keyword_score = sum(
        1 for keyword in self.keywords
        if keyword in query_lower
    ) / len(self.keywords)
    return keyword_score
```

**2. Regex Pattern Matching** (More precise)
```python
patterns = [
    r"what (is|are|does)",
    r"explain",
    r"tell me about"
]
pattern_score = sum(
    1 for pattern in patterns
    if re.search(pattern, query_lower)
) / len(patterns)
```

**3. Embedding-Based** (Most sophisticated, not in solution but recommended)
```python
# Embed query and agent descriptions
query_embedding = embed(query)
agent_embeddings = [embed(agent.description) for agent in agents]

# Cosine similarity
scores = [cosine_similarity(query_embedding, agent_emb)
          for agent_emb in agent_embeddings]
```

### Confidence Tuning

**Confidence Threshold**:
- Too low (e.g., 0.1): Too many agents triggered, wasted compute
- Too high (e.g., 0.7): Might miss relevant agents
- Sweet spot: **0.2-0.4**

**Example**:
```python
Query: "Explain RAG from our documents"

Scores:
- RAG Agent: 0.85 ✅ (keywords: "explain", "documents")
- SQL Agent: 0.10 ❌ (below threshold)
- Code Agent: 0.05 ❌ (below threshold)

Selected: [RAG Agent]
```

---

## DAG Workflow Orchestration

### Why DAG (Directed Acyclic Graph)?

**Without DAG** (Sequential):
```
RAG → SQL → Code → Analysis
Total Time: 0.1s + 0.15s + 0.2s + 0.25s = 0.7s
```

**With DAG** (Parallel):
```
     ┌→ RAG (0.1s) →┐
     ├→ SQL (0.15s) ┼→ Analysis (0.25s)
     └→ Code (0.2s) ┘
Total Time: max(0.1, 0.15, 0.2) + 0.25 = 0.45s  (36% faster!)
```

### Dependency Graph

**WorkflowStep** (`section3_multiagent_solution.py:693-703`)
```python
@dataclass
class WorkflowStep:
    step_id: str
    agent: Agent
    input_data: AgentInput
    depends_on: List[str] = field(default_factory=list)  # Dependencies
    status: AgentStatus = AgentStatus.PENDING
    output: Optional[AgentOutput] = None
    retry_count: int = 0
```

**Building Graph** (`section3_multiagent_solution.py:924-946`)
```python
def build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
    """Build adjacency list: {step_id: [dependent_step_ids]}"""
    graph = defaultdict(list)

    for step in steps:
        if not graph[step.step_id]:
            graph[step.step_id] = []

        # Add edges from dependencies to this step
        for dep_id in step.depends_on:
            graph[dep_id].append(step.step_id)

    return dict(graph)
```

**Example**:
```python
steps = [
    WorkflowStep(step_id="rag", agent=rag_agent, depends_on=[]),
    WorkflowStep(step_id="sql", agent=sql_agent, depends_on=[]),
    WorkflowStep(step_id="analysis", agent=analysis_agent, depends_on=["rag", "sql"])
]

graph = {
    "rag": ["analysis"],      # RAG must finish before analysis
    "sql": ["analysis"],      # SQL must finish before analysis
    "analysis": []            # Analysis has no dependents
}
```

### Topological Sort (Kahn's Algorithm)

**Purpose**: Determine execution order while respecting dependencies

**Implementation** (`section3_multiagent_solution.py:948-987`)
```python
def topological_sort(self, graph: Dict[str, List[str]]) -> List[List[str]]:
    """Returns list of levels (steps at same level run in parallel)"""

    # 1. Calculate in-degrees (how many dependencies each node has)
    in_degree = defaultdict(int)
    all_nodes = set(graph.keys())

    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
            all_nodes.add(neighbor)

    # 2. Find nodes with no dependencies (in-degree = 0)
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    levels = []

    # 3. Process level by level
    while queue:
        # Current level (can run in parallel)
        current_level = list(queue)
        levels.append(current_level)

        # Prepare next level
        next_queue = deque()
        for node in current_level:
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_queue.append(neighbor)

        queue = next_queue

    return levels
```

**Example Output**:
```python
levels = [
    ["rag", "sql"],           # Level 0: Run in parallel
    ["analysis"]              # Level 1: Wait for level 0, then run
]
```

### Parallel Execution

**execute_parallel** (`section3_multiagent_solution.py:989-1017`)
```python
async def execute_parallel(
    self,
    steps: List[WorkflowStep],
    context: Dict[str, Any]
) -> List[AgentOutput]:
    """Execute multiple steps in parallel using asyncio"""

    # Create tasks for all steps
    tasks = [self.execute_step(step, context) for step in steps]

    # Execute in parallel (asyncio.gather)
    outputs = await asyncio.gather(*tasks, return_exceptions=True)

    # Update shared context with results
    for step, output in zip(steps, outputs):
        if not isinstance(output, Exception) and output.result:
            context[step.step_id] = output.result

    return [o for o in outputs if not isinstance(o, Exception)]
```

**Key Pattern**: `asyncio.gather` with `return_exceptions=True` prevents one failure from canceling others

---

## Error Handling & Resilience

### Retry Logic with Exponential Backoff

**execute_step** (`section3_multiagent_solution.py:853-923`)
```python
async def execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> AgentOutput:
    """Execute with retries"""

    for attempt in range(self.max_retries + 1):
        try:
            # Add context from previous steps
            step.input_data.context = context.copy()

            # Execute with timeout
            output = await asyncio.wait_for(
                step.agent.execute_with_timing(step.input_data),
                timeout=self.timeout
            )

            if output.error:
                raise Exception(output.error)

            step.status = AgentStatus.COMPLETED
            return output

        except asyncio.TimeoutError:
            step.status = AgentStatus.RETRYING
            step.retry_count += 1

            if attempt < self.max_retries:
                # Exponential backoff: 1s, 2s, 4s
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
            else:
                step.status = AgentStatus.FAILED
                return AgentOutput(
                    result=None,
                    confidence=0.0,
                    error="Timeout after retries"
                )

        except Exception as e:
            # Similar retry logic for other exceptions
```

**Exponential Backoff**:
```
Attempt 0: Delay = 1s × 2^0 = 1s
Attempt 1: Delay = 1s × 2^1 = 2s
Attempt 2: Delay = 1s × 2^2 = 4s
```

**Why Exponential?**
- Gives failing service time to recover
- Reduces load during incidents
- Standard industry practice

### Timeout Handling

**asyncio.wait_for**:
```python
output = await asyncio.wait_for(
    step.agent.execute_with_timing(input_data),
    timeout=30  # seconds
)
```

**Best Practices**:
- RAG queries: 5-10s timeout
- SQL queries: 10-30s timeout
- Code generation: 30-60s timeout

### Result Aggregation

**aggregate_results** (`section3_multiagent_solution.py:1019-1072`)
```python
def aggregate_results(self, outputs: List[AgentOutput]) -> Dict[str, Any]:
    """Combine results from multiple agents"""

    # Filter successful outputs
    successful = [o for o in outputs if o.result and not o.error]

    if not successful:
        return {
            "message": "All agents failed",
            "errors": [o.error for o in outputs if o.error]
        }

    # Sort by confidence
    successful.sort(key=lambda x: x.confidence, reverse=True)

    # Single result: return directly
    if len(successful) == 1:
        return successful[0].result

    # Multiple results: primary + additional
    return {
        "primary_result": successful[0].result,
        "primary_confidence": successful[0].confidence,
        "additional_results": [
            {
                "result": o.result,
                "confidence": o.confidence,
                "agent": o.metadata.get("agent_type")
            }
            for o in successful[1:]
        ],
        "total_agents": len(successful)
    }
```

**Aggregation Strategies**:
1. **Best Only**: Return highest confidence result
2. **Weighted Average**: Combine results weighted by confidence
3. **Ensemble**: Let LLM synthesize multiple results
4. **Voting**: Majority vote (for classification)

---

## Circuit Breaker Pattern

### Why Circuit Breaker?

**Problem**: Failing agent causes cascading failures
```
Agent fails → Retry 3x → Timeout → User waits 90s → Bad UX
```

**Solution**: Circuit breaker temporarily disables failing agent
```
Agent fails 5x → Circuit OPEN → Skip agent → Fast failure → Better UX
```

### States

```
        success count < threshold
       ┌─────────────────┐
       │                 ↓
   ┌──────┐  timeout  ┌──────────┐  success count >= threshold  ┌────────┐
   │ OPEN │─────────→ │ HALF-OPEN│───────────────────────────→ │ CLOSED │
   └──────┘           └──────────┘                              └────────┘
       ↑                    │                                        │
       └────────────────────┴────────────────────────────────────────┘
                    failure count >= threshold
```

### Implementation

**CircuitBreaker** (`section3_multiagent_solution.py:1174-1247`)
```python
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,       # Failures before opening
        timeout_seconds: int = 60,        # Time before trying again
        success_threshold: int = 2        # Successes to close
    ):
        self.state = defaultdict(lambda: "closed")
        self.failures = defaultdict(int)
        self.successes = defaultdict(int)
        self.opened_at = {}

    def call(self, agent_name: str) -> bool:
        """Check if agent can be called"""
        state = self.state[agent_name]

        if state == "closed":
            return True  # Normal operation

        if state == "open":
            # Check if timeout expired
            if time.time() - self.opened_at[agent_name] > self.timeout:
                self.state[agent_name] = "half-open"
                return True  # Try again
            return False  # Still open, skip

        if state == "half-open":
            return True  # Testing if agent recovered

    def record_success(self, agent_name: str):
        """Record successful call"""
        self.failures[agent_name] = 0

        if self.state[agent_name] == "half-open":
            self.successes[agent_name] += 1
            if self.successes[agent_name] >= self.success_threshold:
                self.state[agent_name] = "closed"  # Fully recovered

    def record_failure(self, agent_name: str):
        """Record failed call"""
        self.failures[agent_name] += 1

        if self.failures[agent_name] >= self.failure_threshold:
            self.state[agent_name] = "open"
            self.opened_at[agent_name] = time.time()
```

**Usage**:
```python
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60)

async def execute_with_circuit_breaker(agent, input_data):
    if not circuit_breaker.call(agent.name):
        return AgentOutput(result=None, confidence=0.0, error="Circuit open")

    try:
        output = await agent.execute(input_data)
        circuit_breaker.record_success(agent.name)
        return output
    except Exception as e:
        circuit_breaker.record_failure(agent.name)
        raise
```

---

## Production Patterns

### Execution Tracing

**WorkflowExecutionTracer** (`section3_multiagent_solution.py:1078-1168`)
```python
class WorkflowExecutionTracer:
    """Trace execution for debugging and monitoring"""

    def start_trace(self, workflow_id: str, query: str):
        self.traces[workflow_id] = {
            "workflow_id": workflow_id,
            "query": query,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "status": "running"
        }

    def log_step(self, workflow_id, step_id, agent_name, status, duration_ms, error=None):
        self.traces[workflow_id]["steps"].append({
            "step_id": step_id,
            "agent": agent_name,
            "status": status,
            "duration_ms": duration_ms,
            "error": error
        })

    def end_trace(self, workflow_id, total_duration_ms, success):
        self.traces[workflow_id]["status"] = "completed" if success else "failed"
        self.traces[workflow_id]["total_duration_ms"] = total_duration_ms
```

**Benefits**:
- Debug failures
- Identify bottlenecks
- Monitor agent performance
- Audit trail

### Agent Metrics

**Tracking** (`section3_multiagent_solution.py:169-183`):
```python
def get_metrics(self) -> Dict[str, Any]:
    """Get agent performance metrics"""
    avg_time = (
        self.total_execution_time / self.execution_count
        if self.execution_count > 0
        else 0.0
    )

    return {
        "name": self.name,
        "type": self.agent_type.value,
        "execution_count": self.execution_count,
        "total_execution_time_ms": self.total_execution_time,
        "avg_execution_time_ms": avg_time
    }
```

**Monitoring Dashboard**:
```python
# After running workflows
for agent in agents:
    metrics = agent.get_metrics()
    print(f"{metrics['name']}:")
    print(f"  Executions: {metrics['execution_count']}")
    print(f"  Avg Time: {metrics['avg_execution_time_ms']:.2f}ms")
```

---

## Code Templates

### Template 1: Create Custom Agent

```python
from abc import ABC, abstractmethod
import asyncio

class CustomAgent(Agent):
    """Your custom agent"""

    def __init__(self):
        super().__init__("Custom Agent", AgentType.CUSTOM)
        self.keywords = ["custom", "special"]

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        # Your agent logic here
        await asyncio.sleep(0.1)  # Simulate work

        return AgentOutput(
            result={"data": "..."},
            confidence=0.8,
            metadata={"agent_type": "custom"}
        )

    def can_handle(self, query: str) -> float:
        query_lower = query.lower()

        # Implement your routing logic
        keyword_score = sum(
            1 for kw in self.keywords if kw in query_lower
        ) / len(self.keywords)

        return min(keyword_score * 2, 1.0)
```

### Template 2: Simple Multi-Agent System

```python
async def simple_multi_agent(query: str):
    # 1. Initialize agents
    agents = [RAGAgent(), SQLAgent(), CodeAgent()]

    # 2. Create router
    router = QueryRouter(agents, confidence_threshold=0.3)

    # 3. Route query
    selected_agents = router.route(query)

    # 4. Execute in parallel
    tasks = [
        agent.execute_with_timing(AgentInput(query=query))
        for agent, _ in selected_agents
    ]
    outputs = await asyncio.gather(*tasks)

    # 5. Aggregate results
    best_output = max(outputs, key=lambda o: o.confidence)
    return best_output.result
```

### Template 3: Workflow with Dependencies

```python
async def workflow_with_deps(query: str):
    # Define steps with dependencies
    steps = [
        WorkflowStep(
            step_id="retrieve",
            agent=rag_agent,
            input_data=AgentInput(query=query),
            depends_on=[]  # No dependencies
        ),
        WorkflowStep(
            step_id="analyze",
            agent=analysis_agent,
            input_data=AgentInput(query="Analyze retrieved data"),
            depends_on=["retrieve"]  # Wait for retrieve
        )
    ]

    # Execute workflow
    orchestrator = WorkflowOrchestrator()
    graph = orchestrator.build_dependency_graph(steps)
    context = {}
    await orchestrator.execute_dag(steps, graph, context)

    # Return final result
    return steps[-1].output
```

---

## Interview Tips

### Common Questions

**Q: How do you handle agent failures?**
A: 1) Retry with exponential backoff, 2) Circuit breaker to prevent cascading failures, 3) Graceful degradation (return partial results)

**Q: How do you decide which agent to use?**
A: Query router scores agents using: 1) Keyword matching, 2) Regex patterns, 3) Optionally embedding similarity. Select agents above confidence threshold.

**Q: How do you make multi-agent systems fast?**
A: 1) Parallel execution where possible (DAG), 2) Async/await for I/O-bound tasks, 3) Timeouts to prevent hanging, 4) Caching frequent queries

**Q: How would you add a new agent type?**
A: 1) Create class inheriting Agent, 2) Implement execute() and can_handle(), 3) Add to agents list, 4) Router automatically includes it

**Q: What's the difference between sequential and parallel execution?**
A: Sequential runs one after another (total time = sum). Parallel runs simultaneously (total time = max). Use DAG to identify which steps can run in parallel.

---

**Quick Reference: section3_multiagent_solution.py**

| Component | Lines | Purpose |
|-----------|-------|---------|
| `Agent` (base class) | 80-184 | Abstract interface for all agents |
| `RAGAgent` | 190-269 | Document Q&A |
| `SQLAgent` | 271-356 | Database queries |
| `CodeAgent` | 358-451 | Code generation |
| `QueryRouter` | 614-687 | Route queries to agents |
| `WorkflowOrchestrator` | 705-1072 | DAG execution + retry |
| `CircuitBreaker` | 1174-1247 | Prevent cascading failures |
| `topological_sort` | 948-987 | Kahn's algorithm for DAG |

---

**Ready to Practice?**
Check `/tracks/02_workflow_multiagent/` for progressive challenges:
- Beginner: Single agent with can_handle()
- Intermediate: Multi-agent routing
- Advanced: DAG workflow with parallel execution

Good luck! 🚀
