"""
ELGO AI - Multi-Agent Workflow System Solution
Section 3: Complete Implementation

Features:
- Multiple specialized agents (RAG, SQL, Code, Analysis, Web Search)
- Intelligent query routing
- DAG-based workflow orchestration
- Parallel execution where possible
- Error handling with retry logic
- Circuit breaker pattern
- Comprehensive logging and tracing

Author: Reference Solution
"""

import asyncio
import logging
import time
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
from pydantic import BaseModel
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class AgentType(Enum):
    """Supported agent types"""
    RAG = "rag"
    SQL = "sql"
    CODE = "code"
    ANALYSIS = "analysis"
    WEB_SEARCH = "web_search"


class AgentStatus(Enum):
    """Agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class AgentInput(BaseModel):
    """Standard input for all agents"""
    query: str
    context: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}


class AgentOutput(BaseModel):
    """Standard output from all agents"""
    result: Any
    confidence: float
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None
    execution_time_ms: float = 0.0


# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class Agent(ABC):
    """
    Base class for all agents

    Provides:
    - Common interface for all agents
    - Error handling
    - Execution timing
    - Confidence scoring
    """

    def __init__(self, name: str, agent_type: AgentType):
        """
        Initialize agent

        Args:
            name: Agent name
            agent_type: Agent type enum
        """
        self.name = name
        self.agent_type = agent_type
        self.execution_count = 0
        self.total_execution_time = 0.0
        logger.info(f"Initialized agent: {name} ({agent_type.value})")

    @abstractmethod
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute agent logic

        Args:
            input_data: Standardized input

        Returns:
            AgentOutput with results
        """
        pass

    @abstractmethod
    def can_handle(self, query: str) -> float:
        """
        Determine if agent can handle query

        Args:
            query: User query

        Returns:
            Confidence score (0.0 to 1.0)
        """
        pass

    async def execute_with_timing(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute with timing and metrics

        Args:
            input_data: Agent input

        Returns:
            Agent output with execution time
        """
        start_time = time.time()

        try:
            output = await self.execute(input_data)
            execution_time_ms = (time.time() - start_time) * 1000

            output.execution_time_ms = execution_time_ms
            self.execution_count += 1
            self.total_execution_time += execution_time_ms

            logger.info(
                f"Agent {self.name} completed in {execution_time_ms:.2f}ms "
                f"(confidence: {output.confidence:.2f})"
            )

            return output

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Agent {self.name} failed: {str(e)}")

            return AgentOutput(
                result=None,
                confidence=0.0,
                error=str(e),
                execution_time_ms=execution_time_ms
            )

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


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class RAGAgent(Agent):
    """
    Agent for document Q&A using RAG

    Handles queries about:
    - Document content
    - Knowledge base questions
    - Information retrieval
    """

    def __init__(self):
        super().__init__("RAG Agent", AgentType.RAG)
        # In production, initialize vectorstore, embeddings, LLM
        self.keywords = [
            "document", "explain", "what is", "tell me about",
            "information", "details", "describe", "knowledge"
        ]

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute RAG query

        Args:
            input_data: Query input

        Returns:
            Retrieved and generated answer
        """
        # Simulate RAG pipeline
        await asyncio.sleep(0.1)  # Simulate retrieval

        # In production:
        # 1. Embed query
        # 2. Retrieve relevant chunks from vectorstore
        # 3. Generate answer using LLM with context

        mock_result = {
            "answer": f"Based on the documents, here's the answer to '{input_data.query}'...",
            "sources": ["doc1.pdf", "doc2.pdf"],
            "chunks_used": 3
        }

        return AgentOutput(
            result=mock_result,
            confidence=0.85,
            metadata={
                "agent_type": "rag",
                "sources_count": 2
            }
        )

    def can_handle(self, query: str) -> float:
        """Check if query is document-related"""
        query_lower = query.lower()

        # Keyword matching
        keyword_score = sum(
            1 for keyword in self.keywords
            if keyword in query_lower
        ) / len(self.keywords)

        # Question patterns
        question_patterns = [
            r"what (is|are|does)",
            r"explain",
            r"tell me about",
            r"describe"
        ]

        pattern_score = sum(
            1 for pattern in question_patterns
            if re.search(pattern, query_lower)
        ) / len(question_patterns)

        # Combined score
        score = max(keyword_score, pattern_score)

        logger.debug(f"RAGAgent.can_handle('{query[:50]}...'): {score:.2f}")
        return min(score * 2, 1.0)  # Scale up but cap at 1.0


class SQLAgent(Agent):
    """
    Agent for SQL generation and execution

    Handles queries about:
    - Database queries
    - Data retrieval
    - Analytics queries
    """

    def __init__(self, db_connection_string: Optional[str] = None):
        super().__init__("SQL Agent", AgentType.SQL)
        self.db_connection = db_connection_string
        self.keywords = [
            "database", "query", "sql", "select", "table",
            "data", "records", "rows", "count", "sum", "average"
        ]

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Generate and execute SQL

        Args:
            input_data: Query input

        Returns:
            Query results
        """
        # Simulate SQL generation and execution
        await asyncio.sleep(0.15)

        # In production:
        # 1. Use LLM to generate SQL from natural language
        # 2. Validate SQL query
        # 3. Execute against database
        # 4. Format results

        mock_sql = "SELECT * FROM products WHERE category = 'Electronics' LIMIT 10"
        mock_results = [
            {"id": 1, "name": "Laptop", "price": 999.99},
            {"id": 2, "name": "Mouse", "price": 29.99}
        ]

        return AgentOutput(
            result={
                "sql_query": mock_sql,
                "results": mock_results,
                "row_count": len(mock_results)
            },
            confidence=0.90,
            metadata={
                "agent_type": "sql",
                "query_type": "SELECT"
            }
        )

    def can_handle(self, query: str) -> float:
        """Check if query requires database access"""
        query_lower = query.lower()

        # SQL keyword matching
        keyword_score = sum(
            1 for keyword in self.keywords
            if keyword in query_lower
        ) / len(self.keywords)

        # SQL action patterns
        sql_patterns = [
            r"(get|fetch|retrieve|show).*(data|records|rows)",
            r"how many",
            r"count of",
            r"sum of",
            r"average",
            r"from (the )?(database|table)"
        ]

        pattern_score = sum(
            1 for pattern in sql_patterns
            if re.search(pattern, query_lower)
        ) / len(sql_patterns)

        score = max(keyword_score, pattern_score)

        logger.debug(f"SQLAgent.can_handle('{query[:50]}...'): {score:.2f}")
        return min(score * 2.5, 1.0)


class CodeAgent(Agent):
    """
    Agent for code generation

    Handles queries about:
    - Writing code
    - Code examples
    - Implementation help
    """

    def __init__(self):
        super().__init__("Code Agent", AgentType.CODE)
        self.keywords = [
            "code", "function", "implement", "write", "create",
            "program", "script", "algorithm", "python", "javascript"
        ]

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Generate code

        Args:
            input_data: Query input

        Returns:
            Generated code
        """
        # Simulate code generation
        await asyncio.sleep(0.2)

        # In production:
        # 1. Use code-specialized LLM (e.g., Codex, CodeLlama)
        # 2. Generate code with proper syntax
        # 3. Add docstrings and comments
        # 4. Optionally test generated code

        language = input_data.parameters.get("language", "python")

        mock_code = '''
def example_function(param1, param2):
    """
    Generated function based on the query

    Args:
        param1: First parameter
        param2: Second parameter

    Returns:
        Result of computation
    """
    return param1 + param2
'''

        return AgentOutput(
            result={
                "code": mock_code.strip(),
                "language": language,
                "explanation": f"Generated {language} code for: {input_data.query}"
            },
            confidence=0.80,
            metadata={
                "agent_type": "code",
                "language": language
            }
        )

    def can_handle(self, query: str) -> float:
        """Check if query requires code generation"""
        query_lower = query.lower()

        # Keyword matching
        keyword_score = sum(
            1 for keyword in self.keywords
            if keyword in query_lower
        ) / len(self.keywords)

        # Code action patterns
        code_patterns = [
            r"(write|create|implement|generate).*(code|function|script|program)",
            r"how (do|can) (i|we) (code|implement)",
            r"example (code|implementation)",
            r"show me (the )?(code|implementation)"
        ]

        pattern_score = sum(
            1 for pattern in code_patterns
            if re.search(pattern, query_lower)
        ) / len(code_patterns)

        score = max(keyword_score, pattern_score)

        logger.debug(f"CodeAgent.can_handle('{query[:50]}...'): {score:.2f}")
        return min(score * 2, 1.0)


class AnalysisAgent(Agent):
    """
    Agent for data analysis

    Handles queries about:
    - Data analysis
    - Statistical computations
    - Trend analysis
    """

    def __init__(self):
        super().__init__("Analysis Agent", AgentType.ANALYSIS)
        self.keywords = [
            "analyze", "analysis", "trend", "pattern", "insight",
            "statistics", "correlation", "distribution"
        ]

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Perform data analysis

        Args:
            input_data: Query input

        Returns:
            Analysis results
        """
        await asyncio.sleep(0.25)

        # In production: Use pandas, numpy, scipy for analysis

        mock_analysis = {
            "summary": "Analysis of the provided data",
            "insights": [
                "Trend is increasing over time",
                "Strong correlation between variables X and Y"
            ],
            "metrics": {
                "mean": 42.5,
                "median": 40.0,
                "std_dev": 12.3
            }
        }

        return AgentOutput(
            result=mock_analysis,
            confidence=0.75,
            metadata={
                "agent_type": "analysis",
                "analysis_type": "descriptive"
            }
        )

    def can_handle(self, query: str) -> float:
        """Check if query requires data analysis"""
        query_lower = query.lower()

        keyword_score = sum(
            1 for keyword in self.keywords
            if keyword in query_lower
        ) / len(self.keywords)

        analysis_patterns = [
            r"analyze (the )?data",
            r"what (are|is) the (trend|pattern|insight)",
            r"statistical analysis"
        ]

        pattern_score = sum(
            1 for pattern in analysis_patterns
            if re.search(pattern, query_lower)
        ) / len(analysis_patterns)

        score = max(keyword_score, pattern_score)

        logger.debug(f"AnalysisAgent.can_handle('{query[:50]}...'): {score:.2f}")
        return min(score * 2, 1.0)


class WebSearchAgent(Agent):
    """
    Agent for web search

    Handles queries about:
    - Current events
    - Real-time information
    - External data
    """

    def __init__(self):
        super().__init__("Web Search Agent", AgentType.WEB_SEARCH)
        self.keywords = [
            "search", "find", "look up", "current", "latest",
            "recent", "news", "today", "now"
        ]

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Perform web search

        Args:
            input_data: Query input

        Returns:
            Search results
        """
        await asyncio.sleep(0.3)

        # In production: Use search API (Google, Bing, etc.)

        mock_results = [
            {
                "title": "Relevant Article",
                "url": "https://example.com/article",
                "snippet": "This article discusses..."
            }
        ]

        return AgentOutput(
            result={
                "query": input_data.query,
                "results": mock_results,
                "result_count": len(mock_results)
            },
            confidence=0.70,
            metadata={
                "agent_type": "web_search",
                "source": "web"
            }
        )

    def can_handle(self, query: str) -> float:
        """Check if query requires web search"""
        query_lower = query.lower()

        keyword_score = sum(
            1 for keyword in self.keywords
            if keyword in query_lower
        ) / len(self.keywords)

        web_patterns = [
            r"(search|find|look up).*(online|web|internet)",
            r"(latest|recent|current|today)",
            r"what('s| is) happening"
        ]

        pattern_score = sum(
            1 for pattern in web_patterns
            if re.search(pattern, query_lower)
        ) / len(web_patterns)

        score = max(keyword_score, pattern_score)

        logger.debug(f"WebSearchAgent.can_handle('{query[:50]}...'): {score:.2f}")
        return min(score * 2, 1.0)


# ============================================================================
# QUERY ROUTER
# ============================================================================

class QueryRouter:
    """
    Route queries to appropriate agents

    Uses confidence scoring to select best agent(s)
    """

    def __init__(
        self,
        agents: List[Agent],
        confidence_threshold: float = 0.3,
        max_agents: int = 3
    ):
        """
        Initialize query router

        Args:
            agents: List of available agents
            confidence_threshold: Minimum confidence to consider agent
            max_agents: Maximum agents to route to
        """
        self.agents = agents
        self.confidence_threshold = confidence_threshold
        self.max_agents = max_agents
        logger.info(f"QueryRouter initialized with {len(agents)} agents")

    def route(self, query: str) -> List[Tuple[Agent, float]]:
        """
        Route query to agent(s)

        Args:
            query: User query

        Returns:
            List of (agent, confidence) tuples, sorted by confidence
        """
        scores = []

        for agent in self.agents:
            confidence = agent.can_handle(query)
            if confidence >= self.confidence_threshold:
                scores.append((agent, confidence))

        # Sort by confidence descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top max_agents
        selected = scores[:self.max_agents]

        logger.info(
            f"Routed query to {len(selected)} agent(s): "
            f"{', '.join(f'{a.name}({c:.2f})' for a, c in selected)}"
        )

        return selected

    def classify_query(self, query: str) -> Optional[AgentType]:
        """
        Classify query to determine primary agent type

        Args:
            query: User query

        Returns:
            Most appropriate agent type or None
        """
        routed = self.route(query)

        if routed:
            best_agent, _ = routed[0]
            return best_agent.agent_type

        return None


# ============================================================================
# WORKFLOW ORCHESTRATION
# ============================================================================

@dataclass
class WorkflowStep:
    """Single step in workflow"""
    step_id: str
    agent: Agent
    input_data: AgentInput
    depends_on: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.PENDING
    output: Optional[AgentOutput] = None
    retry_count: int = 0


class WorkflowOrchestrator:
    """
    Orchestrate multi-agent workflows

    Features:
    - DAG execution
    - Parallel execution where possible
    - Error handling and retries
    - Result aggregation
    - Execution tracing
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        retry_delay: float = 1.0
    ):
        """
        Initialize orchestrator

        Args:
            max_retries: Maximum retry attempts
            timeout_seconds: Timeout for each step
            retry_delay: Delay between retries (seconds)
        """
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        self.retry_delay = retry_delay
        self.execution_history = []
        logger.info(
            f"WorkflowOrchestrator initialized: "
            f"max_retries={max_retries}, timeout={timeout_seconds}s"
        )

    async def execute_workflow(
        self,
        query: str,
        router: QueryRouter
    ) -> Dict[str, Any]:
        """
        Execute complete workflow for a query

        Args:
            query: User query
            router: Query router for agent selection

        Returns:
            Aggregated results from all agents
        """
        workflow_id = hashlib.md5(
            f"{query}{time.time()}".encode()
        ).hexdigest()[:8]

        logger.info(f"Starting workflow {workflow_id} for query: '{query}'")
        start_time = time.time()

        try:
            # Route query to agents
            routed_agents = router.route(query)

            if not routed_agents:
                logger.warning(f"No agents found for query: '{query}'")
                return {
                    "workflow_id": workflow_id,
                    "query": query,
                    "result": "No suitable agents found for this query",
                    "agents_used": [],
                    "total_time_ms": 0.0
                }

            # Create workflow steps
            steps = []
            for i, (agent, confidence) in enumerate(routed_agents):
                step = WorkflowStep(
                    step_id=f"step_{i}",
                    agent=agent,
                    input_data=AgentInput(query=query)
                )
                steps.append(step)

            # Build dependency graph (for now, all independent)
            graph = self.build_dependency_graph(steps)

            # Execute workflow
            context = {}
            await self.execute_dag(steps, graph, context)

            # Aggregate results
            outputs = [step.output for step in steps if step.output]
            aggregated = self.aggregate_results(outputs)

            total_time_ms = (time.time() - start_time) * 1000

            result = {
                "workflow_id": workflow_id,
                "query": query,
                "result": aggregated,
                "agents_used": [step.agent.name for step in steps],
                "total_time_ms": total_time_ms,
                "success": all(step.status == AgentStatus.COMPLETED for step in steps)
            }

            self.execution_history.append(result)
            logger.info(f"Workflow {workflow_id} completed in {total_time_ms:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            return {
                "workflow_id": workflow_id,
                "query": query,
                "error": str(e),
                "total_time_ms": (time.time() - start_time) * 1000
            }

    async def execute_dag(
        self,
        steps: List[WorkflowStep],
        graph: Dict[str, List[str]],
        context: Dict[str, Any]
    ):
        """
        Execute workflow steps as DAG

        Args:
            steps: All workflow steps
            graph: Dependency graph (adjacency list)
            context: Shared context for steps
        """
        # Get execution order (topological sort)
        levels = self.topological_sort(graph)

        logger.info(f"Executing {len(levels)} level(s) of workflow")

        step_map = {step.step_id: step for step in steps}

        # Execute each level (steps in same level run in parallel)
        for level_num, level_steps in enumerate(levels):
            logger.info(f"Executing level {level_num + 1}: {len(level_steps)} step(s)")

            # Get step objects
            level_step_objs = [step_map[step_id] for step_id in level_steps]

            # Execute in parallel
            await self.execute_parallel(level_step_objs, context)

    async def execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> AgentOutput:
        """
        Execute single workflow step with retry logic

        Args:
            step: Workflow step to execute
            context: Results from previous steps

        Returns:
            Agent output
        """
        logger.info(f"Executing step {step.step_id} with agent {step.agent.name}")

        for attempt in range(self.max_retries + 1):
            try:
                step.status = AgentStatus.RUNNING

                # Add context to input
                step.input_data.context = context.copy()

                # Execute with timeout
                output = await asyncio.wait_for(
                    step.agent.execute_with_timing(step.input_data),
                    timeout=self.timeout
                )

                if output.error:
                    raise Exception(output.error)

                step.output = output
                step.status = AgentStatus.COMPLETED

                logger.info(f"Step {step.step_id} completed successfully")
                return output

            except asyncio.TimeoutError:
                logger.warning(f"Step {step.step_id} timed out (attempt {attempt + 1})")
                step.status = AgentStatus.RETRYING
                step.retry_count += 1

                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    step.status = AgentStatus.FAILED
                    return AgentOutput(
                        result=None,
                        confidence=0.0,
                        error="Timeout after retries"
                    )

            except Exception as e:
                logger.error(
                    f"Step {step.step_id} failed: {str(e)} (attempt {attempt + 1})"
                )
                step.status = AgentStatus.RETRYING
                step.retry_count += 1

                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    step.status = AgentStatus.FAILED
                    return AgentOutput(
                        result=None,
                        confidence=0.0,
                        error=str(e)
                    )

    def build_dependency_graph(
        self,
        steps: List[WorkflowStep]
    ) -> Dict[str, List[str]]:
        """
        Build DAG from workflow steps

        Args:
            steps: Workflow steps

        Returns:
            Adjacency list representation {step_id: [dependent_step_ids]}
        """
        graph = defaultdict(list)

        for step in steps:
            if not graph[step.step_id]:
                graph[step.step_id] = []

            for dep_id in step.depends_on:
                graph[dep_id].append(step.step_id)

        return dict(graph)

    def topological_sort(
        self,
        graph: Dict[str, List[str]]
    ) -> List[List[str]]:
        """
        Sort workflow steps for execution using Kahn's algorithm

        Returns:
            List of levels (steps at same level can run in parallel)
        """
        # Calculate in-degrees
        in_degree = defaultdict(int)
        all_nodes = set(graph.keys())

        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
                all_nodes.add(neighbor)

        # Find nodes with no dependencies
        queue = deque([node for node in all_nodes if in_degree[node] == 0])
        levels = []

        while queue:
            # Process current level
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

    async def execute_parallel(
        self,
        steps: List[WorkflowStep],
        context: Dict[str, Any]
    ) -> List[AgentOutput]:
        """
        Execute multiple steps in parallel

        Args:
            steps: Steps to execute (no dependencies between them)
            context: Shared context

        Returns:
            List of outputs
        """
        logger.info(f"Executing {len(steps)} step(s) in parallel")

        # Create tasks for all steps
        tasks = [self.execute_step(step, context) for step in steps]

        # Execute in parallel
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        # Update context with results
        for step, output in zip(steps, outputs):
            if not isinstance(output, Exception) and output.result:
                context[step.step_id] = output.result

        return [o for o in outputs if not isinstance(o, Exception)]

    def aggregate_results(
        self,
        outputs: List[AgentOutput]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents

        Combines outputs intelligently based on:
        - Confidence scores
        - Agent types
        - Result quality

        Args:
            outputs: List of agent outputs

        Returns:
            Aggregated result
        """
        if not outputs:
            return {"message": "No results available"}

        # Filter successful outputs
        successful = [o for o in outputs if o.result and not o.error]

        if not successful:
            return {
                "message": "All agents failed",
                "errors": [o.error for o in outputs if o.error]
            }

        # Sort by confidence
        successful.sort(key=lambda x: x.confidence, reverse=True)

        # If single result, return it
        if len(successful) == 1:
            return successful[0].result

        # Multiple results - combine them
        aggregated = {
            "primary_result": successful[0].result,
            "primary_confidence": successful[0].confidence,
            "additional_results": [
                {
                    "result": o.result,
                    "confidence": o.confidence,
                    "agent": o.metadata.get("agent_type", "unknown")
                }
                for o in successful[1:]
            ],
            "total_agents": len(successful)
        }

        return aggregated


# ============================================================================
# EXECUTION TRACING
# ============================================================================

class WorkflowExecutionTracer:
    """Trace workflow execution for debugging and monitoring"""

    def __init__(self):
        """Initialize tracer"""
        self.traces = {}
        self.logger = logging.getLogger("WorkflowTracer")

    def start_trace(self, workflow_id: str, query: str):
        """
        Start tracing a workflow

        Args:
            workflow_id: Workflow identifier
            query: User query
        """
        self.traces[workflow_id] = {
            "workflow_id": workflow_id,
            "query": query,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "status": "running"
        }
        self.logger.info(f"Started trace for workflow {workflow_id}")

    def log_step(
        self,
        workflow_id: str,
        step_id: str,
        agent_name: str,
        status: str,
        duration_ms: float,
        error: Optional[str] = None
    ):
        """
        Log individual step execution

        Args:
            workflow_id: Workflow identifier
            step_id: Step identifier
            agent_name: Agent name
            status: Execution status
            duration_ms: Step duration
            error: Error message if failed
        """
        if workflow_id not in self.traces:
            return

        step_log = {
            "step_id": step_id,
            "agent": agent_name,
            "status": status,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        }

        if error:
            step_log["error"] = error

        self.traces[workflow_id]["steps"].append(step_log)

    def end_trace(self, workflow_id: str, total_duration_ms: float, success: bool):
        """
        End workflow trace

        Args:
            workflow_id: Workflow identifier
            total_duration_ms: Total duration
            success: Whether workflow succeeded
        """
        if workflow_id not in self.traces:
            return

        self.traces[workflow_id]["end_time"] = datetime.now().isoformat()
        self.traces[workflow_id]["total_duration_ms"] = total_duration_ms
        self.traces[workflow_id]["status"] = "completed" if success else "failed"

        self.logger.info(f"Ended trace for workflow {workflow_id}")

    def get_trace(self, workflow_id: str) -> Optional[Dict]:
        """
        Get execution trace for analysis

        Args:
            workflow_id: Workflow identifier

        Returns:
            Trace data or None
        """
        return self.traces.get(workflow_id)


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern for agent failures

    Prevents cascading failures by temporarily disabling failing agents
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker

        Args:
            failure_threshold: Failures before opening circuit
            timeout_seconds: Time before trying again
            success_threshold: Successes to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout_seconds
        self.success_threshold = success_threshold

        self.failures = defaultdict(int)
        self.successes = defaultdict(int)
        self.opened_at = {}
        self.state = defaultdict(lambda: "closed")  # closed, open, half-open

    def call(self, agent_name: str):
        """Check if agent can be called"""
        state = self.state[agent_name]

        if state == "closed":
            return True

        if state == "open":
            # Check if timeout expired
            if time.time() - self.opened_at[agent_name] > self.timeout:
                self.state[agent_name] = "half-open"
                logger.info(f"Circuit breaker for {agent_name} moving to half-open")
                return True
            return False

        if state == "half-open":
            return True

        return False

    def record_success(self, agent_name: str):
        """Record successful call"""
        self.failures[agent_name] = 0

        if self.state[agent_name] == "half-open":
            self.successes[agent_name] += 1

            if self.successes[agent_name] >= self.success_threshold:
                self.state[agent_name] = "closed"
                self.successes[agent_name] = 0
                logger.info(f"Circuit breaker for {agent_name} closed")

    def record_failure(self, agent_name: str):
        """Record failed call"""
        self.failures[agent_name] += 1

        if self.failures[agent_name] >= self.failure_threshold:
            self.state[agent_name] = "open"
            self.opened_at[agent_name] = time.time()
            logger.warning(
                f"Circuit breaker for {agent_name} opened "
                f"({self.failures[agent_name]} failures)"
            )


# ============================================================================
# MAIN DEMO
# ============================================================================

async def main():
    """Demo of multi-agent workflow system"""

    print("=" * 70)
    print("ELGO AI - MULTI-AGENT WORKFLOW SYSTEM DEMO")
    print("=" * 70)

    # Initialize agents
    agents = [
        RAGAgent(),
        SQLAgent(),
        CodeAgent(),
        AnalysisAgent(),
        WebSearchAgent()
    ]

    # Create router
    router = QueryRouter(agents=agents, confidence_threshold=0.2)

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(max_retries=2, timeout_seconds=10)

    # Test queries
    test_queries = [
        "Explain the concept of machine learning from our documents",
        "Get the top 10 customers by revenue from the database",
        "Write a Python function to calculate fibonacci numbers",
        "Analyze the sales trend for Q4 2024",
        "What is the latest news about AI developments?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print('=' * 70)

        result = await orchestrator.execute_workflow(query, router)

        print(f"\nWorkflow ID: {result['workflow_id']}")
        print(f"Agents Used: {', '.join(result.get('agents_used', []))}")
        print(f"Total Time: {result.get('total_time_ms', 0):.2f}ms")
        print(f"Success: {result.get('success', False)}")
        print(f"\nResult Preview:")
        print(f"  {str(result.get('result', 'N/A'))[:200]}...")

    # Show agent metrics
    print(f"\n{'='*70}")
    print("AGENT PERFORMANCE METRICS")
    print('=' * 70)

    for agent in agents:
        metrics = agent.get_metrics()
        print(f"\n{metrics['name']}:")
        print(f"  Executions: {metrics['execution_count']}")
        print(f"  Avg Time: {metrics['avg_execution_time_ms']:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
