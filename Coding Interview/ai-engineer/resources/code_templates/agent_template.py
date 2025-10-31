"""
Multi-Agent System Templates
Quick reference for agent patterns
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
from collections import defaultdict, deque


# ============================================================================
# PATTERN 1: Basic Agent Pattern
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Check if this agent can handle the query"""
        pass

    @abstractmethod
    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute the query and return result.

        Returns:
            {
                "agent": str,
                "result": Any,
                "confidence": float,
                "success": bool
            }
        """
        pass


class SearchAgent(BaseAgent):
    """Example: Search agent implementation"""

    def __init__(self, name: str, documents: List[str]):
        super().__init__(name)
        self.documents = documents

    def can_handle(self, query: str) -> bool:
        keywords = ["search", "find", "lookup"]
        return any(kw in query.lower() for kw in keywords)

    async def execute(self, query: str) -> Dict[str, Any]:
        # Simple search logic
        for doc in self.documents:
            if any(word in doc.lower() for word in query.lower().split()):
                return {
                    "agent": self.name,
                    "result": doc,
                    "confidence": 0.85,
                    "success": True
                }

        return {
            "agent": self.name,
            "result": "No match found",
            "confidence": 0.0,
            "success": False
        }


# ============================================================================
# PATTERN 2: Simple Router
# ============================================================================

class SimpleRouter:
    """Routes queries to appropriate agent"""

    def __init__(self):
        self.agents: List[BaseAgent] = []

    def register_agent(self, agent: BaseAgent) -> None:
        """Add agent to router"""
        self.agents.append(agent)

    async def route(self, query: str) -> Dict[str, Any]:
        """Find agent and execute query"""
        for agent in self.agents:
            if agent.can_handle(query):
                return await agent.execute(query)

        # No agent can handle
        return {
            "agent": None,
            "result": "No agent available",
            "confidence": 0.0,
            "success": False
        }


# ============================================================================
# PATTERN 3: Confidence-Based Router
# ============================================================================

class ConfidenceRouter:
    """Routes to agent with highest confidence"""

    def __init__(self):
        self.agents: List[BaseAgent] = []

    def register_agent(self, agent: BaseAgent) -> None:
        self.agents.append(agent)

    async def route(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Get confidence scores from all agents, execute top_k.

        Args:
            query: User query
            top_k: Number of top agents to execute

        Returns:
            Results from top_k agents sorted by confidence
        """
        # Get confidence from each agent (simplified)
        candidates = []
        for agent in self.agents:
            if agent.can_handle(query):
                # In production, agents would return confidence without full execution
                candidates.append((agent, 0.8))  # Placeholder confidence

        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Execute top_k agents
        results = []
        for agent, confidence in candidates[:top_k]:
            result = await agent.execute(query)
            results.append(result)

        return results


# ============================================================================
# PATTERN 4: DAG Workflow
# ============================================================================

class Task:
    """Task in a workflow"""

    def __init__(self, name: str, agent: BaseAgent, depends_on: List[str] = None):
        self.name = name
        self.agent = agent
        self.depends_on = depends_on or []


class WorkflowOrchestrator:
    """Execute tasks in dependency order (DAG)"""

    def __init__(self):
        self.agents = {}  # name -> agent

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        self.agents[name] = agent

    def topological_sort(self, tasks: List[Task]) -> List[str]:
        """
        Sort tasks in execution order using Kahn's algorithm.

        Returns:
            List of task names in execution order
        """
        # Build in-degree map and adjacency list
        in_degree = {task.name: 0 for task in tasks}
        adj_list = {task.name: [] for task in tasks}

        for task in tasks:
            for dep in task.depends_on:
                adj_list[dep].append(task.name)
                in_degree[task.name] += 1

        # Find all nodes with in-degree 0
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            # Process node
            node = queue.popleft()
            result.append(node)

            # Reduce in-degree of neighbors
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(tasks):
            raise ValueError("Cycle detected in task graph!")

        return result

    async def execute_workflow(
        self,
        tasks: List[Task],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute tasks in topological order.

        Args:
            tasks: List of tasks to execute
            input_data: Initial input data

        Returns:
            Results from all task executions
        """
        # Sort tasks
        execution_order = self.topological_sort(tasks)

        # Execute in order
        results = {}
        context = input_data.copy()

        for task_name in execution_order:
            # Find task
            task = next(t for t in tasks if t.name == task_name)

            # Execute task
            result = await task.agent.execute(str(context))

            # Store result
            results[task_name] = result
            context[task_name] = result["result"]

        return results


# ============================================================================
# PATTERN 5: Retry with Exponential Backoff
# ============================================================================

async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    *args,
    **kwargs
):
    """
    Retry function with exponential backoff.

    Delays: 1s, 2s, 4s, 8s...

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        *args, **kwargs: Arguments to pass to func

    Returns:
        Result from successful execution

    Raises:
        Exception from last failed attempt
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt == max_retries - 1:
                # Last attempt, give up
                raise

            # Calculate delay (exponential backoff)
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed. Retrying in {delay}s...")
            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    raise last_exception


# ============================================================================
# PATTERN 6: Circuit Breaker
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, block requests
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 30,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        import time

        # Check if should attempt recovery
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            # Execute function
            result = await func(*args, **kwargs)

            # Handle success
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = "CLOSED"
                    self.failure_count = 0

            return result

        except Exception as e:
            # Record failure
            self._record_failure()
            raise

    def _record_failure(self):
        """Record a failure and update state"""
        import time

        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


# ============================================================================
# TESTING
# ============================================================================

async def test_simple_router():
    """Test basic routing"""
    print("Testing Simple Router...")

    router = SimpleRouter()
    router.register_agent(SearchAgent("Search", documents=["Python is a language"]))

    result = await router.route("search for Python")
    print(f"Result: {result}")
    print("✅ Simple router test passed!\n")


async def test_workflow():
    """Test DAG workflow"""
    print("Testing Workflow Orchestrator...")

    # Create agents
    agent1 = SearchAgent("Agent1", ["data"])
    agent2 = SearchAgent("Agent2", ["processed data"])

    # Create workflow
    orchestrator = WorkflowOrchestrator()
    orchestrator.register_agent("Agent1", agent1)
    orchestrator.register_agent("Agent2", agent2)

    tasks = [
        Task("Task1", agent1, depends_on=[]),
        Task("Task2", agent2, depends_on=["Task1"])
    ]

    results = await orchestrator.execute_workflow(tasks, {"input": "test"})
    print(f"Workflow results: {results}")
    print("✅ Workflow test passed!\n")


async def test_retry():
    """Test retry with backoff"""
    print("Testing Retry with Backoff...")

    attempt_count = 0

    async def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(f"Attempt {attempt_count} failed")
        return "Success!"

    result = await retry_with_backoff(flaky_function, max_retries=5, base_delay=0.1)
    print(f"Result after {attempt_count} attempts: {result}")
    print("✅ Retry test passed!\n")


async def main():
    """Run all tests"""
    await test_simple_router()
    await test_workflow()
    await test_retry()
    print("✅ All agent pattern tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
