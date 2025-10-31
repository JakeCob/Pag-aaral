"""
Multi-Agent System - VANILLA PYTHON (No LangChain)
Build agent orchestration from scratch

Shows how to build agent systems without frameworks.
Perfect for custom agent architectures!

Key Concepts:
- Agent base class with polymorphism
- Query routing logic
- Agent execution patterns
- Simple workflow orchestration
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import re
import asyncio


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    This pattern is used in production systems.

    Every agent must implement:
    - can_handle(): Decide if agent can process query
    - execute(): Process the query and return result
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """
        Determine if this agent can handle the query.

        Args:
            query: User query string

        Returns:
            True if agent can handle, False otherwise
        """
        pass

    @abstractmethod
    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute the query and return structured result.

        Returns:
            {
                "agent": str,          # Agent name
                "result": Any,         # Result data
                "confidence": float,   # 0.0-1.0
                "success": bool        # Execution success
            }
        """
        pass


class SearchAgent(BaseAgent):
    """
    Agent specialized in document search.
    Built from scratch - no frameworks!
    """

    def __init__(self, name: str, documents: List[str]):
        super().__init__(name)
        self.documents = documents

        # Pre-process documents for faster search
        self._index_documents()

    def _index_documents(self):
        """
        Simple inverted index: word -> list of doc indices.
        This makes search O(1) for word lookup instead of O(n) scan.
        """
        self.index = {}

        for doc_idx, doc in enumerate(self.documents):
            words = set(doc.lower().split())
            for word in words:
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append(doc_idx)

    def can_handle(self, query: str) -> bool:
        """Check if query is a search request"""
        search_keywords = ["search", "find", "lookup", "look up", "locate"]
        query_lower = query.lower()

        return any(keyword in query_lower for keyword in search_keywords)

    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Search documents for relevant matches.

        Algorithm:
        1. Extract search terms from query
        2. Use inverted index to find matching docs
        3. Rank by number of matching terms
        """
        try:
            # Extract search terms
            search_terms = self._extract_search_terms(query)

            # Find matching documents using inverted index
            doc_scores = {}
            for term in search_terms:
                if term in self.index:
                    for doc_idx in self.index[term]:
                        doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1

            # Get best matching document
            if doc_scores:
                best_doc_idx = max(doc_scores, key=doc_scores.get)
                best_doc = self.documents[best_doc_idx]
                match_count = doc_scores[best_doc_idx]
                confidence = min(match_count / len(search_terms), 1.0)

                return {
                    "agent": self.name,
                    "result": best_doc,
                    "confidence": confidence,
                    "success": True,
                    "matched_terms": match_count
                }

            return {
                "agent": self.name,
                "result": "No matching documents found.",
                "confidence": 0.0,
                "success": False
            }

        except Exception as e:
            return {
                "agent": self.name,
                "result": f"Error: {str(e)}",
                "confidence": 0.0,
                "success": False
            }

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract meaningful search terms from query"""
        # Remove common search keywords
        stop_keywords = ["search", "find", "lookup", "look", "up", "for", "about"]

        words = query.lower().split()
        search_terms = [
            word for word in words
            if word not in stop_keywords and len(word) > 2
        ]

        return search_terms


class CalculatorAgent(BaseAgent):
    """
    Agent specialized in mathematical calculations.
    Implements safe expression evaluation.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def can_handle(self, query: str) -> bool:
        """Check if query contains math expression"""
        # Check for numbers AND operators
        has_number = bool(re.search(r'\d', query))
        has_operator = bool(re.search(r'[+\-*/×÷]', query))

        return has_number and has_operator

    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Safely evaluate mathematical expression.

        Uses AST parsing for security (no eval of arbitrary code).
        """
        try:
            # Extract mathematical expression
            expression = self._extract_expression(query)

            # Safely evaluate
            result = self._safe_eval(expression)

            return {
                "agent": self.name,
                "result": result,
                "confidence": 1.0,
                "success": True,
                "expression": expression
            }

        except Exception as e:
            return {
                "agent": self.name,
                "result": f"Cannot evaluate: {str(e)}",
                "confidence": 0.0,
                "success": False
            }

    def _extract_expression(self, query: str) -> str:
        """Extract math expression from natural language"""
        # Replace word operators
        expression = query.replace("plus", "+")
        expression = expression.replace("minus", "-")
        expression = expression.replace("times", "*")
        expression = expression.replace("divided by", "/")

        # Extract pattern with numbers and operators
        match = re.search(r'[\d+\-*/().\s]+', expression)
        if match:
            return match.group().strip()

        raise ValueError("No valid math expression found")

    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate math expression using AST.
        This prevents code injection attacks!
        """
        import ast
        import operator

        # Allowed operators
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,  # Unary minus
        }

        def eval_node(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                return operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                return operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")

        # Parse expression
        tree = ast.parse(expression, mode='eval')
        result = eval_node(tree.body)

        return float(result)


class DataAgent(BaseAgent):
    """
    Agent specialized in data analysis queries.
    Shows how to handle different query types.
    """

    def __init__(self, name: str):
        super().__init__(name)
        # In production, this might connect to a database
        self.sample_data = {
            "users": 1250,
            "revenue": 45000,
            "growth_rate": 15.5
        }

    def can_handle(self, query: str) -> bool:
        """Check if query is data-related"""
        data_keywords = ["data", "statistics", "stats", "metrics", "numbers", "count", "total"]
        query_lower = query.lower()

        return any(keyword in query_lower for keyword in data_keywords)

    async def execute(self, query: str) -> Dict[str, Any]:
        """Return relevant data metrics"""
        try:
            # Simple keyword matching for demo
            query_lower = query.lower()

            if "user" in query_lower:
                result = f"Total users: {self.sample_data['users']}"
                metric = "users"
            elif "revenue" in query_lower:
                result = f"Total revenue: ${self.sample_data['revenue']:,}"
                metric = "revenue"
            elif "growth" in query_lower:
                result = f"Growth rate: {self.sample_data['growth_rate']}%"
                metric = "growth_rate"
            else:
                result = f"Available metrics: {', '.join(self.sample_data.keys())}"
                metric = "summary"

            return {
                "agent": self.name,
                "result": result,
                "confidence": 0.85,
                "success": True,
                "metric": metric
            }

        except Exception as e:
            return {
                "agent": self.name,
                "result": f"Error: {str(e)}",
                "confidence": 0.0,
                "success": False
            }


class SimpleRouter:
    """
    Route queries to appropriate agent.
    Built from scratch - shows orchestration logic!

    Routing strategies:
    1. First match (simple, fast)
    2. Best confidence (more sophisticated)
    3. Ensemble (multiple agents)
    """

    def __init__(self, strategy: str = "first_match"):
        """
        Initialize router.

        Args:
            strategy: "first_match" or "best_confidence"
        """
        self.agents: List[BaseAgent] = []
        self.strategy = strategy

    def register_agent(self, agent: BaseAgent):
        """Add agent to router"""
        self.agents.append(agent)
        print(f"✓ Registered agent: {agent.name}")

    async def route(self, query: str) -> Dict[str, Any]:
        """
        Route query to best agent.

        Returns structured result from agent execution.
        """
        if self.strategy == "first_match":
            return await self._route_first_match(query)
        elif self.strategy == "best_confidence":
            return await self._route_best_confidence(query)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    async def _route_first_match(self, query: str) -> Dict[str, Any]:
        """Route to first agent that can handle query"""
        for agent in self.agents:
            if agent.can_handle(query):
                print(f"→ Routing to: {agent.name}")
                return await agent.execute(query)

        # No agent can handle
        return {
            "agent": None,
            "result": "No agent available to handle this query.",
            "confidence": 0.0,
            "success": False
        }

    async def _route_best_confidence(self, query: str) -> Dict[str, Any]:
        """
        Execute all capable agents, return best result.
        More sophisticated but slower.
        """
        capable_agents = [
            agent for agent in self.agents
            if agent.can_handle(query)
        ]

        if not capable_agents:
            return {
                "agent": None,
                "result": "No agent available to handle this query.",
                "confidence": 0.0,
                "success": False
            }

        # Execute all capable agents in parallel
        print(f"→ Found {len(capable_agents)} capable agents, executing...")
        tasks = [agent.execute(query) for agent in capable_agents]
        results = await asyncio.gather(*tasks)

        # Return result with highest confidence
        best_result = max(results, key=lambda x: x["confidence"])
        print(f"→ Best result from: {best_result['agent']}")

        return best_result


# =============================================================================
# EXAMPLE USAGE & TESTING
# =============================================================================

async def test_individual_agents():
    """Test each agent separately"""
    print("\n" + "="*70)
    print("Testing Individual Agents")
    print("="*70)

    # Test SearchAgent
    print("\n--- SearchAgent ---")
    search_agent = SearchAgent(
        "SearchAgent",
        documents=[
            "Python is a programming language",
            "FastAPI is a web framework",
            "Machine learning uses neural networks"
        ]
    )

    query = "search for Python"
    print(f"Query: {query}")
    print(f"Can handle: {search_agent.can_handle(query)}")
    result = await search_agent.execute(query)
    print(f"Result: {result}")

    # Test CalculatorAgent
    print("\n--- CalculatorAgent ---")
    calc_agent = CalculatorAgent("CalculatorAgent")

    query = "calculate 25 * 4 + 10"
    print(f"Query: {query}")
    print(f"Can handle: {calc_agent.can_handle(query)}")
    result = await calc_agent.execute(query)
    print(f"Result: {result}")

    # Test DataAgent
    print("\n--- DataAgent ---")
    data_agent = DataAgent("DataAgent")

    query = "show me user statistics"
    print(f"Query: {query}")
    print(f"Can handle: {data_agent.can_handle(query)}")
    result = await data_agent.execute(query)
    print(f"Result: {result}")


async def test_router():
    """Test routing system"""
    print("\n" + "="*70)
    print("Testing Router System")
    print("="*70)

    # Create agents
    search_agent = SearchAgent(
        "Search",
        documents=[
            "Python is excellent for data science",
            "FastAPI enables fast API development",
            "Machine learning requires large datasets"
        ]
    )
    calc_agent = CalculatorAgent("Calculator")
    data_agent = DataAgent("Data")

    # Test first-match strategy
    print("\n--- Strategy: First Match ---")
    router = SimpleRouter(strategy="first_match")
    router.register_agent(search_agent)
    router.register_agent(calc_agent)
    router.register_agent(data_agent)

    test_queries = [
        "search for FastAPI",
        "calculate 100 / 5",
        "show me revenue data",
        "tell me a joke"  # No agent handles this
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = await router.route(query)
        print(f"   Agent: {result['agent']}")
        print(f"   Result: {result['result']}")
        print(f"   Success: {'✅' if result['success'] else '❌'}")

    # Test best-confidence strategy
    print("\n\n--- Strategy: Best Confidence ---")
    router2 = SimpleRouter(strategy="best_confidence")
    router2.register_agent(search_agent)
    router2.register_agent(calc_agent)
    router2.register_agent(data_agent)

    query = "search for data about Python"  # Both search and data agents can handle
    print(f"\n📝 Query: {query}")
    result = await router2.route(query)
    print(f"   Best agent: {result['agent']}")
    print(f"   Result: {result['result']}")
    print(f"   Confidence: {result['confidence']:.2f}")


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("VANILLA MULTI-AGENT SYSTEM")
    print("Built from scratch - No LangChain!")
    print("="*70)

    await test_individual_agents()
    await test_router()

    print("\n" + "="*70)
    print("✅ All tests complete!")
    print("\nKey Takeaways:")
    print("1. Agent pattern: can_handle() + execute()")
    print("2. Router orchestrates agent selection")
    print("3. Different routing strategies for different needs")
    print("4. No frameworks needed - just clean Python!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
