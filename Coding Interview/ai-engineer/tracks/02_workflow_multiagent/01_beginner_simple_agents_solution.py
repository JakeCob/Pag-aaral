"""
Challenge 01: Simple Agent System (Beginner)
COMPLETE SOLUTION

This demonstrates a production-quality simple multi-agent system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import re


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, name: str):
        """
        Initialize base agent.

        Args:
            name: Agent name for identification
        """
        self.name = name

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """
        Check if this agent can handle the query.

        Args:
            query: User query string

        Returns:
            True if agent can handle, False otherwise
        """
        pass

    @abstractmethod
    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute the query and return result.

        Args:
            query: User query string

        Returns:
            Dictionary with keys: agent, result, confidence, success
        """
        pass


class SearchAgent(BaseAgent):
    """Agent specialized in searching through documents"""

    def __init__(self, name: str, documents: List[str]):
        """
        Initialize search agent.

        Args:
            name: Agent name
            documents: List of documents to search through
        """
        super().__init__(name)
        self.documents = documents

    def can_handle(self, query: str) -> bool:
        """Check if query is a search request"""
        keywords = ["search", "find", "lookup", "look up"]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in keywords)

    async def execute(self, query: str) -> Dict[str, Any]:
        """Search for relevant documents"""
        # Extract search term from query
        query_lower = query.lower()
        search_term = query_lower

        # Try to extract term after keyword
        for keyword in ["search for", "find", "lookup"]:
            if keyword in query_lower:
                search_term = query_lower.split(keyword)[-1].strip()
                break

        # Search documents for matches
        for doc in self.documents:
            # Check if any word from search_term appears in document
            if any(word in doc.lower() for word in search_term.split()):
                return {
                    "agent": self.name,
                    "result": doc,
                    "confidence": 0.85,
                    "success": True
                }

        # No match found
        return {
            "agent": self.name,
            "result": "No matching documents found.",
            "confidence": 0.0,
            "success": False
        }


class CalculatorAgent(BaseAgent):
    """Agent specialized in mathematical calculations"""

    def __init__(self, name: str):
        """
        Initialize calculator agent.

        Args:
            name: Agent name
        """
        super().__init__(name)

    def can_handle(self, query: str) -> bool:
        """Check if query contains a math expression"""
        # Check if query has numbers AND operators
        has_number = bool(re.search(r'\d', query))
        has_operator = bool(re.search(r'[+\-*/]', query))
        return has_number and has_operator

    async def execute(self, query: str) -> Dict[str, Any]:
        """Calculate the mathematical expression"""
        try:
            # Extract mathematical expression from query
            # Pattern matches: numbers, operators, spaces, parentheses
            match = re.search(r'[\d+\-*/\s().]+', query)
            if not match:
                raise ValueError("No valid expression found")

            expression = match.group()

            # Safely evaluate the expression
            result = self._safe_eval(expression)

            return {
                "agent": self.name,
                "result": result,
                "confidence": 1.0,
                "success": True
            }

        except Exception as e:
            return {
                "agent": self.name,
                "result": f"Error: {str(e)}",
                "confidence": 0.0,
                "success": False
            }

    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate mathematical expression.

        Args:
            expression: Math expression string

        Returns:
            Calculated result
        """
        # Remove whitespace
        expression = expression.strip()

        # Validate: only allow numbers, operators, spaces, parentheses
        if not re.match(r'^[\d+\-*/\s().]+$', expression):
            raise ValueError("Invalid characters in expression")

        # Use eval (validated input only)
        # Note: In production, prefer AST parsing for absolute safety
        try:
            result = eval(expression)
            return float(result)
        except Exception as e:
            raise ValueError(f"Cannot evaluate expression: {e}")


class WeatherAgent(BaseAgent):
    """Agent specialized in weather information (mock data for demo)"""

    def __init__(self, name: str):
        """
        Initialize weather agent.

        Args:
            name: Agent name
        """
        super().__init__(name)

    def can_handle(self, query: str) -> bool:
        """Check if query is about weather"""
        keywords = ["weather", "temperature", "forecast", "temp"]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in keywords)

    async def execute(self, query: str) -> Dict[str, Any]:
        """Return mock weather data"""
        # Try to extract location from query
        location = "Default"

        # Simple location extraction: look for "in <location>"
        match = re.search(r'\bin\s+(\w+)', query, re.IGNORECASE)
        if match:
            location = match.group(1).capitalize()

        # Return mock weather data
        return {
            "agent": self.name,
            "result": {
                "location": location,
                "temp": 25,  # Celsius
                "condition": "Clear"
            },
            "confidence": 0.75,
            "success": True
        }


class SimpleRouter:
    """Routes queries to appropriate agents"""

    def __init__(self):
        """Initialize router"""
        self.agents: List[BaseAgent] = []

    def register_agent(self, agent: BaseAgent):
        """
        Add an agent to the router.

        Args:
            agent: Agent instance to register
        """
        self.agents.append(agent)

    async def route(self, query: str) -> Dict[str, Any]:
        """
        Find the right agent and execute the query.

        Args:
            query: User query string

        Returns:
            Agent execution result
        """
        # Loop through agents and find one that can handle query
        for agent in self.agents:
            if agent.can_handle(query):
                return await agent.execute(query)

        # No agent can handle this query
        return {
            "agent": None,
            "result": "No agent available to handle this query.",
            "confidence": 0.0,
            "success": False
        }


# =============================================================================
# TESTING CODE
# =============================================================================

async def test_agents():
    """Test individual agents"""
    print("\n" + "="*70)
    print("Test 1: Individual Agents")
    print("="*70)

    # Test SearchAgent
    print("\n--- SearchAgent ---")
    search_agent = SearchAgent(
        name="Search",
        documents=[
            "Python is a high-level programming language.",
            "FastAPI is a modern web framework.",
            "LangChain helps build LLM applications."
        ]
    )

    test_queries = [
        "search for Python",
        "find FastAPI",
        "lookup LangChain"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"Can handle: {search_agent.can_handle(query)}")
        result = await search_agent.execute(query)
        print(f"Result: {result['result']}")
        print(f"Success: {result['success']}")

    # Test CalculatorAgent
    print("\n--- CalculatorAgent ---")
    calc_agent = CalculatorAgent(name="Calculator")

    test_queries = [
        "calculate 15 + 27",
        "what is 10 * 5 - 8?",
        "(100 + 50) / 3"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"Can handle: {calc_agent.can_handle(query)}")
        result = await calc_agent.execute(query)
        print(f"Result: {result['result']}")
        print(f"Success: {result['success']}")

    # Test WeatherAgent
    print("\n--- WeatherAgent ---")
    weather_agent = WeatherAgent(name="Weather")

    test_queries = [
        "what's the weather in Singapore?",
        "temperature in Tokyo",
        "weather forecast"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"Can handle: {weather_agent.can_handle(query)}")
        result = await weather_agent.execute(query)
        print(f"Result: {result['result']}")
        print(f"Success: {result['success']}")


async def test_router():
    """Test the routing system"""
    print("\n" + "="*70)
    print("Test 2: Router System")
    print("="*70)

    # Initialize agents
    search_agent = SearchAgent(
        name="Search",
        documents=[
            "Python is a high-level programming language.",
            "FastAPI is a modern web framework.",
            "LangChain helps build LLM applications."
        ]
    )
    calc_agent = CalculatorAgent(name="Calculator")
    weather_agent = WeatherAgent(name="Weather")

    # Create router and register agents
    router = SimpleRouter()
    router.register_agent(search_agent)
    router.register_agent(calc_agent)
    router.register_agent(weather_agent)

    print(f"\nRegistered {len(router.agents)} agents: ", end="")
    print(", ".join([agent.name for agent in router.agents]))

    # Test various queries
    test_queries = [
        "search for FastAPI",
        "calculate 42 * 10 + 8",
        "what's the weather?",
        "tell me a story",  # Should fail - no agent handles this
        "find Python information",
        "10 + 20 + 30",
        "temperature in London"
    ]

    for query in test_queries:
        print(f"\n{'-'*70}")
        print(f"Query: {query}")
        result = await router.route(query)
        print(f"Agent: {result['agent']}")
        print(f"Result: {result['result']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Success: {'✅' if result['success'] else '❌'}")


async def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*70)
    print("Test 3: Edge Cases")
    print("="*70)

    # Initialize router
    router = SimpleRouter()
    router.register_agent(SearchAgent("Search", documents=["Test document"]))
    router.register_agent(CalculatorAgent("Calculator"))
    router.register_agent(WeatherAgent("Weather"))

    # Test edge cases
    print("\n--- Empty query ---")
    result = await router.route("")
    print(f"Success: {result['success']}")

    print("\n--- Invalid calculation ---")
    result = await router.route("calculate xyz + abc")
    print(f"Agent: {result['agent']}")
    print(f"Success: {result['success']}")
    print(f"Result: {result['result']}")

    print("\n--- Case insensitivity ---")
    result1 = await router.route("SEARCH for test")
    result2 = await router.route("search for test")
    print(f"Uppercase agent: {result1['agent']}")
    print(f"Lowercase agent: {result2['agent']}")
    print(f"Both handled by same agent: {result1['agent'] == result2['agent']}")


async def main():
    """Run all tests"""
    await test_agents()
    await test_router()
    await test_edge_cases()

    print("\n" + "="*70)
    print("✅ All tests completed successfully!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
