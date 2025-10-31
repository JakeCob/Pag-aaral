"""
Challenge 01: Simple Agent System (Beginner)
DIY Starter Template

Complete the TODOs to build a multi-agent system with specialized agents.
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
        # TODO 1: Check if query contains search keywords
        # Keywords: "search", "find", "lookup", "look up"
        # Hint: Use query.lower() and check if any keyword is in it

        # YOUR CODE HERE
        return False

    async def execute(self, query: str) -> Dict[str, Any]:
        """Search for relevant documents"""
        # TODO 2: Extract search term from query
        # Example: "search for Python" -> search_term = "python"
        # YOUR CODE HERE
        search_term = ""

        # TODO 3: Search documents for matches
        # Simple approach: Check if any word from search_term is in document
        # YOUR CODE HERE
        for doc in self.documents:
            # Check if document matches
            # If match found, return success response
            pass

        # TODO 4: Return failure response if no match found
        # YOUR CODE HERE
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
        # TODO 5: Check if query has numbers and operators
        # Hint: Use regex to check for digits and operators (+, -, *, /)
        # YOUR CODE HERE
        return False

    async def execute(self, query: str) -> Dict[str, Any]:
        """Calculate the mathematical expression"""
        try:
            # TODO 6: Extract mathematical expression from query
            # Example: "calculate 10 + 20" -> expression = "10 + 20"
            # Hint: Use regex to match pattern like [\d+\-*/\s().]+
            # YOUR CODE HERE
            expression = ""

            # TODO 7: Safely evaluate the expression
            # WARNING: Never use eval() directly! Use _safe_eval() instead
            # YOUR CODE HERE
            result = 0  # Replace with actual calculation

            # TODO 8: Return success response
            # YOUR CODE HERE
            return {
                "agent": self.name,
                "result": result,
                "confidence": 1.0,
                "success": True
            }

        except Exception as e:
            # Return error response
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
        # TODO 9: Implement safe evaluation
        # Simple approach for beginners: Parse and calculate manually
        # Advanced: Use ast.parse() for safety

        # Simple implementation (handles +, -, *, / only):
        # 1. Remove all whitespace
        # 2. Use eval() ONLY after validating input contains only numbers and operators

        # YOUR CODE HERE
        # For now, use eval with validation (not ideal but acceptable for practice)
        expression = expression.strip()

        # Validate: only allow numbers, operators, spaces, parentheses
        if not re.match(r'^[\d+\-*/\s().]+$', expression):
            raise ValueError("Invalid expression")

        return eval(expression)  # Note: In production, use AST parsing!


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
        # TODO 10: Check if query contains weather keywords
        # Keywords: "weather", "temperature", "forecast"
        # YOUR CODE HERE
        return False

    async def execute(self, query: str) -> Dict[str, Any]:
        """Return mock weather data"""
        # TODO 11: Extract location from query (optional, use "Default" if not found)
        # Example: "weather in Singapore" -> location = "Singapore"
        # YOUR CODE HERE
        location = "Default"

        # TODO 12: Return mock weather data
        # YOUR CODE HERE
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
        # TODO 13: Initialize list to store agents
        self.agents = []  # YOUR CODE HERE

    def register_agent(self, agent: BaseAgent):
        """
        Add an agent to the router.

        Args:
            agent: Agent instance to register
        """
        # TODO 14: Add agent to the list
        # YOUR CODE HERE
        pass

    async def route(self, query: str) -> Dict[str, Any]:
        """
        Find the right agent and execute the query.

        Args:
            query: User query string

        Returns:
            Agent execution result
        """
        # TODO 15: Loop through agents and find one that can handle query
        # Hint: Call agent.can_handle(query) for each agent
        # YOUR CODE HERE

        for agent in self.agents:
            # Check if agent can handle
            # If yes, execute and return result
            pass

        # TODO 16: If no agent can handle, return error
        # YOUR CODE HERE
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

    query = "search for Python"
    print(f"Query: {query}")
    print(f"Can handle: {search_agent.can_handle(query)}")
    result = await search_agent.execute(query)
    print(f"Result: {result}")

    # Test CalculatorAgent
    print("\n--- CalculatorAgent ---")
    calc_agent = CalculatorAgent(name="Calculator")

    query = "calculate 15 + 27"
    print(f"Query: {query}")
    print(f"Can handle: {calc_agent.can_handle(query)}")
    result = await calc_agent.execute(query)
    print(f"Result: {result}")

    # Test WeatherAgent
    print("\n--- WeatherAgent ---")
    weather_agent = WeatherAgent(name="Weather")

    query = "what's the weather in Singapore?"
    print(f"Query: {query}")
    print(f"Can handle: {weather_agent.can_handle(query)}")
    result = await weather_agent.execute(query)
    print(f"Result: {result}")


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

    # Test various queries
    test_queries = [
        "search for FastAPI",
        "calculate 42 * 10 + 8",
        "what's the weather?",
        "tell me a story"  # Should fail - no agent handles this
    ]

    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        result = await router.route(query)
        print(f"Agent: {result['agent']}")
        print(f"Result: {result['result']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Success: {result['success']}")


async def main():
    """Run all tests"""
    # TODO 17: Complete all TODOs above, then run these tests
    await test_agents()
    await test_router()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
