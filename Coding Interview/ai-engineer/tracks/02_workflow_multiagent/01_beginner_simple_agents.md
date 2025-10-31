# Challenge 01: Simple Agent System (Beginner)

**Difficulty**: Beginner
**Time Estimate**: 30-40 minutes
**Interview Section**: Section 3 - Part A

---

## 📋 Challenge Description

Build a **simple multi-agent system** with 2-3 specialized agents that can handle different types of queries. This introduces the fundamental agent pattern used in production multi-agent systems.

### What are Agents?

Think of agents as specialized "experts":
- **SearchAgent**: Expert at finding information in documents
- **CalculatorAgent**: Expert at math calculations
- **CodeAgent**: Expert at code generation

Each agent knows:
1. What queries it can handle (`can_handle()`)
2. How to process those queries (`execute()`)

---

## 🎯 Requirements

### Part A: Base Agent Class (10 min)

1. **Abstract BaseAgent class** with:
   ```python
   from abc import ABC, abstractmethod

   class BaseAgent(ABC):
       def __init__(self, name: str):
           self.name = name

       @abstractmethod
       def can_handle(self, query: str) -> bool:
           """Check if this agent can handle the query"""
           pass

       @abstractmethod
       async def execute(self, query: str) -> Dict[str, Any]:
           """Execute the query and return result"""
           pass
   ```

2. **Return format**:
   ```python
   {
       "agent": "AgentName",
       "result": "...",
       "confidence": 0.0-1.0,
       "success": True/False
   }
   ```

### Part B: Implement 3 Specialized Agents (15 min)

#### 1. SearchAgent
- **Can handle**: Queries with keywords "search", "find", "lookup"
- **Executes**: Searches in a predefined list of documents
- **Example**: "search for Python tutorials" → finds matching documents

#### 2. CalculatorAgent
- **Can handle**: Queries with numbers and math operators (+, -, *, /, %)
- **Executes**: Evaluates the mathematical expression
- **Example**: "calculate 15 * 8 + 32" → returns 152

#### 3. WeatherAgent
- **Can handle**: Queries with keywords "weather", "temperature", "forecast"
- **Executes**: Returns mock weather data (for testing purposes)
- **Example**: "what's the weather in Singapore?" → returns temperature and conditions

### Part C: Simple Router (10 min)

1. **SimpleRouter class** with:
   ```python
   class SimpleRouter:
       def __init__(self):
           self.agents = []

       def register_agent(self, agent: BaseAgent):
           """Add an agent to the router"""

       async def route(self, query: str) -> Dict[str, Any]:
           """Find the right agent and execute the query"""
   ```

2. **Routing logic**:
   - Loop through all agents
   - Call `can_handle(query)` on each
   - Use the first agent that returns `True`
   - If no agent can handle it, return error

---

## 📊 Example Usage

```python
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

# Create router
router = SimpleRouter()
router.register_agent(search_agent)
router.register_agent(calc_agent)
router.register_agent(weather_agent)

# Test queries
result1 = await router.route("search for Python")
print(result1)
# {
#     "agent": "Search",
#     "result": "Python is a high-level programming language.",
#     "confidence": 0.9,
#     "success": True
# }

result2 = await router.route("calculate 15 + 27")
print(result2)
# {
#     "agent": "Calculator",
#     "result": 42.0,
#     "confidence": 1.0,
#     "success": True
# }

result3 = await router.route("weather in Singapore")
print(result3)
# {
#     "agent": "Weather",
#     "result": {"location": "Singapore", "temp": 28, "condition": "Sunny"},
#     "confidence": 0.8,
#     "success": True
# }

result4 = await router.route("tell me a joke")
print(result4)
# {
#     "agent": None,
#     "result": "No agent available to handle this query.",
#     "confidence": 0.0,
#     "success": False
# }
```

---

## ✅ Expected Output

```
=== Multi-Agent System Test ===

Query: "search for FastAPI"
Agent: Search
Result: FastAPI is a modern web framework.
Confidence: 0.85
Success: True

---

Query: "calculate 42 * 10 + 8"
Agent: Calculator
Result: 428.0
Confidence: 1.0
Success: True

---

Query: "what's the weather?"
Agent: Weather
Result: {'location': 'Default', 'temp': 25, 'condition': 'Clear'}
Confidence: 0.75
Success: True

---

Query: "tell me a story"
Agent: None
Result: No agent available to handle this query.
Confidence: 0.0
Success: False
```

---

## 🧪 Test Cases

### Test 1: Agent Registration
```python
router = SimpleRouter()
router.register_agent(SearchAgent("Search", documents=["test"]))
router.register_agent(CalculatorAgent("Calc"))

assert len(router.agents) == 2
```

### Test 2: Correct Routing
```python
# Search query should go to SearchAgent
result = await router.route("find information about Python")
assert result["agent"] == "Search"
assert result["success"] == True

# Math query should go to CalculatorAgent
result = await router.route("10 + 20")
assert result["agent"] == "Calculator"
assert result["result"] == 30.0
```

### Test 3: Unknown Query Handling
```python
result = await router.route("random unknown query xyz123")
assert result["success"] == False
assert result["agent"] is None
```

### Test 4: Case Insensitivity
```python
# Should work regardless of case
result1 = await router.route("SEARCH for data")
result2 = await router.route("search for data")

assert result1["agent"] == result2["agent"] == "Search"
```

---

## 💡 Implementation Tips

### SearchAgent with Simple Matching
```python
class SearchAgent(BaseAgent):
    def __init__(self, name: str, documents: List[str]):
        super().__init__(name)
        self.documents = documents

    def can_handle(self, query: str) -> bool:
        keywords = ["search", "find", "lookup", "look up"]
        return any(keyword in query.lower() for keyword in keywords)

    async def execute(self, query: str) -> Dict[str, Any]:
        # Extract search term (words after the keyword)
        query_lower = query.lower()
        for keyword in ["search for", "find", "lookup"]:
            if keyword in query_lower:
                search_term = query_lower.split(keyword)[-1].strip()
                break
        else:
            search_term = query_lower

        # Simple keyword matching
        for doc in self.documents:
            if any(word in doc.lower() for word in search_term.split()):
                return {
                    "agent": self.name,
                    "result": doc,
                    "confidence": 0.85,
                    "success": True
                }

        return {
            "agent": self.name,
            "result": "No matching documents found.",
            "confidence": 0.0,
            "success": False
        }
```

### CalculatorAgent with Safe Eval
```python
import re
import ast
import operator

class CalculatorAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name)
        # Safe operators
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow
        }

    def can_handle(self, query: str) -> bool:
        # Check if query contains numbers and operators
        has_number = bool(re.search(r'\d', query))
        has_operator = bool(re.search(r'[+\-*/]', query))
        return has_number and has_operator

    async def execute(self, query: str) -> Dict[str, Any]:
        try:
            # Extract mathematical expression
            expr = re.search(r'[\d+\-*/\s().]+', query).group()

            # Safe evaluation using AST
            result = self._safe_eval(expr)

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

    def _safe_eval(self, expr: str) -> float:
        """Safely evaluate math expression using AST"""
        node = ast.parse(expr, mode='eval')
        return self._eval_node(node.body)

    def _eval_node(self, node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.operators[type(node.op)](
                self._eval_node(node.left),
                self._eval_node(node.right)
            )
        else:
            raise ValueError(f"Unsupported operation")
```

---

## 🎓 Key Concepts to Demonstrate

1. **Abstract Base Classes (ABC)**: Using inheritance for common interface
2. **Polymorphism**: Each agent implements same methods differently
3. **Keyword Detection**: Simple pattern matching for query classification
4. **Safe Code Execution**: Using AST instead of `eval()` for security
5. **Error Handling**: Graceful failures with structured responses

---

## 🚀 Extensions (If Time Permits)

1. **Confidence Scoring**: Each agent returns confidence score
2. **Agent Priority**: Higher-priority agents checked first
3. **Logging**: Add logging for debugging
4. **Async Support**: Proper async/await for I/O operations

---

**Time Allocation**:
- Base Agent Class: 10 min
- 3 Specialized Agents: 15 min (5 min each)
- Router Implementation: 10 min
- Testing: 5 min
- **Total**: 40 min

**Good luck!** 🎯
