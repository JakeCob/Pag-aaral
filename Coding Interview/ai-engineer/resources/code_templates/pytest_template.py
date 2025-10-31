"""
Pytest Testing Templates
Quick reference for writing tests
"""

import pytest
import asyncio
from typing import List
from fastapi.testclient import TestClient


# ============================================================================
# BASIC TEST STRUCTURE
# ============================================================================

def test_basic_example():
    """Basic test example"""
    # Arrange
    x = 5
    y = 10

    # Act
    result = x + y

    # Assert
    assert result == 15
    assert isinstance(result, int)


def test_with_multiple_assertions():
    """Test with multiple checks"""
    data = {"name": "Alice", "age": 30}

    assert "name" in data
    assert data["name"] == "Alice"
    assert data["age"] > 18
    assert len(data) == 2


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Fixture providing test data"""
    return ["item1", "item2", "item3"]


@pytest.fixture
def sample_cache():
    """Fixture providing a cache instance"""
    from lru_cache_template import LRUCacheSimple
    cache = LRUCacheSimple(capacity=10)
    # Populate with test data
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    return cache


def test_using_fixture(sample_data):
    """Test using fixture"""
    assert len(sample_data) == 3
    assert "item1" in sample_data


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.parametrize("input,expected", [
    ("hello", 5),
    ("world", 5),
    ("test", 4),
    ("", 0),
])
def test_string_length(input, expected):
    """Test string length with multiple inputs"""
    assert len(input) == expected


@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (5, 10, 15),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_addition(a, b, expected):
    """Test addition with multiple cases"""
    assert a + b == expected


# ============================================================================
# EXCEPTION TESTING
# ============================================================================

def test_exception_raised():
    """Test that exception is raised"""
    with pytest.raises(ValueError):
        int("not a number")


def test_exception_message():
    """Test exception message"""
    with pytest.raises(ValueError, match="invalid literal"):
        int("invalid")


def test_zero_division():
    """Test division by zero"""
    with pytest.raises(ZeroDivisionError):
        1 / 0


# ============================================================================
# ASYNC TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_async_function():
    """Test async function"""
    async def fetch_data():
        await asyncio.sleep(0.1)
        return {"status": "success"}

    result = await fetch_data()
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_async_with_timeout():
    """Test async with timeout"""
    async def slow_function():
        await asyncio.sleep(2)
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_function(), timeout=0.5)


# ============================================================================
# FASTAPI TESTING
# ============================================================================

# Example FastAPI app (would be in your main file)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/items")
async def create_item(item: Item):
    return {"item": item, "id": 123}


# Test client
client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_create_item():
    """Test item creation endpoint"""
    response = client.post(
        "/items",
        json={"name": "Test Item", "price": 9.99}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["item"]["name"] == "Test Item"
    assert data["item"]["price"] == 9.99
    assert "id" in data


def test_invalid_request():
    """Test invalid request handling"""
    response = client.post(
        "/items",
        json={"name": "Test"}  # Missing required 'price' field
    )

    assert response.status_code == 422  # Unprocessable Entity


# ============================================================================
# CACHE TESTING
# ============================================================================

def test_cache_basic_operations(sample_cache):
    """Test basic cache operations"""
    # Test get
    assert sample_cache.get("key1") == "value1"

    # Test put
    sample_cache.put("key3", "value3")
    assert sample_cache.get("key3") == "value3"

    # Test missing key
    assert sample_cache.get("nonexistent") is None


def test_cache_eviction():
    """Test LRU eviction"""
    from lru_cache_template import LRUCacheSimple

    cache = LRUCacheSimple(capacity=2)

    # Fill cache
    cache.put("a", 1)
    cache.put("b", 2)

    # Access 'a' (makes it most recent)
    cache.get("a")

    # Add new item (should evict 'b', the LRU)
    cache.put("c", 3)

    assert cache.get("a") == 1  # Still exists
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == 3  # New item


def test_cache_update():
    """Test updating existing key"""
    from lru_cache_template import LRUCacheSimple

    cache = LRUCacheSimple(capacity=5)

    cache.put("key", "value1")
    assert cache.get("key") == "value1"

    cache.put("key", "value2")  # Update
    assert cache.get("key") == "value2"
    assert cache.size() == 1  # Size should still be 1


# ============================================================================
# AGENT TESTING
# ============================================================================

@pytest.mark.asyncio
async def test_agent_can_handle():
    """Test agent's can_handle method"""
    from agent_template import SearchAgent

    agent = SearchAgent("TestAgent", documents=["test doc"])

    assert agent.can_handle("search for something") == True
    assert agent.can_handle("find information") == True
    assert agent.can_handle("random query") == False


@pytest.mark.asyncio
async def test_agent_execution():
    """Test agent execution"""
    from agent_template import SearchAgent

    agent = SearchAgent(
        "TestAgent",
        documents=["Python is a programming language"]
    )

    result = await agent.execute("search for Python")

    assert result["success"] == True
    assert result["agent"] == "TestAgent"
    assert "Python" in result["result"]
    assert result["confidence"] > 0


@pytest.mark.asyncio
async def test_router():
    """Test agent router"""
    from agent_template import SimpleRouter, SearchAgent

    router = SimpleRouter()
    router.register_agent(SearchAgent("Search", ["doc1", "doc2"]))

    result = await router.route("search for doc1")

    assert result["success"] == True
    assert result["agent"] == "Search"


# ============================================================================
# MARKS AND SKIPS
# ============================================================================

@pytest.mark.slow
def test_slow_operation():
    """Test marked as slow (run with: pytest -m slow)"""
    import time
    time.sleep(1)
    assert True


@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    """Test to be implemented later"""
    pass


@pytest.mark.skipif(
    pytest.__version__ < "7.0",
    reason="Requires pytest 7.0 or higher"
)
def test_new_feature():
    """Test requiring specific pytest version"""
    assert True


# ============================================================================
# SETUP AND TEARDOWN
# ============================================================================

class TestWithSetup:
    """Test class with setup/teardown"""

    def setup_method(self):
        """Run before each test method"""
        self.data = []

    def teardown_method(self):
        """Run after each test method"""
        self.data = None

    def test_append(self):
        """Test appending to list"""
        self.data.append(1)
        assert len(self.data) == 1

    def test_clear(self):
        """Test clearing list"""
        self.data.append(1)
        self.data.clear()
        assert len(self.data) == 0


# ============================================================================
# MOCKING (with pytest-mock)
# ============================================================================

def test_with_mock(mocker):
    """Test with mocked function (requires pytest-mock)"""
    # Mock a function
    mock_func = mocker.patch('builtins.print')

    print("This won't actually print")

    mock_func.assert_called_once_with("This won't actually print")


# ============================================================================
# RUNNING TESTS
# ============================================================================

"""
Common pytest commands:

# Run all tests
pytest

# Run specific file
pytest test_file.py

# Run specific test
pytest test_file.py::test_function_name

# Run with verbose output
pytest -v

# Run with print statements shown
pytest -s

# Run tests matching pattern
pytest -k "test_cache"

# Run marked tests
pytest -m slow

# Run with coverage
pytest --cov=mymodule

# Stop on first failure
pytest -x

# Show slowest tests
pytest --durations=10

# Parallel execution (requires pytest-xdist)
pytest -n 4
"""


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
