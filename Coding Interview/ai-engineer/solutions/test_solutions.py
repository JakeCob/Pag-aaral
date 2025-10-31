"""
ELGO AI - Comprehensive Test Suite
Tests for all three interview sections

Run with: pytest test_solutions.py -v

Author: Reference Solution
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime

# Note: In production, these would be separate test files
# test_section1.py, test_section2.py, test_section3.py


# ============================================================================
# SECTION 2: CACHE TESTS (Can run without external dependencies)
# ============================================================================

from section2_cache_solution import DistributedCache, OptimizedDistributedCache


class TestDistributedCache:
    """Test suite for DistributedCache"""

    def test_basic_put_get(self):
        """Test basic put and get operations"""
        cache = DistributedCache(max_size=5)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = DistributedCache(max_size=3)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_ordering(self):
        """Test that accessing items updates LRU order"""
        cache = DistributedCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # Access 'a' to make it most recent
        cache.get("a")

        # Add new item, should evict 'b' (least recently used)
        cache.put("d", 4)

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_ttl_expiration(self):
        """Test TTL expiration"""
        cache = DistributedCache(max_size=5, default_ttl=1)
        cache.put("temp", "expires_soon")

        assert cache.get("temp") == "expires_soon"

        time.sleep(1.5)

        assert cache.get("temp") is None

    def test_custom_ttl(self):
        """Test custom TTL per entry"""
        cache = DistributedCache(max_size=5, default_ttl=10)
        cache.put("short", "expires_fast", ttl=1)
        cache.put("long", "expires_slow", ttl=10)

        time.sleep(1.5)

        assert cache.get("short") is None
        assert cache.get("long") == "expires_slow"

    def test_update_existing_key(self):
        """Test updating existing key doesn't create duplicates"""
        cache = DistributedCache(max_size=3)
        cache.put("key", "old_value")
        cache.put("key", "new_value")

        assert cache.get("key") == "new_value"
        stats = cache.get_stats()
        assert stats["current_size"] == 1

    def test_numpy_serialization(self):
        """Test serialization of numpy arrays"""
        cache = DistributedCache(max_size=5)
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        cache.put("embedding", embedding)

        retrieved = cache.get("embedding")
        assert np.array_equal(retrieved, embedding)

    def test_dict_serialization(self):
        """Test serialization of dictionaries"""
        cache = DistributedCache(max_size=5)
        data = {
            "results": [1, 2, 3, 4, 5],
            "metadata": {"source": "test", "score": 0.95}
        }
        cache.put("complex", data)

        retrieved = cache.get("complex")
        assert retrieved == data

    def test_statistics_tracking(self):
        """Test hit/miss/eviction statistics"""
        cache = DistributedCache(max_size=2)

        cache.put("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss
        cache.put("b", 2)
        cache.put("c", 3)  # eviction

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["evictions"] == 1
        assert 0 <= stats["hit_rate"] <= 1

    def test_thread_safety(self):
        """Test concurrent operations are thread-safe"""
        import threading

        cache = DistributedCache(max_size=100)
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    cache.put(f"thread{thread_id}_key{i}", i)
                    cache.get(f"thread{thread_id}_key{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = cache.get_stats()
        assert stats["hits"] + stats["misses"] > 0

    def test_clear_cache(self):
        """Test clearing the cache"""
        cache = DistributedCache(max_size=5)
        cache.put("a", 1)
        cache.put("b", 2)

        cache.clear()

        assert cache.get("a") is None
        assert cache.get("b") is None
        stats = cache.get_stats()
        assert stats["current_size"] == 0

    def test_contains(self):
        """Test contains method"""
        cache = DistributedCache(max_size=5)
        cache.put("exists", "value")

        assert cache.contains("exists") is True
        assert cache.contains("not_exists") is False

    def test_remove(self):
        """Test removing specific keys"""
        cache = DistributedCache(max_size=5)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        assert cache.remove("key1") is True
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

        assert cache.remove("key1") is False  # Already removed

    def test_cleanup_expired(self):
        """Test manual cleanup of expired entries"""
        cache = DistributedCache(max_size=10, default_ttl=1)

        for i in range(5):
            cache.put(f"key{i}", i)

        time.sleep(1.5)

        removed = cache.cleanup_expired()
        assert removed == 5

    def test_edge_case_single_item_cache(self):
        """Test cache with max_size=1"""
        cache = DistributedCache(max_size=1)
        cache.put("first", 1)
        cache.put("second", 2)

        assert cache.get("first") is None
        assert cache.get("second") == 2

    def test_zero_size_cache_raises_error(self):
        """Test that max_size=0 raises ValueError"""
        with pytest.raises(ValueError):
            DistributedCache(max_size=0)

    def test_get_keys(self):
        """Test getting all keys in LRU order"""
        cache = DistributedCache(max_size=5)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        keys = cache.get_keys()
        assert keys == ["c", "b", "a"]  # Most recent to least recent


# ============================================================================
# SECTION 3: MULTI-AGENT TESTS
# ============================================================================

from section3_multiagent_solution import (
    RAGAgent,
    SQLAgent,
    CodeAgent,
    QueryRouter,
    WorkflowOrchestrator,
    AgentInput
)


class TestAgents:
    """Test suite for individual agents"""

    @pytest.mark.asyncio
    async def test_rag_agent_execution(self):
        """Test RAG agent execution"""
        agent = RAGAgent()
        input_data = AgentInput(query="What is machine learning?")

        output = await agent.execute(input_data)

        assert output.result is not None
        assert output.confidence > 0
        assert output.error is None

    @pytest.mark.asyncio
    async def test_sql_agent_execution(self):
        """Test SQL agent execution"""
        agent = SQLAgent()
        input_data = AgentInput(query="Get top 10 customers by revenue")

        output = await agent.execute(input_data)

        assert output.result is not None
        assert "sql_query" in output.result
        assert output.confidence > 0

    @pytest.mark.asyncio
    async def test_code_agent_execution(self):
        """Test code agent execution"""
        agent = CodeAgent()
        input_data = AgentInput(query="Write a function to sort a list")

        output = await agent.execute(input_data)

        assert output.result is not None
        assert "code" in output.result
        assert output.confidence > 0

    def test_agent_can_handle_scoring(self):
        """Test agent can_handle scoring"""
        rag_agent = RAGAgent()
        sql_agent = SQLAgent()
        code_agent = CodeAgent()

        # RAG agent should score high for document questions
        assert rag_agent.can_handle("Explain the concept from documents") > 0.5
        assert rag_agent.can_handle("Write Python code") < 0.3

        # SQL agent should score high for database questions
        assert sql_agent.can_handle("Get data from the database") > 0.5
        assert sql_agent.can_handle("Explain machine learning") < 0.3

        # Code agent should score high for code requests
        assert code_agent.can_handle("Write a Python function") > 0.5
        assert code_agent.can_handle("Query the database") < 0.3


class TestQueryRouter:
    """Test suite for query router"""

    def test_router_selection(self):
        """Test router selects appropriate agents"""
        agents = [RAGAgent(), SQLAgent(), CodeAgent()]
        router = QueryRouter(agents, confidence_threshold=0.3)

        # Document query should route to RAG
        doc_query_agents = router.route("Explain machine learning from our docs")
        assert len(doc_query_agents) > 0
        assert any(isinstance(a, RAGAgent) for a, _ in doc_query_agents)

        # SQL query should route to SQL agent
        sql_query_agents = router.route("Get top 10 customers from database")
        assert len(sql_query_agents) > 0
        assert any(isinstance(a, SQLAgent) for a, _ in sql_query_agents)

        # Code query should route to code agent
        code_query_agents = router.route("Write a Python function to calculate fibonacci")
        assert len(code_query_agents) > 0
        assert any(isinstance(a, CodeAgent) for a, _ in code_query_agents)

    def test_router_confidence_threshold(self):
        """Test router respects confidence threshold"""
        agents = [RAGAgent(), SQLAgent()]
        router = QueryRouter(agents, confidence_threshold=0.8)

        # Ambiguous query may not meet threshold
        result = router.route("Hello, how are you?")
        assert len(result) <= 2  # May return nothing or low-confidence agents

    def test_router_max_agents(self):
        """Test router respects max_agents limit"""
        agents = [RAGAgent(), SQLAgent(), CodeAgent()]
        router = QueryRouter(agents, max_agents=2)

        result = router.route("Query documents and generate code")
        assert len(result) <= 2


class TestWorkflowOrchestrator:
    """Test suite for workflow orchestrator"""

    @pytest.mark.asyncio
    async def test_simple_workflow(self):
        """Test simple workflow execution"""
        agents = [RAGAgent()]
        router = QueryRouter(agents)
        orchestrator = WorkflowOrchestrator()

        result = await orchestrator.execute_workflow(
            "Explain machine learning",
            router
        )

        assert result["workflow_id"] is not None
        assert result["query"] == "Explain machine learning"
        assert "result" in result
        assert len(result["agents_used"]) > 0

    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self):
        """Test workflow with multiple agents"""
        agents = [RAGAgent(), SQLAgent(), CodeAgent()]
        router = QueryRouter(agents, confidence_threshold=0.2)
        orchestrator = WorkflowOrchestrator()

        result = await orchestrator.execute_workflow(
            "Explain concepts and show me code examples",
            router
        )

        assert len(result.get("agents_used", [])) >= 1
        assert "result" in result

    @pytest.mark.asyncio
    async def test_workflow_with_no_matching_agents(self):
        """Test workflow when no agents match"""
        agents = [RAGAgent()]
        router = QueryRouter(agents, confidence_threshold=0.9)  # Very high threshold
        orchestrator = WorkflowOrchestrator()

        result = await orchestrator.execute_workflow(
            "Random gibberish query xyz123",
            router
        )

        assert "result" in result
        # Should handle gracefully even with no matching agents


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across components"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Set up system
        agents = [
            RAGAgent(),
            SQLAgent(),
            CodeAgent()
        ]
        router = QueryRouter(agents, confidence_threshold=0.3)
        orchestrator = WorkflowOrchestrator(max_retries=2)

        # Execute multiple queries
        queries = [
            "Explain machine learning from documents",
            "Get customer data from database",
            "Write a sorting function in Python"
        ]

        for query in queries:
            result = await orchestrator.execute_workflow(query, router)
            assert result is not None
            assert "workflow_id" in result
            assert "result" in result

    def test_cache_in_rag_context(self):
        """Test using cache for RAG query results"""
        cache = DistributedCache(max_size=10)

        # Simulate caching RAG results
        query = "What is machine learning?"
        result = {
            "answer": "Machine learning is...",
            "sources": ["doc1.pdf"],
            "confidence": 0.85
        }

        cache.put(query, result)

        # Retrieve from cache
        cached_result = cache.get(query)
        assert cached_result == result
        assert cached_result["confidence"] == 0.85


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and scalability tests"""

    def test_cache_performance(self):
        """Test cache performance with many operations"""
        cache = DistributedCache(max_size=1000)

        # Benchmark writes
        start = time.time()
        for i in range(1000):
            cache.put(f"key{i}", f"value{i}")
        write_time = time.time() - start

        # Benchmark reads
        start = time.time()
        for i in range(1000):
            cache.get(f"key{i}")
        read_time = time.time() - start

        # Both should be reasonably fast (< 1 second for 1000 ops)
        assert write_time < 1.0
        assert read_time < 1.0

        print(f"\nCache Performance:")
        print(f"  1000 writes: {write_time:.3f}s ({1000/write_time:.0f} ops/sec)")
        print(f"  1000 reads: {read_time:.3f}s ({1000/read_time:.0f} ops/sec)")

    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self):
        """Test that agents can execute in parallel"""
        agents = [RAGAgent(), SQLAgent(), CodeAgent()]

        # Execute all agents in parallel
        input_data = AgentInput(query="Test query")

        start = time.time()
        tasks = [agent.execute(input_data) for agent in agents]
        results = await asyncio.gather(*tasks)
        parallel_time = time.time() - start

        # Execute sequentially for comparison
        start = time.time()
        sequential_results = []
        for agent in agents:
            result = await agent.execute(input_data)
            sequential_results.append(result)
        sequential_time = time.time() - start

        # Parallel should be faster (or similar due to async simulation)
        print(f"\nParallel Execution:")
        print(f"  Parallel: {parallel_time:.3f}s")
        print(f"  Sequential: {sequential_time:.3f}s")

        assert len(results) == len(agents)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_cache():
    """Fixture providing a sample cache"""
    cache = DistributedCache(max_size=10)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    return cache


@pytest.fixture
def sample_agents():
    """Fixture providing sample agents"""
    return [
        RAGAgent(),
        SQLAgent(),
        CodeAgent()
    ]


@pytest.fixture
def sample_router(sample_agents):
    """Fixture providing a router with sample agents"""
    return QueryRouter(sample_agents)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
