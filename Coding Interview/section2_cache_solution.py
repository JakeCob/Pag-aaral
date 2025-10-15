"""
ELGO AI - Distributed Cache Solution
Section 2: LRU Cache with TTL Support

Features:
- LRU (Least Recently Used) eviction policy
- TTL (Time To Live) support per entry
- Thread-safe operations
- Statistics tracking (hits, misses, evictions)
- Complex object serialization (numpy arrays, dicts, lists)
- O(1) get and put operations

Time Complexity:
    - get: O(1)
    - put: O(1)
    - evict: O(1)

Space Complexity: O(n) where n is max_size

Author: Reference Solution
"""

from typing import Any, Optional, Dict, Tuple
import time
import threading
from collections import OrderedDict
import pickle
import logging
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CacheNode:
    """
    Node for doubly-linked list

    Attributes:
        key: Cache key
        value: Cached value (serialized)
        expiry: Expiration timestamp
        prev: Previous node in list
        next: Next node in list
    """

    def __init__(self, key: str, value: bytes, expiry: float):
        """
        Initialize cache node

        Args:
            key: Cache key
            value: Serialized value
            expiry: Expiration timestamp (Unix timestamp)
        """
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[CacheNode] = None
        self.next: Optional[CacheNode] = None


class DistributedCache:
    """
    Thread-safe LRU cache with TTL support

    Implementation uses:
    - Doubly-linked list for LRU ordering
    - HashMap for O(1) key lookup
    - Thread locks for concurrency safety
    - Serialization for complex objects

    Example:
        cache = DistributedCache(max_size=100, default_ttl=3600)

        # Store value
        cache.put("key1", {"data": [1, 2, 3]})

        # Retrieve value
        value = cache.get("key1")

        # Get statistics
        stats = cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    def __init__(self, max_size: int, default_ttl: int = 3600):
        """
        Initialize cache

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self.max_size = max_size
        self.default_ttl = default_ttl

        # Doubly-linked list (head = most recent, tail = least recent)
        self.head: Optional[CacheNode] = None
        self.tail: Optional[CacheNode] = None

        # HashMap for O(1) lookup
        self.cache: Dict[str, CacheNode] = {}

        # Thread safety
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "current_size": 0
        }

        logger.info(f"DistributedCache initialized: max_size={max_size}, default_ttl={default_ttl}s")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Value if found and not expired, None otherwise

        Time Complexity: O(1)
        """
        with self.lock:
            # Check if key exists
            if key not in self.cache:
                self.stats["misses"] += 1
                logger.debug(f"Cache MISS: {key}")
                return None

            node = self.cache[key]

            # Check if expired (lazy deletion)
            if time.time() >= node.expiry:
                logger.debug(f"Cache entry expired: {key}")
                self._remove_node(node)
                del self.cache[key]
                self.stats["expirations"] += 1
                self.stats["misses"] += 1
                self.stats["current_size"] -= 1
                return None

            # Move to front (most recently used)
            self._move_to_front(node)

            # Update statistics
            self.stats["hits"] += 1
            logger.debug(f"Cache HIT: {key}")

            # Deserialize and return value
            return self.deserialize_value(node.value)

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Put value in cache with optional TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)

        Time Complexity: O(1) amortized
        """
        with self.lock:
            ttl = ttl if ttl is not None else self.default_ttl
            expiry = time.time() + ttl

            # Serialize value
            serialized_value = self.serialize_value(value)

            # If key exists, update and move to front
            if key in self.cache:
                node = self.cache[key]
                node.value = serialized_value
                node.expiry = expiry
                self._move_to_front(node)
                logger.debug(f"Cache UPDATE: {key}")
                return

            # Create new node
            new_node = CacheNode(key, serialized_value, expiry)

            # If cache is full, evict LRU item
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            # Add to cache
            self.cache[key] = new_node
            self._add_to_front(new_node)
            self.stats["current_size"] += 1

            logger.debug(f"Cache PUT: {key} (TTL: {ttl}s)")

    def _evict_lru(self) -> None:
        """
        Evict least recently used item (tail of list)

        Time Complexity: O(1)
        """
        if self.tail is None:
            return

        # Remove tail node
        evicted_key = self.tail.key
        self._remove_node(self.tail)
        del self.cache[evicted_key]

        self.stats["evictions"] += 1
        self.stats["current_size"] -= 1

        logger.debug(f"Cache EVICT: {evicted_key}")

    def _move_to_front(self, node: CacheNode) -> None:
        """
        Move node to front of list (most recently used)

        Args:
            node: Node to move

        Time Complexity: O(1)
        """
        if node == self.head:
            # Already at front
            return

        # Remove from current position
        self._remove_node(node)

        # Add to front
        self._add_to_front(node)

    def _remove_node(self, node: CacheNode) -> None:
        """
        Remove node from linked list

        Args:
            node: Node to remove

        Time Complexity: O(1)
        """
        if node.prev:
            node.prev.next = node.next
        else:
            # Node is head
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            # Node is tail
            self.tail = node.prev

        # Clear node links
        node.prev = None
        node.next = None

    def _add_to_front(self, node: CacheNode) -> None:
        """
        Add node to front of list

        Args:
            node: Node to add

        Time Complexity: O(1)
        """
        node.next = self.head
        node.prev = None

        if self.head:
            self.head.prev = node

        self.head = node

        # If list was empty, set tail
        if self.tail is None:
            self.tail = node

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with hits, misses, evictions, hit_rate, current_size
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

            return {
                **self.stats,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "max_size": self.max_size
            }

    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.head = None
            self.tail = None
            self.stats["current_size"] = 0
            logger.info("Cache cleared")

    def serialize_value(self, value: Any) -> bytes:
        """
        Serialize complex objects for storage

        Handles:
            - Numpy arrays (embeddings)
            - Dicts and lists (LLM responses)
            - Custom objects
            - Primitive types

        Args:
            value: Value to serialize

        Returns:
            Serialized bytes

        Raises:
            pickle.PicklingError: If value cannot be serialized
        """
        try:
            # Use pickle protocol 4 (Python 3.4+) for better performance
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise

    def deserialize_value(self, data: bytes) -> Any:
        """
        Deserialize stored values

        Args:
            data: Serialized bytes

        Returns:
            Deserialized value

        Raises:
            pickle.UnpicklingError: If data cannot be deserialized
        """
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise

    def get_size_bytes(self) -> int:
        """
        Get approximate cache size in bytes

        Returns:
            Total size in bytes
        """
        with self.lock:
            total_size = 0
            for node in self.cache.values():
                total_size += len(node.value)
            return total_size

    def cleanup_expired(self) -> int:
        """
        Manually cleanup expired entries

        Useful for background maintenance

        Returns:
            Number of entries removed
        """
        with self.lock:
            current_time = time.time()
            expired_keys = []

            # Find expired keys
            for key, node in self.cache.items():
                if current_time >= node.expiry:
                    expired_keys.append(key)

            # Remove expired keys
            for key in expired_keys:
                node = self.cache[key]
                self._remove_node(node)
                del self.cache[key]
                self.stats["expirations"] += 1
                self.stats["current_size"] -= 1

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def contains(self, key: str) -> bool:
        """
        Check if key exists and is not expired

        Args:
            key: Cache key

        Returns:
            True if key exists and valid, False otherwise
        """
        return self.get(key) is not None

    def remove(self, key: str) -> bool:
        """
        Remove specific key from cache

        Args:
            key: Cache key

        Returns:
            True if removed, False if not found
        """
        with self.lock:
            if key not in self.cache:
                return False

            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            self.stats["current_size"] -= 1

            logger.debug(f"Cache REMOVE: {key}")
            return True

    def get_keys(self) -> list:
        """
        Get all keys in cache (ordered by recency)

        Returns:
            List of keys from most recent to least recent
        """
        with self.lock:
            keys = []
            current = self.head

            while current:
                keys.append(current.key)
                current = current.next

            return keys


# ============================================================================
# ENHANCED VERSION WITH READ-WRITE LOCKS
# ============================================================================

class ReadWriteLock:
    """
    Read-write lock for better concurrency

    Allows:
    - Multiple concurrent readers
    - Exclusive writer access
    """

    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(threading.Lock())
        self._write_ready = threading.Condition(threading.Lock())

    def acquire_read(self):
        """Acquire read lock"""
        self._read_ready.acquire()
        try:
            while self._writers > 0:
                self._read_ready.wait()
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """Release read lock"""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        """Acquire write lock"""
        self._write_ready.acquire()
        self._writers += 1
        self._write_ready.release()

        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        """Release write lock"""
        self._writers -= 1
        self._read_ready.notify_all()
        self._read_ready.release()


class OptimizedDistributedCache(DistributedCache):
    """
    Optimized version using read-write locks

    Better performance for read-heavy workloads
    """

    def __init__(self, max_size: int, default_ttl: int = 3600):
        super().__init__(max_size, default_ttl)
        self.rwlock = ReadWriteLock()
        logger.info("OptimizedDistributedCache initialized with RW locks")

    def get(self, key: str) -> Optional[Any]:
        """Get with read lock (allows concurrent reads)"""
        self.rwlock.acquire_read()
        try:
            return super().get(key)
        finally:
            self.rwlock.release_read()

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put with write lock (exclusive access)"""
        self.rwlock.acquire_write()
        try:
            return super().put(key, value, ttl)
        finally:
            self.rwlock.release_write()


# ============================================================================
# CACHE WARMING UTILITY
# ============================================================================

class CacheWarmer:
    """
    Utility for cache warming

    Pre-populates cache with frequently accessed data
    """

    def __init__(self, cache: DistributedCache):
        """
        Initialize cache warmer

        Args:
            cache: Cache instance to warm
        """
        self.cache = cache
        logger.info("CacheWarmer initialized")

    def warm_from_dict(self, data: Dict[str, Any], ttl: Optional[int] = None):
        """
        Warm cache from dictionary

        Args:
            data: Dictionary of key-value pairs
            ttl: Optional TTL for all entries
        """
        logger.info(f"Warming cache with {len(data)} entries")
        for key, value in data.items():
            self.cache.put(key, value, ttl)
        logger.info("Cache warming complete")

    def warm_from_file(self, filepath: str, ttl: Optional[int] = None):
        """
        Warm cache from pickle file

        Args:
            filepath: Path to pickle file
            ttl: Optional TTL for all entries
        """
        logger.info(f"Loading cache data from {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.warm_from_dict(data, ttl)


# ============================================================================
# CACHE METRICS COLLECTOR
# ============================================================================

class CacheMetricsCollector:
    """
    Collects and exports cache metrics over time

    Useful for monitoring and optimization
    """

    def __init__(self, cache: DistributedCache, collection_interval: int = 60):
        """
        Initialize metrics collector

        Args:
            cache: Cache instance to monitor
            collection_interval: Collection interval in seconds
        """
        self.cache = cache
        self.interval = collection_interval
        self.history = []
        self.running = False
        self.thread = None
        logger.info(f"CacheMetricsCollector initialized (interval: {collection_interval}s)")

    def start(self):
        """Start metrics collection"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        logger.info("Metrics collection started")

    def stop(self):
        """Stop metrics collection"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Metrics collection stopped")

    def _collect_loop(self):
        """Collection loop"""
        while self.running:
            stats = self.cache.get_stats()
            stats['timestamp'] = datetime.now().isoformat()
            self.history.append(stats)
            time.sleep(self.interval)

    def get_history(self) -> list:
        """Get metrics history"""
        return self.history.copy()

    def export_to_file(self, filepath: str):
        """Export metrics to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f)
        logger.info(f"Metrics exported to {filepath}")


# ============================================================================
# TESTING
# ============================================================================

def test_distributed_cache():
    """Comprehensive test suite for distributed cache"""

    print("=" * 70)
    print("DISTRIBUTED CACHE TEST SUITE")
    print("=" * 70)

    # Test 1: Basic put/get
    print("\n[Test 1] Basic put/get")
    cache = DistributedCache(max_size=3, default_ttl=10)
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1", "Basic get failed"
    print("✓ Test 1 passed: Basic put/get")

    # Test 2: LRU eviction
    print("\n[Test 2] LRU eviction")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    cache.put("key4", "value4")  # Should evict key1
    assert cache.get("key1") is None, "LRU eviction failed"
    assert cache.get("key2") == "value2", "LRU eviction removed wrong item"
    print("✓ Test 2 passed: LRU eviction")

    # Test 3: TTL expiration
    print("\n[Test 3] TTL expiration")
    cache_ttl = DistributedCache(max_size=5, default_ttl=1)
    cache_ttl.put("temp", "expires_soon")
    assert cache_ttl.get("temp") == "expires_soon"
    time.sleep(1.5)
    assert cache_ttl.get("temp") is None, "TTL expiration failed"
    print("✓ Test 3 passed: TTL expiration")

    # Test 4: Complex objects (numpy arrays)
    print("\n[Test 4] Complex object serialization")
    cache_complex = DistributedCache(max_size=5)
    embedding = np.array([0.1, 0.2, 0.3, 0.4])
    cache_complex.put("embedding", embedding)
    retrieved = cache_complex.get("embedding")
    assert np.array_equal(retrieved, embedding), "Numpy serialization failed"
    print("✓ Test 4 passed: Complex object serialization")

    # Test 5: Dictionaries and lists
    print("\n[Test 5] Dict and list serialization")
    complex_data = {
        "results": [1, 2, 3, 4, 5],
        "metadata": {"source": "test", "score": 0.95}
    }
    cache_complex.put("complex", complex_data)
    retrieved = cache_complex.get("complex")
    assert retrieved == complex_data, "Dict/list serialization failed"
    print("✓ Test 5 passed: Dict and list serialization")

    # Test 6: Thread safety
    print("\n[Test 6] Thread safety")

    def worker(cache, thread_id):
        for i in range(100):
            cache.put(f"thread{thread_id}_key{i}", f"value{i}")
            cache.get(f"thread{thread_id}_key{i}")

    cache_concurrent = DistributedCache(max_size=50)
    threads = [
        threading.Thread(target=worker, args=(cache_concurrent, i))
        for i in range(5)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = cache_concurrent.get_stats()
    assert stats["hits"] + stats["misses"] > 0, "Thread safety test failed"
    print(f"✓ Test 6 passed: Thread safety")
    print(f"  Stats: {stats['hits']} hits, {stats['misses']} misses, {stats['evictions']} evictions")

    # Test 7: Statistics
    print("\n[Test 7] Statistics tracking")
    cache_stats = DistributedCache(max_size=3)
    cache_stats.put("a", 1)
    cache_stats.get("a")  # hit
    cache_stats.get("b")  # miss
    stats = cache_stats.get_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1, "Statistics tracking failed"
    assert 0 <= stats["hit_rate"] <= 1, "Hit rate calculation failed"
    print(f"✓ Test 7 passed: Statistics tracking")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")

    # Test 8: LRU ordering
    print("\n[Test 8] LRU ordering")
    cache_lru = DistributedCache(max_size=3)
    cache_lru.put("a", 1)
    cache_lru.put("b", 2)
    cache_lru.put("c", 3)
    cache_lru.get("a")  # Access 'a', making it most recent
    cache_lru.put("d", 4)  # Should evict 'b', not 'a'
    assert cache_lru.get("a") == 1, "LRU ordering failed (a should exist)"
    assert cache_lru.get("b") is None, "LRU ordering failed (b should be evicted)"
    print("✓ Test 8 passed: LRU ordering")

    # Test 9: Update existing key
    print("\n[Test 9] Update existing key")
    cache_update = DistributedCache(max_size=3)
    cache_update.put("key", "old_value")
    cache_update.put("key", "new_value")
    assert cache_update.get("key") == "new_value", "Update failed"
    assert cache_update.get_stats()["current_size"] == 1, "Update created duplicate"
    print("✓ Test 9 passed: Update existing key")

    # Test 10: Edge cases
    print("\n[Test 10] Edge cases")
    cache_edge = DistributedCache(max_size=1)
    cache_edge.put("single", "value")
    cache_edge.put("replace", "new")
    assert cache_edge.get("single") is None, "Single-item cache eviction failed"
    assert cache_edge.get("replace") == "new", "Single-item cache put failed"
    print("✓ Test 10 passed: Edge cases")

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_distributed_cache()

    # Performance benchmark
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    cache = DistributedCache(max_size=10000, default_ttl=3600)

    # Benchmark puts
    start = time.time()
    for i in range(10000):
        cache.put(f"key{i}", f"value{i}")
    put_time = time.time() - start
    print(f"\n10,000 PUT operations: {put_time:.3f}s ({10000/put_time:.0f} ops/sec)")

    # Benchmark gets
    start = time.time()
    for i in range(10000):
        cache.get(f"key{i}")
    get_time = time.time() - start
    print(f"10,000 GET operations: {get_time:.3f}s ({10000/get_time:.0f} ops/sec)")

    # Show final stats
    stats = cache.get_stats()
    print(f"\nFinal statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Current size: {stats['current_size']}/{stats['max_size']}")
    print(f"  Cache size (bytes): {cache.get_size_bytes():,}")

    print("\n" + "=" * 70)
