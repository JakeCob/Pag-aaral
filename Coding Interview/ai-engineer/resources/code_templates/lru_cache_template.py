"""
LRU Cache Implementation Templates
Quick reference for cache algorithms
"""

from collections import OrderedDict
from typing import Any, Optional
import threading
import time


# ============================================================================
# METHOD 1: Using OrderedDict (Simplest)
# ============================================================================

class LRUCacheSimple:
    """
    LRU Cache using OrderedDict.
    O(1) for both get() and put().
    """

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        """Get value and mark as recently used"""
        if key not in self.cache:
            return None

        # Move to end (most recent)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        """Put value and evict LRU if needed"""
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Evict oldest if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove first (oldest)

    def size(self) -> int:
        """Return current cache size"""
        return len(self.cache)


# ============================================================================
# METHOD 2: Doubly-Linked List + HashMap (Interview Favorite)
# ============================================================================

class Node:
    """Node in doubly-linked list"""
    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None


class LRUCacheLinkedList:
    """
    LRU Cache using doubly-linked list + HashMap.
    Demonstrates understanding of data structures.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Dummy head and tail for easier manipulation
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_front(self, node: Node) -> None:
        """Add node right after head (most recent position)"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _remove(self, node: Node) -> None:
        """Remove node from its current position"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def get(self, key: str) -> Optional[Any]:
        """Get value and move to front"""
        if key not in self.cache:
            return None

        node = self.cache[key]

        # Move to front (most recent)
        self._remove(node)
        self._add_to_front(node)

        return node.value

    def put(self, key: str, value: Any) -> None:
        """Put value and evict LRU if needed"""
        if key in self.cache:
            # Update existing key
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_front(node)
        else:
            # Add new key
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_front(node)

            # Evict LRU if over capacity
            if len(self.cache) > self.capacity:
                # Remove from tail (least recent)
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]


# ============================================================================
# METHOD 3: With TTL Support (Production Use)
# ============================================================================

class TTLCache:
    """
    LRU Cache with Time-To-Live support.
    Items expire after specified duration.
    """

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.expiry = {}  # key -> expiration_timestamp

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Put value with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = never expires)
        """
        # Set expiration time
        if ttl is not None:
            self.expiry[key] = time.time() + ttl
        else:
            self.expiry[key] = None  # Never expires

        # Standard LRU put
        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Evict LRU if over capacity
        if len(self.cache) > self.capacity:
            evicted_key, _ = self.cache.popitem(last=False)
            if evicted_key in self.expiry:
                del self.expiry[evicted_key]

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired"""
        if key not in self.cache:
            return None

        # Check expiration
        if self._is_expired(key):
            # Lazy deletion
            del self.cache[key]
            del self.expiry[key]
            return None

        # Move to end and return
        self.cache.move_to_end(key)
        return self.cache[key]

    def _is_expired(self, key: str) -> bool:
        """Check if key has expired"""
        if key not in self.expiry:
            return False

        expiry_time = self.expiry[key]
        if expiry_time is None:
            return False  # Never expires

        return time.time() > expiry_time

    def cleanup_expired(self) -> int:
        """Remove all expired items, return count removed"""
        removed = 0
        expired_keys = []

        for key in list(self.cache.keys()):
            if self._is_expired(key):
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]
            del self.expiry[key]
            removed += 1

        return removed


# ============================================================================
# METHOD 4: Thread-Safe Cache
# ============================================================================

class ThreadSafeCache:
    """
    Thread-safe LRU Cache with locks.
    Safe for concurrent access.
    """

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Thread-safe get"""
        with self.lock:
            if key not in self.cache:
                return None

            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        """Thread-safe put"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)

            self.cache[key] = value

            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


# ============================================================================
# TESTING
# ============================================================================

def test_lru_cache():
    """Test basic LRU functionality"""
    print("Testing LRU Cache...")

    cache = LRUCacheSimple(capacity=3)

    # Add items
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    print(f"Size after adding 3 items: {cache.size()}")

    # Access item (moves to front)
    print(f"Get 'a': {cache.get('a')}")

    # Add 4th item (evicts LRU)
    cache.put("d", 4)
    print(f"Get 'b' (should be None, evicted): {cache.get('b')}")
    print(f"Get 'a' (should exist): {cache.get('a')}")

    print("✅ LRU Cache test passed!")


def test_ttl_cache():
    """Test TTL functionality"""
    print("\nTesting TTL Cache...")

    cache = TTLCache(capacity=10)

    # Add with 2-second TTL
    cache.put("temp", "value", ttl=2)
    print(f"Get 'temp' immediately: {cache.get('temp')}")

    time.sleep(1)
    print(f"Get 'temp' after 1s: {cache.get('temp')}")

    time.sleep(1.5)
    print(f"Get 'temp' after 2.5s (should be None): {cache.get('temp')}")

    # Add without TTL
    cache.put("permanent", "forever")
    time.sleep(3)
    print(f"Get 'permanent' after 3s: {cache.get('permanent')}")

    print("✅ TTL Cache test passed!")


if __name__ == "__main__":
    test_lru_cache()
    test_ttl_cache()
