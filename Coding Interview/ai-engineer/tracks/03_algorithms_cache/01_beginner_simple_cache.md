# Challenge 01: Simple LRU Cache (Beginner)

**Difficulty**: Beginner
**Time Estimate**: 25-35 minutes
**Interview Section**: Section 2 - Part A

---

## 📋 Challenge Description

Implement a **Least Recently Used (LRU) Cache** with fixed capacity. When the cache is full, remove the least recently used item before adding a new one.

### What is LRU?

**Least Recently Used** means: If the cache is full, evict the item that hasn't been accessed for the longest time.

**Example:**
```
Cache (capacity=3): []

put(1, "a") → [1]
put(2, "b") → [1, 2]
put(3, "c") → [1, 2, 3]
get(1)      → [2, 3, 1]  # 1 moved to front (most recent)
put(4, "d") → [3, 1, 4]  # 2 evicted (least recent)
```

---

## 🎯 Requirements

### Part A: Basic LRU Operations (20 min)

```python
class LRUCache:
    def __init__(self, capacity: int):
        """Initialize cache with fixed capacity"""

    def get(self, key: str) -> Any:
        """
        Get value for key.
        Mark key as recently used.
        Return None if key not found.
        """

    def put(self, key: str, value: Any) -> None:
        """
        Store key-value pair.
        Mark key as recently used.
        Evict LRU item if capacity exceeded.
        """

    def size(self) -> int:
        """Return current number of items in cache"""
```

### Part B: Implementation Requirements

1. **O(1) time complexity** for both `get()` and `put()`
2. **Data structure**: Use `OrderedDict` (Python) or Doubly-Linked List + HashMap
3. **Capacity enforcement**: Never exceed max capacity

---

## 📊 Example Usage

```python
cache = LRUCache(capacity=3)

# Add items
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(cache.size())  # 3

# Access item (moves to front)
print(cache.get("a"))  # 1

# Add another item (evicts LRU)
cache.put("d", 4)  # Evicts "b" (least recent)
print(cache.get("b"))  # None (evicted)
print(cache.get("a"))  # 1 (still present)

# Update existing key
cache.put("a", 100)  # Updates value, moves to front
print(cache.get("a"))  # 100
```

---

## ✅ Expected Output

```
=== LRU Cache Test ===

Initial size: 0

Adding 3 items...
put("key1", "value1")
put("key2", "value2")
put("key3", "value3")
Size: 3

get("key1") → "value1" (moved to front)

Adding 4th item (capacity=3)
put("key4", "value4")
>>> Evicted: key2 (least recently used)

get("key2") → None (evicted)
get("key1") → "value1" (still present)
get("key3") → "value3" (still present)
get("key4") → "value4" (still present)
```

---

## 💡 Implementation Tips

### Using OrderedDict (Simplest)

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None

        # Move to end (most recent)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        self.cache[key] = value

        # Evict LRU if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove first (oldest)
```

### Using Doubly-Linked List + HashMap (Interview Favorite)

```python
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Dummy head and tail
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_front(self, node: Node):
        """Add node right after head (most recent)"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _remove(self, node: Node):
        """Remove node from list"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None

        node = self.cache[key]
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any):
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_front(node)
        else:
            # Add new
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_front(node)

            # Evict if over capacity
            if len(self.cache) > self.capacity:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
```

---

## 🎓 Key Concepts

1. **LRU Eviction Policy**: Remove oldest accessed item
2. **O(1) Operations**: Both get and put must be constant time
3. **Data Structures**:
   - HashMap for O(1) lookup
   - Doubly-linked list for O(1) reordering
4. **Move-to-front**: Recent items stay at front of list

---

**Time Allocation**:
- Basic structure: 10 min
- Get/Put implementation: 10 min
- Testing: 5 min
- **Total**: 25-30 min

**Good luck!** 🎯
