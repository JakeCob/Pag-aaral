# Challenge 02: LRU Cache with TTL (Intermediate)

**Difficulty**: Intermediate
**Time Estimate**: 35-45 minutes
**Interview Section**: Section 2 - Part B

---

## 📋 Challenge Description

Extend the basic LRU cache with **Time-To-Live (TTL)** support. Items expire after a specified duration and are automatically removed.

### What is TTL?

**Time-To-Live**: Each cached item has an expiration time. Once expired, the item is treated as if it doesn't exist.

**Example:**
```
put("key1", "value1", ttl=5)  # Expires in 5 seconds

[3 seconds later]
get("key1") → "value1" (still valid)

[3 more seconds later, total 6 seconds]
get("key1") → None (expired)
```

---

## 🎯 Requirements

### Part A: TTL Management (15 min)

```python
class TTLCache:
    def put(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Store with optional TTL (seconds).
        If ttl=None, item never expires.
        """

    def get(self, key: str) -> Any:
        """
        Get value if exists and not expired.
        Lazy deletion: Check expiration on access.
        """

    def _is_expired(self, key: str) -> bool:
        """Check if key has expired"""
```

### Part B: Thread Safety (15 min)

```python
import threading

class ThreadSafeTTLCache:
    def __init__(self, capacity: int):
        self.cache = {}
        self.capacity = capacity
        self.lock = threading.Lock()  # For thread safety

    def get(self, key: str) -> Any:
        with self.lock:
            # Thread-safe get
            ...
```

### Part C: Cleanup Task (10 min)

```python
def cleanup_expired(self):
    """
    Remove all expired items.
    Call periodically to free memory.
    """
```

---

## 📊 Example Usage

```python
cache = TTLCache(capacity=100)

# Store with 5-second TTL
cache.put("session_token", "abc123", ttl=5)

time.sleep(3)
print(cache.get("session_token"))  # "abc123" (still valid)

time.sleep(3)  # Total 6 seconds
print(cache.get("session_token"))  # None (expired)

# Store without TTL (never expires)
cache.put("config", {"theme": "dark"})
time.sleep(100)
print(cache.get("config"))  # {"theme": "dark"} (still there)
```

---

## 💡 Implementation Tips

### TTL Tracking

```python
import time
from typing import Optional

class TTLCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.expiry = {}  # key -> expiration_timestamp

    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        # Calculate expiration time
        if ttl is not None:
            self.expiry[key] = time.time() + ttl
        else:
            self.expiry[key] = None  # Never expires

        # Standard LRU put
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        if len(self.cache) > self.capacity:
            evicted = self.cache.popitem(last=False)
            if evicted[0] in self.expiry:
                del self.expiry[evicted[0]]

    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None

        # Check if expired
        if self._is_expired(key):
            del self.cache[key]
            del self.expiry[key]
            return None

        # Move to end and return
        self.cache.move_to_end(key)
        return self.cache[key]

    def _is_expired(self, key: str) -> bool:
        if key not in self.expiry:
            return False

        expiry_time = self.expiry[key]
        if expiry_time is None:
            return False  # Never expires

        return time.time() > expiry_time
```

---

## 🎓 Key Concepts

1. **TTL (Time-To-Live)**: Automatic expiration
2. **Lazy Deletion**: Check expiration on access, not proactively
3. **Thread Safety**: Use locks for concurrent access
4. **Cleanup Strategy**: Periodic cleanup vs lazy deletion
5. **Dual Eviction**: By LRU and by TTL

---

**Time Allocation**:
- TTL logic: 15 min
- Thread safety: 15 min
- Cleanup: 10 min
- Testing: 5 min
- **Total**: 45 min

**Good luck!** 🎯
