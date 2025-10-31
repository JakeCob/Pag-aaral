# Challenge 03: Production Cache with Statistics (Advanced)

**Difficulty**: Advanced
**Time Estimate**: 45-55 minutes
**Interview Section**: Section 2 - Part C + Extensions

---

## 📋 Challenge Description

Build a **production-ready cache** with:
1. Statistics tracking (hit rate, evictions, etc.)
2. Serialization support (pickle for complex objects)
3. Read-write locks for better concurrency
4. Cache warming and bulk operations

---

## 🎯 Requirements

### Part A: Statistics Tracking (15 min)

```python
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

class ProductionCache:
    def get_stats(self) -> CacheStats:
        """Return cache statistics"""
```

### Part B: Serialization Support (15 min)

```python
import pickle

def put(self, key: str, value: Any, ttl: int = None):
    """
    Serialize value using pickle.
    Supports complex objects: numpy arrays, dataclasses, etc.
    """

def get(self, key: str) -> Any:
    """
    Deserialize value from pickle.
    """
```

### Part C: Read-Write Locks (10 min)

```python
from threading import RLock
from collections import defaultdict

class RWLock:
    """
    Read-Write lock for better concurrency.
    Multiple readers OR single writer.
    """

    def reader_lock(self):
        """Acquire read lock (shared)"""

    def writer_lock(self):
        """Acquire write lock (exclusive)"""
```

### Part D: Bulk Operations (10 min)

```python
def mget(self, keys: List[str]) -> Dict[str, Any]:
    """Get multiple keys at once"""

def mput(self, items: Dict[str, Any], ttl: int = None):
    """Put multiple items at once"""

def warm_cache(self, data_source):
    """Pre-populate cache from data source"""
```

---

## 📊 Example Usage

```python
cache = ProductionCache(capacity=1000)

# Bulk operations
cache.mput({
    "user:1": {"name": "Alice", "age": 30},
    "user:2": {"name": "Bob", "age": 25},
    "user:3": {"name": "Charlie", "age": 35}
}, ttl=300)

# Single operations
cache.put("config", {"db": "postgres", "port": 5432})

# Multi-get
users = cache.mget(["user:1", "user:2"])
print(users)  # {"user:1": {...}, "user:2": {...}}

# Statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2f}%")
print(f"Evictions: {stats.evictions}")
print(f"Total requests: {stats.hits + stats.misses}")
```

---

## ✅ Expected Output

```
=== Production Cache Test ===

Warming cache with 100 items...
✓ Loaded 100 items

Performing 1000 operations...
  get("user:42") → HIT
  get("user:99") → HIT
  get("user:500") → MISS
  ...

=== Statistics ===
Total requests: 1000
Hits: 847
Misses: 153
Hit rate: 84.70%
Evictions: 23
Expirations: 15
Current size: 100/1000

=== Performance ===
Avg get latency: 0.05ms
Avg put latency: 0.08ms
Throughput: 12,500 ops/sec
```

---

## 💡 Implementation Tips

### Statistics Tracking

```python
class ProductionCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.stats = CacheStats()

    def get(self, key: str) -> Any:
        if key in self.cache and not self._is_expired(key):
            self.stats.hits += 1
            # ...
            return value
        else:
            self.stats.misses += 1
            return None

    def _evict_lru(self):
        evicted = self.cache.popitem(last=False)
        self.stats.evictions += 1
        return evicted
```

### Serialization

```python
import pickle

def put(self, key: str, value: Any, ttl: int = None):
    # Serialize value
    serialized = pickle.dumps(value)

    with self.lock:
        self.cache[key] = serialized
        # ...

def get(self, key: str) -> Any:
    with self.lock:
        serialized = self.cache.get(key)
        if serialized:
            return pickle.loads(serialized)
        return None
```

### Read-Write Lock

```python
import threading

class RWLock:
    def __init__(self):
        self.readers = 0
        self.writer = False
        self.lock = threading.Lock()
        self.read_ready = threading.Condition(self.lock)
        self.write_ready = threading.Condition(self.lock)

    def reader_acquire(self):
        with self.lock:
            while self.writer:
                self.read_ready.wait()
            self.readers += 1

    def reader_release(self):
        with self.lock:
            self.readers -= 1
            if self.readers == 0:
                self.write_ready.notify()

    def writer_acquire(self):
        with self.lock:
            while self.writer or self.readers > 0:
                self.write_ready.wait()
            self.writer = True

    def writer_release(self):
        with self.lock:
            self.writer = False
            self.write_ready.notify()
            self.read_ready.notify_all()
```

---

## 🎓 Key Concepts

1. **Cache Metrics**: Hit rate, eviction rate, latency
2. **Serialization**: Pickle for Python object persistence
3. **Concurrency**: Read-write locks for better throughput
4. **Bulk Operations**: Reduce lock contention
5. **Cache Warming**: Pre-populate frequently accessed data

---

**Time Allocation**:
- Statistics: 15 min
- Serialization: 15 min
- RW Locks: 10 min
- Bulk operations: 10 min
- Testing: 5 min
- **Total**: 55 min

**Good luck!** 🎯
