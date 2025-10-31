# Caching Strategies Guide
**Reference Guide for ELGO AI Interview Preparation**

Based on: `section2_cache_solution.py`

---

## Table of Contents
1. [Overview](#overview)
2. [LRU Algorithm Implementation](#lru-algorithm-implementation)
3. [TTL Management](#ttl-management)
4. [Thread Safety Patterns](#thread-safety-patterns)
5. [Serialization & Complex Objects](#serialization--complex-objects)
6. [Performance Optimization](#performance-optimization)
7. [Production Patterns](#production-patterns)
8. [Common Pitfalls](#common-pitfalls)
9. [Code Templates](#code-templates)

---

## Overview

### Why Caching?

**RAG System Without Cache**:
```
Query → Retrieve (300ms) → Embed (100ms) → LLM (1000ms) → Answer
Total: ~1400ms per query
```

**RAG System With Cache**:
```
Query → Cache Hit → Answer (cached)
Total: ~5ms (280x faster!)
```

### Cache Requirements for AI Systems

1. **Fast Access**: O(1) get/put operations
2. **Eviction Policy**: LRU (Least Recently Used)
3. **TTL Support**: Embeddings go stale, LLM responses need refresh
4. **Thread Safety**: Multiple concurrent requests
5. **Complex Objects**: Numpy arrays (embeddings), dicts (LLM responses)

### Cache Architecture

```
┌─────────────────────────────────────┐
│          HashMap (O(1) lookup)      │
│  key1 → Node1                       │
│  key2 → Node2                       │
│  key3 → Node3                       │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│    Doubly-Linked List (LRU order)   │
│  Head (MRU) → Node3 ↔ Node1 ↔ Node2 → Tail (LRU) │
└─────────────────────────────────────┘
```

**Why This Design?**
- HashMap: O(1) key lookup
- Doubly-linked list: O(1) move to front, O(1) evict tail
- Combined: O(1) get, O(1) put

---

## LRU Algorithm Implementation

### Data Structures

**Cache Node** (`section2_cache_solution.py:40-66`)
```python
class CacheNode:
    """Node in doubly-linked list"""
    def __init__(self, key: str, value: bytes, expiry: float):
        self.key = key
        self.value = value  # Serialized
        self.expiry = expiry  # Unix timestamp
        self.prev: Optional[CacheNode] = None
        self.next: Optional[CacheNode] = None
```

**Key Design Decisions**:
1. **Doubly-linked** (not singly) → Can remove from middle in O(1)
2. **Store key in node** → Needed for HashMap cleanup on eviction
3. **Serialized value** → Handles any Python object
4. **Unix timestamp** → Efficient expiry checking

**Cache Structure** (`section2_cache_solution.py:106-111`)
```python
# Doubly-linked list (head = most recent, tail = least recent)
self.head: Optional[CacheNode] = None
self.tail: Optional[CacheNode] = None

# HashMap for O(1) lookup
self.cache: Dict[str, CacheNode] = {}
```

### Core Operations

#### GET Operation

**Algorithm** (`section2_cache_solution.py:127-166`)
```python
def get(self, key: str) -> Optional[Any]:
    """
    Get value from cache

    Time Complexity: O(1)
    """
    with self.lock:
        # 1. Check if key exists
        if key not in self.cache:
            self.stats["misses"] += 1
            return None

        node = self.cache[key]

        # 2. Check if expired (lazy deletion)
        if time.time() >= node.expiry:
            self._remove_node(node)
            del self.cache[key]
            self.stats["expirations"] += 1
            self.stats["misses"] += 1
            return None

        # 3. Move to front (most recently used)
        self._move_to_front(node)

        # 4. Update statistics
        self.stats["hits"] += 1

        # 5. Deserialize and return
        return self.deserialize_value(node.value)
```

**Key Points**:
- ✅ Lazy expiration (check on access, not background thread)
- ✅ Update LRU order (move accessed item to front)
- ✅ Track statistics (hits/misses)

#### PUT Operation

**Algorithm** (`section2_cache_solution.py:168-207`)
```python
def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
    """
    Put value in cache with optional TTL

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
```

**Key Points**:
- ✅ Handle updates (don't create duplicates)
- ✅ Evict before adding (maintain max_size)
- ✅ Custom TTL per entry

### Linked List Operations

#### Move to Front

**Algorithm** (`section2_cache_solution.py:228-245`)
```python
def _move_to_front(self, node: CacheNode) -> None:
    """Move node to front of list (most recently used)"""
    if node == self.head:
        return  # Already at front

    # Remove from current position
    self._remove_node(node)

    # Add to front
    self._add_to_front(node)
```

#### Remove Node

**Algorithm** (`section2_cache_solution.py:247-270`)
```python
def _remove_node(self, node: CacheNode) -> None:
    """Remove node from linked list - O(1)"""

    # Update previous node's next pointer
    if node.prev:
        node.prev.next = node.next
    else:
        # Node is head
        self.head = node.next

    # Update next node's prev pointer
    if node.next:
        node.next.prev = node.prev
    else:
        # Node is tail
        self.tail = node.prev

    # Clear node links (prevent memory leaks)
    node.prev = None
    node.next = None
```

**Why O(1)?**
- No traversal needed (have direct node reference from HashMap)
- Just update 4 pointers: prev.next, next.prev, node.prev, node.next

#### Add to Front

**Algorithm** (`section2_cache_solution.py:272-291`)
```python
def _add_to_front(self, node: CacheNode) -> None:
    """Add node to front of list - O(1)"""
    node.next = self.head
    node.prev = None

    if self.head:
        self.head.prev = node

    self.head = node

    # If list was empty, set tail
    if self.tail is None:
        self.tail = node
```

#### Evict LRU

**Algorithm** (`section2_cache_solution.py:209-226`)
```python
def _evict_lru(self) -> None:
    """Evict least recently used item (tail) - O(1)"""
    if self.tail is None:
        return

    # Remove tail node
    evicted_key = self.tail.key
    self._remove_node(self.tail)
    del self.cache[evicted_key]

    self.stats["evictions"] += 1
    self.stats["current_size"] -= 1
```

---

## TTL Management

### Why TTL Matters in AI Systems

**Problem**: Cached data becomes stale
- **Embeddings**: Model updated → old embeddings invalid
- **LLM Responses**: Facts change, newer data available
- **Document Chunks**: Document updated → cache outdated

**Solution**: Time-To-Live (TTL)

### TTL Strategies

#### 1. Lazy Deletion (Used in Solution)

**Pros**:
- ✅ No background thread overhead
- ✅ Simple implementation
- ✅ Automatic cleanup on access

**Cons**:
- ❌ Expired entries occupy memory until accessed
- ❌ get_size_bytes() includes expired entries

**Implementation** (`section2_cache_solution.py:148-156`)
```python
# Check if expired on GET
if time.time() >= node.expiry:
    self._remove_node(node)
    del self.cache[key]
    self.stats["expirations"] += 1
    return None
```

#### 2. Active Cleanup (Manual)

**When to Use**:
- Long-running cache with infrequent access
- Need accurate memory usage
- Scheduled maintenance windows

**Implementation** (`section2_cache_solution.py:378-407`)
```python
def cleanup_expired(self) -> int:
    """Manually cleanup expired entries"""
    with self.lock:
        current_time = time.time()
        expired_keys = []

        # Find expired keys (O(n))
        for key, node in self.cache.items():
            if current_time >= node.expiry:
                expired_keys.append(key)

        # Remove expired keys
        for key in expired_keys:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            self.stats["expirations"] += 1

        return len(expired_keys)
```

**Usage Pattern**:
```python
# Schedule cleanup every 5 minutes
import threading

def cleanup_loop(cache):
    while True:
        time.sleep(300)  # 5 minutes
        removed = cache.cleanup_expired()
        logger.info(f"Cleaned up {removed} expired entries")

cleanup_thread = threading.Thread(target=cleanup_loop, args=(cache,), daemon=True)
cleanup_thread.start()
```

### TTL Configuration Best Practices

**For AI Systems**:

| Data Type | Recommended TTL | Reason |
|-----------|----------------|---------|
| **Embeddings** | 1-7 days | Model updates infrequent |
| **LLM Responses** | 1-24 hours | Facts may change |
| **RAG Query Results** | 30 min - 2 hours | Documents may update |
| **BM25 Scores** | 6-24 hours | Stable unless doc changes |
| **User Sessions** | 15-60 minutes | Conversation context |

**Example Usage**:
```python
cache = DistributedCache(max_size=10000, default_ttl=3600)  # 1 hour default

# Short TTL for dynamic content
cache.put("llm_response", answer, ttl=1800)  # 30 min

# Long TTL for stable data
cache.put("embedding", vector, ttl=86400)  # 24 hours

# No expiry (use default)
cache.put("static_data", data)  # Uses default_ttl
```

---

## Thread Safety Patterns

### Pattern 1: Single Lock (Used in Solution)

**Implementation** (`section2_cache_solution.py:114`)
```python
self.lock = threading.Lock()

def get(self, key: str) -> Optional[Any]:
    with self.lock:
        # All operations atomic
        ...
```

**Pros**:
- ✅ Simple and correct
- ✅ No deadlocks possible
- ✅ Easy to reason about

**Cons**:
- ❌ Readers block each other
- ❌ Single point of contention

**Best For**: Small-medium cache sizes, mixed read/write workloads

### Pattern 2: Read-Write Lock (Optimized Version)

**Implementation** (`section2_cache_solution.py:465-515`)
```python
class ReadWriteLock:
    """Allows multiple concurrent readers, exclusive writer"""

    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(threading.Lock())
        self._write_ready = threading.Condition(threading.Lock())

    def acquire_read(self):
        self._read_ready.acquire()
        try:
            while self._writers > 0:
                self._read_ready.wait()
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()
```

**Pros**:
- ✅ Multiple concurrent reads
- ✅ Better throughput for read-heavy workloads
- ✅ RAG systems are typically 80%+ reads

**Cons**:
- ❌ More complex
- ❌ Writer starvation possible
- ❌ Overhead for write-heavy workloads

**Performance Comparison**:
```
Read-Heavy Workload (90% reads, 10% writes):
- Single Lock:   ~50K ops/sec
- RW Lock:       ~200K ops/sec (4x improvement)

Write-Heavy Workload (50% reads, 50% writes):
- Single Lock:   ~40K ops/sec
- RW Lock:       ~35K ops/sec (worse due to overhead)
```

### Pattern 3: Lock-Free (Advanced)

**Not Implemented in Solution** (but mentioned for completeness)

**Approach**: Use atomic operations (CAS - Compare-And-Swap)
```python
import threading

class LockFreeCache:
    def __init__(self):
        self.cache = {}  # Use concurrent.futures or atomic dict
```

**Pros**:
- ✅ Maximum throughput
- ✅ No lock contention

**Cons**:
- ❌ Very complex to implement correctly
- ❌ ABA problem
- ❌ Python GIL limits benefits

**Verdict**: Stick with locks for Python caches

---

## Serialization & Complex Objects

### Why Serialization?

**Problem**: Cache stores bytes, not Python objects
- Memory efficiency
- Potential for persistence (save to disk)
- Network transport (distributed cache)

### Pickle Implementation

**Serialization** (`section2_cache_solution.py:320-344`)
```python
def serialize_value(self, value: Any) -> bytes:
    """Serialize complex objects for storage"""
    try:
        # Use highest protocol for performance
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error(f"Serialization error: {e}")
        raise
```

**Deserialization** (`section2_cache_solution.py:346-363`)
```python
def deserialize_value(self, data: bytes) -> Any:
    """Deserialize stored values"""
    try:
        return pickle.loads(data)
    except Exception as e:
        logger.error(f"Deserialization error: {e}")
        raise
```

### Supported Types

**Test Coverage** (`section2_cache_solution.py:692-710`)

1. **Numpy Arrays** (embeddings):
```python
embedding = np.array([0.1, 0.2, 0.3, 0.4])
cache.put("embedding", embedding)
retrieved = cache.get("embedding")
assert np.array_equal(retrieved, embedding)
```

2. **Dictionaries & Lists** (LLM responses):
```python
complex_data = {
    "results": [1, 2, 3, 4, 5],
    "metadata": {"source": "test", "score": 0.95}
}
cache.put("complex", complex_data)
retrieved = cache.get("complex")
assert retrieved == complex_data
```

3. **Custom Objects**:
```python
class RAGResult:
    def __init__(self, answer, sources, confidence):
        self.answer = answer
        self.sources = sources
        self.confidence = confidence

result = RAGResult("Answer", ["src1", "src2"], 0.9)
cache.put("rag_result", result)
```

### Serialization Alternatives

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **Pickle** | Python-native, handles anything | Security risk, Python-only | Internal cache |
| **JSON** | Language-agnostic, human-readable | Limited types, no numpy | API responses |
| **MessagePack** | Fast, compact | Requires library | High-performance cache |
| **Protocol Buffers** | Schema, versioning | Complex setup | Microservices |

**For RAG Systems**: Pickle is fine for internal cache, use JSON for API layer

---

## Performance Optimization

### Optimization 1: Cache Warming

**Problem**: Cold start (empty cache) → All misses initially

**Solution**: Pre-populate cache with frequent queries

**Implementation** (`section2_cache_solution.py:550-592`)
```python
class CacheWarmer:
    def warm_from_dict(self, data: Dict[str, Any], ttl: Optional[int] = None):
        for key, value in data.items():
            self.cache.put(key, value, ttl)

# Usage
warmer = CacheWarmer(cache)
frequent_queries = {
    "What is RAG?": precomputed_answer_1,
    "How does chunking work?": precomputed_answer_2
}
warmer.warm_from_dict(frequent_queries, ttl=86400)
```

### Optimization 2: Metrics Collection

**Track Performance** (`section2_cache_solution.py:598-654`)
```python
class CacheMetricsCollector:
    def __init__(self, cache: DistributedCache, collection_interval: int = 60):
        self.cache = cache
        self.interval = collection_interval
        self.history = []

    def start(self):
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()

    def _collect_loop(self):
        while self.running:
            stats = self.cache.get_stats()
            stats['timestamp'] = datetime.now().isoformat()
            self.history.append(stats)
            time.sleep(self.interval)
```

**Usage**:
```python
cache = DistributedCache(max_size=1000)
metrics = CacheMetricsCollector(cache, collection_interval=60)
metrics.start()

# ... run your application ...

# Analyze metrics
history = metrics.get_history()
avg_hit_rate = np.mean([h['hit_rate'] for h in history])
print(f"Average hit rate: {avg_hit_rate:.2%}")
```

### Optimization 3: Right-Sizing the Cache

**Too Small**: High eviction rate, low hit rate
**Too Large**: Wasted memory, stale entries

**Formula**:
```
Optimal Size ≈ (Requests per hour) × (Average TTL) / 3600

Example:
- 1000 req/hr
- 30 min TTL (1800s)
→ 1000 × 1800 / 3600 = 500 entries
→ Set max_size = 750 (50% buffer)
```

**Monitoring**:
```python
stats = cache.get_stats()

# Too small if:
if stats['evictions'] > stats['expirations']:
    print("⚠️ Cache too small, increase max_size")

# Too large if:
if stats['current_size'] < 0.5 * stats['max_size']:
    print("⚠️ Cache underutilized, decrease max_size")

# Good hit rate:
if stats['hit_rate'] < 0.3:
    print("⚠️ Low hit rate, check TTL or query patterns")
```

---

## Production Patterns

### Pattern 1: Graceful Degradation

**Never fail requests due to cache errors**

```python
def query_with_cache(query, doc_id):
    try:
        # Try cache first
        cached = cache.get(f"{query}:{doc_id}")
        if cached:
            return cached
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
        # Continue without cache

    # Generate answer
    answer = generate_answer(query, doc_id)

    try:
        # Cache result
        cache.put(f"{query}:{doc_id}", answer, ttl=1800)
    except Exception as e:
        logger.warning(f"Cache put failed: {e}")
        # Continue without caching

    return answer
```

### Pattern 2: Cache Key Design

**Good Keys**:
```python
# Include all relevant parameters
key = f"{query}:{doc_id}:v{version}:{model_name}"

# Hash long queries
import hashlib
query_hash = hashlib.md5(query.encode()).hexdigest()
key = f"{query_hash}:{doc_id}"

# Use structured keys
key = f"rag:{doc_id}:v{version}:q:{query_hash}"
```

**Bad Keys**:
```python
# Too generic (collision risk)
key = query  # What if same query for different docs?

# Too specific (low hit rate)
key = f"{query}:{timestamp}:{random_id}"  # Never hits
```

### Pattern 3: Multi-Level Caching

**L1 (In-Memory)** → L2 (Redis) → L3 (Database)

```python
class TieredCache:
    def __init__(self, l1_cache, redis_client):
        self.l1 = l1_cache  # DistributedCache
        self.l2 = redis_client

    def get(self, key):
        # Try L1 first (fastest)
        value = self.l1.get(key)
        if value:
            return value

        # Try L2 (Redis)
        value = self.l2.get(key)
        if value:
            # Promote to L1
            self.l1.put(key, value, ttl=300)
            return value

        return None

    def put(self, key, value, ttl):
        # Write to both levels
        self.l1.put(key, value, ttl=min(ttl, 300))  # L1: 5 min max
        self.l2.setex(key, ttl, pickle.dumps(value))
```

---

## Common Pitfalls

### Pitfall 1: Not Handling Updates

❌ **Wrong**:
```python
# Duplicate entries created
cache.put("key", "value1")
cache.put("key", "value2")  # Creates 2 nodes!
```

✅ **Correct** (`section2_cache_solution.py:186-193`):
```python
# Check if key exists, update instead of creating new
if key in self.cache:
    node = self.cache[key]
    node.value = serialized_value
    node.expiry = expiry
    self._move_to_front(node)
    return
```

### Pitfall 2: Forgetting LRU Ordering on Access

❌ **Wrong**:
```python
def get(self, key):
    if key in self.cache:
        return self.cache[key].value  # Don't move to front!
```

✅ **Correct**:
```python
def get(self, key):
    if key in self.cache:
        node = self.cache[key]
        self._move_to_front(node)  # Update LRU order
        return node.value
```

### Pitfall 3: Memory Leaks in Linked List

❌ **Wrong**:
```python
def _remove_node(self, node):
    if node.prev:
        node.prev.next = node.next
    if node.next:
        node.next.prev = node.prev
    # Forgot to clear node.prev and node.next!
```

✅ **Correct** (`section2_cache_solution.py:268-270`):
```python
# Clear node links to prevent memory leaks
node.prev = None
node.next = None
```

### Pitfall 4: Race Conditions

❌ **Wrong**:
```python
# Check and update not atomic
if key not in self.cache:
    self.cache[key] = new_node  # Race condition!
```

✅ **Correct**:
```python
# Atomic operation with lock
with self.lock:
    if key not in self.cache:
        self.cache[key] = new_node
```

---

## Code Templates

### Template 1: Basic LRU Cache

```python
class SimpleLRUCache:
    def __init__(self, max_size):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end (most recent)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            # Remove first item (least recent)
            self.cache.popitem(last=False)
```

**Note**: OrderedDict is simpler but less efficient than doubly-linked list for large caches.

### Template 2: Cache Decorator for Functions

```python
def lru_cache_wrapper(max_size=128, ttl=3600):
    cache = DistributedCache(max_size=max_size, default_ttl=ttl)

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from args
            key = f"{func.__name__}:{args}:{kwargs}"

            # Check cache
            cached = cache.get(key)
            if cached is not None:
                return cached

            # Call function
            result = func(*args, **kwargs)

            # Cache result
            cache.put(key, result)

            return result
        return wrapper
    return decorator

# Usage
@lru_cache_wrapper(max_size=100, ttl=1800)
def expensive_rag_query(query, doc_id):
    return generate_answer(query, doc_id)
```

### Template 3: Cache-Aside Pattern

```python
class CacheAsideService:
    def __init__(self, cache, database):
        self.cache = cache
        self.db = database

    def get_document(self, doc_id):
        # 1. Try cache
        cached = self.cache.get(f"doc:{doc_id}")
        if cached:
            return cached

        # 2. Cache miss - query database
        document = self.db.query(doc_id)

        # 3. Store in cache for future
        if document:
            self.cache.put(f"doc:{doc_id}", document, ttl=3600)

        return document

    def update_document(self, doc_id, new_data):
        # 1. Update database
        self.db.update(doc_id, new_data)

        # 2. Invalidate cache
        self.cache.remove(f"doc:{doc_id}")
```

---

## Quick Reference: section2_cache_solution.py

### Key Methods & Complexity

| Method | Time | Space | Description |
|--------|------|-------|-------------|
| `get(key)` | O(1) | O(1) | Retrieve value, update LRU |
| `put(key, value)` | O(1) amortized | O(1) | Store value, evict if full |
| `_move_to_front(node)` | O(1) | O(1) | Update LRU ordering |
| `_remove_node(node)` | O(1) | O(1) | Remove from linked list |
| `_evict_lru()` | O(1) | O(1) | Remove tail node |
| `cleanup_expired()` | O(n) | O(n) | Remove all expired entries |
| `serialize_value(v)` | O(k) | O(k) | Pickle object (k = size) |

### Test Coverage

| Test | Line | What It Tests |
|------|------|---------------|
| Test 1 | 667-672 | Basic put/get |
| Test 2 | 674-681 | LRU eviction |
| Test 3 | 683-690 | TTL expiration |
| Test 4 | 692-699 | Numpy arrays |
| Test 5 | 701-710 | Dicts/lists |
| Test 6 | 712-734 | Thread safety (5 threads × 100 ops) |
| Test 7 | 736-746 | Statistics tracking |
| Test 8 | 748-758 | **LRU ordering (critical!)** |
| Test 9 | 760-767 | Update existing key |
| Test 10 | 769-776 | Edge case (single-item cache) |

---

## Interview Tips

### When Asked About Caching:

1. **Start with Why**: Explain performance benefits (latency reduction)
2. **Eviction Policy**: Explain LRU and why it's good for RAG
3. **Data Structures**: HashMap + doubly-linked list for O(1)
4. **Thread Safety**: Mention locks and read-heavy optimization
5. **TTL**: Explain why AI systems need TTL

### Common Questions:

**Q: Why LRU instead of LFU (Least Frequently Used)?**
A: LRU is simpler (O(1) vs O(log n)) and better for recency-biased workloads like RAG. LFU better for stable frequency patterns.

**Q: How would you make this cache distributed?**
A: Replace in-memory storage with Redis/Memcached. Keep TTL, add consistent hashing for sharding, handle network failures.

**Q: What if you need to invalidate cache on document update?**
A: 1) Include version in key, 2) Active invalidation (delete on update), 3) Short TTL

**Q: How do you prevent cache stampede?**
A: Use locks or "request coalescing" - first request fetches, others wait for result.

---

**Ready to Practice?**
Check out `/tracks/03_algorithms_cache/` for progressive challenges:
- Beginner: Basic LRU (no TTL, single-threaded)
- Intermediate: LRU with TTL and thread safety
- Advanced: Distributed cache with Redis integration

Good luck! 🚀
