# Medium: LRU Cache for LLM API Responses

## Problem Statement

You are building a caching system for an LLM API integration layer. To reduce costs and latency, you want to cache recent API responses. Implement an LRU (Least Recently Used) cache that stores prompt-response pairs.

The cache should:
- Have a maximum capacity
- Store key-value pairs (prompt -> response)
- When capacity is reached, evict the least recently used item
- Update the "recently used" status when a key is accessed (get) or updated (put)

Implement the `LLMResponseCache` class:

- `LLMResponseCache(capacity: int)`: Initialize the cache with a positive capacity
- `get(prompt: str) -> str | None`: Return the cached response for the prompt if it exists, otherwise return None
- `put(prompt: str, response: str) -> None`: Add or update the prompt-response pair. If the cache is at capacity, remove the least recently used item before inserting the new one.

## Function Signature

```python
class LLMResponseCache:
    def __init__(self, capacity: int):
        pass

    def get(self, prompt: str) -> str | None:
        pass

    def put(self, prompt: str, response: str) -> None:
        pass
```

## Examples

### Example 1:
```python
cache = LLMResponseCache(2)

cache.put("What is AI?", "AI stands for Artificial Intelligence...")
cache.put("Define ML", "ML is Machine Learning...")

print(cache.get("What is AI?"))  # Output: "AI stands for Artificial Intelligence..."

cache.put("What is NLP?", "NLP is Natural Language Processing...")
# Cache is full, "Define ML" was least recently used and gets evicted

print(cache.get("Define ML"))  # Output: None (evicted)
print(cache.get("What is NLP?"))  # Output: "NLP is Natural Language Processing..."
```

### Example 2:
```python
cache = LLMResponseCache(3)

cache.put("prompt1", "response1")
cache.put("prompt2", "response2")
cache.put("prompt3", "response3")

cache.get("prompt1")  # Access prompt1, making it recently used

cache.put("prompt4", "response4")
# prompt2 is now LRU and gets evicted

print(cache.get("prompt2"))  # Output: None
print(cache.get("prompt1"))  # Output: "response1"
print(cache.get("prompt3"))  # Output: "response3"
print(cache.get("prompt4"))  # Output: "response4"
```

## Constraints

- `1 <= capacity <= 1000`
- `1 <= len(prompt) <= 500`
- `1 <= len(response) <= 5000`
- At most `10^4` calls will be made to `get` and `put`

## Solution

### Approach 1: Hash Map + Doubly Linked List

```python
class Node:
    """Doubly linked list node to maintain LRU order."""
    def __init__(self, key: str = "", value: str = ""):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LLMResponseCache:
    """
    LRU Cache implementation using hash map and doubly linked list.

    Time Complexity:
        - get(): O(1)
        - put(): O(1)

    Space Complexity: O(capacity)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # prompt -> Node

        # Dummy head and tail for easier list manipulation
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node: Node) -> None:
        """Add a node right after the head (most recently used position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_to_head(self, node: Node) -> None:
        """Move an existing node to the head (mark as recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_tail(self) -> Node:
        """Remove and return the least recently used node (just before tail)."""
        lru_node = self.tail.prev
        self._remove_node(lru_node)
        return lru_node

    def get(self, prompt: str) -> str | None:
        """
        Retrieve cached response for a prompt.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if prompt not in self.cache:
            return None

        node = self.cache[prompt]
        self._move_to_head(node)  # Mark as recently used
        return node.value

    def put(self, prompt: str, response: str) -> None:
        """
        Add or update a prompt-response pair in the cache.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if prompt in self.cache:
            # Update existing entry
            node = self.cache[prompt]
            node.value = response
            self._move_to_head(node)
        else:
            # Add new entry
            new_node = Node(prompt, response)
            self.cache[prompt] = new_node
            self._add_to_head(new_node)

            if len(self.cache) > self.capacity:
                # Evict LRU item
                lru_node = self._remove_tail()
                del self.cache[lru_node.key]
```

### Approach 2: OrderedDict (Python-specific optimization)

```python
from collections import OrderedDict


class LLMResponseCache:
    """
    LRU Cache using Python's OrderedDict.

    Time Complexity:
        - get(): O(1)
        - put(): O(1)

    Space Complexity: O(capacity)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, prompt: str) -> str | None:
        """Retrieve cached response, mark as recently used."""
        if prompt not in self.cache:
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(prompt)
        return self.cache[prompt]

    def put(self, prompt: str, response: str) -> None:
        """Add or update prompt-response pair."""
        if prompt in self.cache:
            # Update and move to end
            self.cache.move_to_end(prompt)

        self.cache[prompt] = response

        if len(self.cache) > self.capacity:
            # Remove first item (least recently used)
            self.cache.popitem(last=False)
```

## Detailed Walkthrough: Understanding `_add_to_head()`

The most confusing part of the LRU cache is understanding how the doubly linked list manipulation works. Let's break it down step by step, starting from the very beginning.

### Initial Setup: Just Head and Tail

When we create a new cache, we start with just two dummy nodes:

```python
self.head = Node()
self.tail = Node()
self.head.next = self.tail
self.tail.prev = self.head
```

**Visual representation:**
```
head ←→ tail

Detailed view:
head.prev = None
head.next = tail
tail.prev = head
tail.next = None
```

**Why dummy nodes?** They prevent us from having to handle special cases when the list is empty.

### Adding the FIRST Node (Node A)

Let's add our first real node with key "prompt1":

```python
node_A = Node("prompt1", "response1")
self._add_to_head(node_A)
```

Now let's trace through `_add_to_head()` line by line:

```python
def _add_to_head(self, node: Node) -> None:
    node.prev = self.head           # Line 1
    node.next = self.head.next      # Line 2
    self.head.next.prev = node      # Line 3 ⚠️
    self.head.next = node           # Line 4 ⚠️
```

#### Before any changes:
```
head ←→ tail

A is floating, not connected:
A.prev = None
A.next = None
```

#### After Line 1: `node.prev = self.head`
```
head ←→ tail
 ↑
 A.prev points here

A is partially connected:
A.prev = head
A.next = None
```

#### After Line 2: `node.next = self.head.next`
```
head ←→ tail
 ↑      ↑
 A.prev A.next points here

A knows where to insert:
A.prev = head
A.next = tail
```

#### After Line 3: `self.head.next.prev = node` ⚠️

Let's break this down:
- `self.head.next` = tail (because head.next currently points to tail)
- `self.head.next.prev` = tail.prev
- `= node` = make tail.prev point to A

```
head     tail
 ↑      ↗ ↑
 └─ A ─┘  │
          └── tail.prev now points to A

Now tail knows A is before it:
tail.prev = A
```

#### After Line 4: `self.head.next = node` ⚠️

Finally, connect head to A:
- `self.head.next = node` means head.next now points to A

```
head ←→ A ←→ tail

Complete! All connections established:
head.next = A
A.prev = head
A.next = tail
tail.prev = A
```

### Adding the SECOND Node (Node B)

Now let's add another node. Current state:

```
head ←→ A ←→ tail
```

We want to add B (and B should become the most recent):

```python
node_B = Node("prompt2", "response2")
self._add_to_head(node_B)
```

#### Before any changes:
```
head ←→ A ←→ tail

B is floating:
B.prev = None
B.next = None
```

#### After Line 1: `node.prev = self.head`
```
head ←→ A ←→ tail
 ↑
 B.prev points here
```

#### After Line 2: `node.next = self.head.next`
```
head ←→ A ←→ tail
 ↑      ↑
 B.prev B.next points here

B knows where to insert:
B.prev = head
B.next = A (because head.next is currently A)
```

#### After Line 3: `self.head.next.prev = node` ⚠️

Critical line! Let's decode:
- `self.head.next` = A (head's next is currently pointing to A)
- `self.head.next.prev` = A.prev
- `= node` = make A.prev point to B

```
head     A ←→ tail
 ↑      ↗ ↑
 └─ B ─┘

Now A knows B is before it:
A.prev = B (changed from head to B!)
```

#### After Line 4: `self.head.next = node` ⚠️

Update head's next pointer:
- `self.head.next = node` means head.next now points to B

```
head ←→ B ←→ A ←→ tail
        ↑
    Most recent!

Complete connections:
head.next = B
B.prev = head
B.next = A
A.prev = B
A.next = tail
tail.prev = A
```

### Why Order Matters: The Wrong Way

What if we swapped Line 3 and Line 4?

```python
def _add_to_head_WRONG(self, node: Node) -> None:
    node.prev = self.head           # Line 1: OK
    node.next = self.head.next      # Line 2: OK
    self.head.next = node           # Line 4 FIRST ❌
    self.head.next.prev = node      # Line 3 SECOND ❌
```

Let's trace adding B with this wrong order:

**After Lines 1 & 2:** Same as before
```
head ←→ A ←→ tail
 ↑      ↑
 B.prev B.next
```

**After Line 4 (executed early):** `self.head.next = node`
```
head ←→ B     A ←→ tail
        ↑
Now head.next = B (not A anymore!)
```

**After Line 3 (executed late):** `self.head.next.prev = node`
- `self.head.next` = B (we just changed it!)
- `self.head.next.prev` = B.prev
- `= node` = B.prev = B

**DISASTER:**
```
head → B ← B     A ←→ tail
       ↻
B points to itself!
A is disconnected!
```

This creates a broken list where:
- B.prev points to B (circular reference to itself)
- A is lost (nothing points to it anymore)

### Summary: The Key Insight

**The confusing lines work because we use the OLD value of `self.head.next` before changing it:**

1. **Line 3** uses `self.head.next` (which is the old first node) to connect it back to the new node
2. **Line 4** changes `self.head.next` to point to the new node

**If we reversed them:**
1. **Line 4** would change `self.head.next` to the new node first
2. **Line 3** would then use `self.head.next` (which is now the new node, not the old first node) and create a self-reference

**The correct order must be:**
1. Prepare the new node's pointers (Lines 1-2)
2. Update the old neighbors to point to the new node (Line 3) ← Uses old value
3. Update head to point to the new node (Line 4) ← Changes the value

## Test Cases

```python
def test_llm_cache():
    # Test 1: Basic functionality
    cache = LLMResponseCache(2)
    cache.put("What is AI?", "AI stands for Artificial Intelligence")
    cache.put("Define ML", "ML is Machine Learning")

    assert cache.get("What is AI?") == "AI stands for Artificial Intelligence"
    assert cache.get("Define ML") == "ML is Machine Learning"

    # Test 2: LRU eviction
    cache.put("What is NLP?", "NLP is Natural Language Processing")
    assert cache.get("What is AI?") == "AI stands for Artificial Intelligence"
    assert cache.get("Define ML") is None  # Evicted

    # Test 3: Update existing key
    cache = LLMResponseCache(2)
    cache.put("prompt1", "response1")
    cache.put("prompt1", "updated_response1")
    assert cache.get("prompt1") == "updated_response1"

    # Test 4: Access order matters
    cache = LLMResponseCache(3)
    cache.put("p1", "r1")
    cache.put("p2", "r2")
    cache.put("p3", "r3")

    cache.get("p1")  # p1 becomes most recent

    cache.put("p4", "r4")  # p2 should be evicted (LRU)

    assert cache.get("p2") is None
    assert cache.get("p1") == "r1"
    assert cache.get("p3") == "r3"
    assert cache.get("p4") == "r4"

    # Test 5: Single capacity
    cache = LLMResponseCache(1)
    cache.put("only", "one")
    assert cache.get("only") == "one"

    cache.put("new", "item")
    assert cache.get("only") is None
    assert cache.get("new") == "item"

    # Test 6: Update doesn't cause eviction
    cache = LLMResponseCache(2)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k1", "v1_updated")  # Update, not new key

    assert cache.get("k1") == "v1_updated"
    assert cache.get("k2") == "v2"

    print("All tests passed!")


if __name__ == "__main__":
    test_llm_cache()
```

## Complexity Analysis

### Approach 1: Hash Map + Doubly Linked List

**Time Complexity:**
- `get()`: O(1)
  - Hash map lookup: O(1)
  - List operations (remove, add): O(1)
- `put()`: O(1)
  - Hash map operations: O(1)
  - List operations: O(1)

**Space Complexity:** O(capacity)
- Hash map stores up to `capacity` entries
- Doubly linked list stores up to `capacity` nodes
- Overall: O(capacity)

### Approach 2: OrderedDict

**Time Complexity:**
- `get()`: O(1) amortized
- `put()`: O(1) amortized

**Space Complexity:** O(capacity)

## Key Hash Map Concepts Used

1. **Hash Map for O(1) Lookup**: Direct access to cached responses by prompt
2. **Combined Data Structures**: Hash map + linked list for efficient LRU tracking
3. **Key-Value Storage**: Natural fit for caching prompt-response pairs
4. **Eviction Policy**: Hash map enables quick removal when capacity is reached

## AI/Backend Context

This problem directly relates to the AI Engineer role:

1. **Cost Optimization**: LLM API calls are expensive (OpenAI, Anthropic, etc.). Caching identical prompts can save significant costs.

2. **Latency Reduction**: Cached responses return instantly vs. waiting for API calls (often 1-5 seconds).

3. **Real-world Application**:
   - In production RAG systems, similar queries often repeat
   - User sessions may ask follow-up questions
   - Common questions can be served from cache

4. **LRU Choice**: Recent prompts are more likely to be repeated than old ones, making LRU a good eviction policy.

5. **Production Considerations**:
   - In practice, you'd use Redis or Memcached
   - Need cache invalidation strategies
   - Consider prompt normalization (whitespace, casing)
   - May want time-based expiration (TTL) in addition to LRU

## Follow-up Questions

1. How would you add time-based expiration (TTL) to the cache?
2. What if you need to cache not just by exact prompt match, but by semantic similarity?
3. How would you implement a distributed version of this cache across multiple servers?
4. How would you handle cache warming (pre-loading common queries)?
5. What metrics would you track to evaluate cache effectiveness?
6. How would you modify this for a multi-tenant system where different users have different cache partitions?

## Extension: Cache with Statistics

```python
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LLMResponseCacheWithStats:
    """Enhanced LRU cache with performance metrics."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.stats = CacheStats()

    def get(self, prompt: str) -> Optional[str]:
        if prompt not in self.cache:
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        self.cache.move_to_end(prompt)
        return self.cache[prompt]

    def put(self, prompt: str, response: str) -> None:
        if prompt in self.cache:
            self.cache.move_to_end(prompt)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
                self.stats.evictions += 1

        self.cache[prompt] = response

    def get_stats(self) -> CacheStats:
        return self.stats
```
