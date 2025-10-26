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


cache = LLMResponseCache(2)

cache.put("What is AI?", "AI stands for Artificial Intelligence...")
cache.put("Define ML", "ML is Machine Learning...")

print(cache.get("What is AI?"))  # Output: "AI stands for Artificial Intelligence..."

cache.put("What is NLP?", "NLP is Natural Language Processing...")
# Cache is full, "Define ML" was least recently used and gets evicted

print(cache.get("Define ML"))  # Output: None (evicted)
print(cache.get("What is NLP?"))  # Output: "NLP is Natural Language Processing..."