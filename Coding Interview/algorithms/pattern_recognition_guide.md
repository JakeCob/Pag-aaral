# Algorithm Pattern Recognition Guide

A comprehensive guide to identifying which algorithm or data structure to use based on problem keywords and patterns.

---

## Table of Contents
1. [Array & String Patterns](#array--string-patterns)
2. [Hash Map / Hash Table](#hash-map--hash-table)
3. [Sliding Window](#sliding-window)
4. [Two Pointers](#two-pointers)
5. [Binary Search](#binary-search)
6. [Stack & Queue](#stack--queue)
7. [Tree & Graph Patterns](#tree--graph-patterns)
8. [Dynamic Programming](#dynamic-programming)
9. [Greedy Algorithms](#greedy-algorithms)
10. [Backtracking](#backtracking)
11. [Heap / Priority Queue](#heap--priority-queue)
12. [Bit Manipulation](#bit-manipulation)

---

## Array & String Patterns

### Keywords to Look For:
- "subarray"
- "contiguous elements"
- "rotate array"
- "merge sorted arrays"
- "in-place modification"
- "prefix sum"

### Common Techniques:
- **Prefix Sum**: Running totals for range queries
- **Kadane's Algorithm**: Maximum subarray sum
- **In-place swapping**: Constant space manipulation

### Template Code:

**1. Prefix Sum**
```python
def prefix_sum(nums):
    prefix = [0] * (len(nums) + 1)
    for i in range(len(nums)):
        prefix[i + 1] = prefix[i] + nums[i]

    # Query sum from index i to j (inclusive)
    # sum = prefix[j + 1] - prefix[i]
    return prefix
```

**2. Kadane's Algorithm (Maximum Subarray Sum)**
```python
def max_subarray(nums):
    max_sum = float('-inf')
    current_sum = 0

    for num in nums:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum
```

**3. Array Rotation**
```python
def rotate_array(nums, k):
    k = k % len(nums)
    # Reverse entire array
    nums.reverse()
    # Reverse first k elements
    nums[:k] = reversed(nums[:k])
    # Reverse remaining elements
    nums[k:] = reversed(nums[k:])
```

### Example Problems:
- "Find maximum sum of contiguous subarray" → Kadane's Algorithm
- "Rotate array by k positions" → Reversal algorithm
- "Product of array except self" → Prefix/suffix products

---

## Hash Map / Hash Table

### Keywords to Look For:
- "count occurrences"
- "frequency"
- "duplicate"
- "unique"
- "O(1) lookup"
- "pair/triplet that sums to X"
- "anagram"
- "first unique character"
- "group by"

### When to Use:
- Need fast lookups (O(1))
- Tracking frequencies or counts
- Finding complements (e.g., two sum)
- Grouping related items
- Detecting duplicates

### Template Code:

**1. Two Sum Pattern**
```python
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

**2. Frequency Counter**
```python
from collections import Counter

def count_frequencies(arr):
    freq = Counter(arr)
    # or manually:
    # freq = {}
    # for item in arr:
    #     freq[item] = freq.get(item, 0) + 1
    return freq
```

**3. Group Anagrams**
```python
from collections import defaultdict

def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))  # or use tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

### Example Problems:
- "Two Sum" → Store complements in hash map
- "Group Anagrams" → Hash map with sorted string as key
- "First non-repeating character" → Frequency map
- "Longest substring without repeating characters" → Hash map + sliding window

---

## Sliding Window

### Keywords to Look For:
- "substring"
- "subarray"
- "contiguous"
- "window"
- "consecutive"
- "longest/shortest substring/subarray with..."
- "maximum/minimum of all subarrays of size k"
- "at most K distinct"

### Types of Sliding Windows:
1. **Fixed Size**: Window size is constant
   - "Maximum sum of subarray of size k"

2. **Variable Size**: Window expands/contracts
   - "Longest substring with at most K distinct characters"
   - "Minimum window substring"

### Template Code:

**1. Fixed Size Sliding Window**
```python
def max_sum_subarray(nums, k):
    if len(nums) < k:
        return 0

    # Calculate sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide the window
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

**2. Variable Size Sliding Window**
```python
def longest_substring_k_distinct(s, k):
    from collections import defaultdict

    char_count = defaultdict(int)
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Expand window: add right character
        char_count[s[right]] += 1

        # Shrink window if condition violated
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        # Update result
        max_length = max(max_length, right - left + 1)

    return max_length
```

**3. Minimum Window (Shrinking)**
```python
def min_window_substring(s, t):
    from collections import Counter

    if not s or not t:
        return ""

    target_count = Counter(t)
    required = len(target_count)
    formed = 0
    window_count = {}

    left = 0
    min_len = float('inf')
    min_window = (0, 0)

    for right in range(len(s)):
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1

        if char in target_count and window_count[char] == target_count[char]:
            formed += 1

        # Shrink window while valid
        while formed == required and left <= right:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_window = (left, right)

            char = s[left]
            window_count[char] -= 1
            if char in target_count and window_count[char] < target_count[char]:
                formed -= 1
            left += 1

    return "" if min_len == float('inf') else s[min_window[0]:min_window[1] + 1]
```

### Example Problems:
- "Maximum sum subarray of size k" → Fixed sliding window
- "Longest substring with K distinct characters" → Variable sliding window
- "Minimum window substring" → Variable window + hash map

---

## Two Pointers

### Keywords to Look For:
- "sorted array"
- "pair"
- "remove duplicates"
- "reverse"
- "palindrome"
- "container with most water"
- "trapping rain water"

### When to Use:
- Input is sorted (or can be sorted)
- Need to find pairs with certain properties
- Moving from both ends toward middle
- In-place array manipulation

### Patterns:
1. **Opposite Ends**: Start at both ends, move inward
2. **Same Direction**: Both pointers move left to right at different speeds
3. **Fast & Slow**: Different speeds (cycle detection)

### Template Code:

**1. Two Pointers - Opposite Ends (Two Sum II)**
```python
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1

    while left < right:
        current_sum = nums[left] + nums[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []
```

**2. Two Pointers - Same Direction (Remove Duplicates)**
```python
def remove_duplicates(nums):
    if not nums:
        return 0

    # Slow pointer tracks position for next unique element
    slow = 0

    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]

    return slow + 1
```

**3. Fast & Slow Pointers (Cycle Detection)**
```python
def has_cycle(head):
    if not head:
        return False

    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True

    return False
```

**4. Palindrome Check**
```python
def is_palindrome(s):
    left, right = 0, len(s) - 1

    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1

    return True
```

### Example Problems:
- "Two Sum II (sorted array)" → Two pointers from ends
- "Remove duplicates from sorted array" → Fast/slow pointers
- "Valid palindrome" → Two pointers from ends
- "Container with most water" → Two pointers from ends

---

## Binary Search

### Keywords to Look For:
- "sorted"
- "rotated sorted array"
- "search in O(log n)"
- "find first/last occurrence"
- "find peak element"
- "search in range [min, max]"
- "minimize/maximize" (search space)

### When to Use:
- Array is sorted
- Search space can be divided in half
- Finding threshold/boundary values
- "Search space binary search" for optimization problems

### Variations:
1. **Classic**: Find exact value
2. **Left boundary**: First occurrence
3. **Right boundary**: Last occurrence
4. **Search space**: Answer lies in range [min, max]

### Template Code:

**1. Classic Binary Search**
```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

**2. Left Boundary (First Occurrence)**
```python
def find_first(nums, target):
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

**3. Right Boundary (Last Occurrence)**
```python
def find_last(nums, target):
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

**4. Search Space Binary Search**
```python
def min_capacity(weights, days):
    def can_ship(capacity):
        current_weight = 0
        days_needed = 1

        for weight in weights:
            if current_weight + weight > capacity:
                days_needed += 1
                current_weight = weight
            else:
                current_weight += weight

        return days_needed <= days

    left = max(weights)  # Min possible capacity
    right = sum(weights)  # Max possible capacity

    while left < right:
        mid = left + (right - left) // 2

        if can_ship(mid):
            right = mid  # Try smaller capacity
        else:
            left = mid + 1

    return left
```

### Example Problems:
- "Search in rotated sorted array" → Modified binary search
- "Find first and last position" → Binary search boundaries
- "Minimum in rotated sorted array" → Binary search
- "Capacity to ship packages within D days" → Search space binary search

---

## Stack & Queue

### Stack Keywords:
- "nested"
- "parentheses"
- "valid brackets"
- "next greater/smaller element"
- "evaluate expression"
- "undo/redo"
- "backtrack"
- "depth-first"

### Queue Keywords:
- "breadth-first"
- "level-order"
- "first-in-first-out"
- "sliding window maximum"
- "recent calls/requests"

### When to Use Stack:
- Matching pairs (brackets, tags)
- Parsing expressions
- Monotonic problems (next greater/smaller)
- Undo operations
- DFS traversal

### When to Use Queue:
- BFS traversal
- Level-order processing
- Sliding window maximum (deque)
- Recently used cache

### Template Code:

**1. Valid Parentheses (Stack)**
```python
def is_valid_parentheses(s):
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}

    for char in s:
        if char in pairs:
            stack.append(char)
        elif not stack or pairs[stack.pop()] != char:
            return False

    return len(stack) == 0
```

**2. Monotonic Stack (Next Greater Element)**
```python
def next_greater_elements(nums):
    result = [-1] * len(nums)
    stack = []  # Store indices

    for i in range(len(nums)):
        # Pop elements smaller than current
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)

    return result
```

**3. BFS with Queue (Level Order Traversal)**
```python
from collections import deque

def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

**4. Monotonic Deque (Sliding Window Maximum)**
```python
from collections import deque

def max_sliding_window(nums, k):
    result = []
    dq = deque()  # Store indices

    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements (maintain decreasing order)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

### Example Problems:
- "Valid Parentheses" → Stack
- "Daily Temperatures" → Monotonic stack
- "Binary Tree Level Order Traversal" → Queue (BFS)
- "Sliding Window Maximum" → Monotonic deque

---

## Tree & Graph Patterns

### Tree Keywords:
- "binary tree"
- "binary search tree"
- "path from root to leaf"
- "level-order"
- "inorder/preorder/postorder"
- "ancestor"
- "depth/height"

### Graph Keywords:
- "connected components"
- "shortest path"
- "network"
- "islands"
- "relationships"
- "dependencies"
- "cycle detection"

### Tree Traversal Methods:
1. **DFS**: Preorder, Inorder, Postorder, Path finding
2. **BFS**: Level-order, Shortest path in unweighted tree

### Graph Algorithms:
1. **DFS**: Cycle detection, pathfinding, topological sort
2. **BFS**: Shortest path (unweighted), level-wise processing
3. **Dijkstra**: Shortest path (weighted)
4. **Union-Find**: Connected components, cycle detection

### Template Code:

**1. DFS - Recursive (Tree)**
```python
def dfs_recursive(root):
    if not root:
        return

    # Process current node (preorder)
    print(root.val)

    # Recurse on children
    dfs_recursive(root.left)
    dfs_recursive(root.right)
```

**2. DFS - Iterative (Graph)**
```python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()

        if node not in visited:
            visited.add(node)
            print(node)

            # Add unvisited neighbors
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return visited
```

**3. BFS - Graph/Tree**
```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])

    while queue:
        node = queue.popleft()
        print(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited
```

**4. DFS - Grid (Number of Islands)**
```python
def num_islands(grid):
    if not grid:
        return 0

    def dfs(i, j):
        if (i < 0 or i >= len(grid) or
            j < 0 or j >= len(grid[0]) or
            grid[i][j] != '1'):
            return

        grid[i][j] = '0'  # Mark as visited

        # Explore 4 directions
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1

    return count
```

**5. Topological Sort (Cycle Detection)**
```python
def can_finish(num_courses, prerequisites):
    from collections import defaultdict, deque

    # Build adjacency list and indegree count
    graph = defaultdict(list)
    indegree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegree[course] += 1

    # Start with courses that have no prerequisites
    queue = deque([i for i in range(num_courses) if indegree[i] == 0])
    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1

        for next_course in graph[course]:
            indegree[next_course] -= 1
            if indegree[next_course] == 0:
                queue.append(next_course)

    return completed == num_courses
```

**6. Union-Find (Disjoint Set)**
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)

        if px == py:
            return False

        # Union by rank
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1

        return True
```

### Example Problems:
- "Maximum Depth of Binary Tree" → DFS recursion
- "Lowest Common Ancestor" → Tree traversal
- "Number of Islands" → DFS/BFS grid traversal
- "Course Schedule" → Topological sort (cycle detection)

---

## Dynamic Programming

### Keywords to Look For:
- "maximum/minimum"
- "count number of ways"
- "longest"
- "shortest"
- "optimize"
- "can we achieve"
- "overlapping subproblems"
- "optimal substructure"

### Key Questions to Ask:
1. Can the problem be broken into smaller subproblems?
2. Do subproblems overlap?
3. Can you define a recurrence relation?
4. What are the base cases?

### Common DP Patterns:
1. **Linear DP**: 1D array (Fibonacci, House Robber)
2. **2D Grid DP**: 2D array (Unique Paths, Edit Distance)
3. **Knapsack**: Subset selection with constraints
4. **LIS**: Longest Increasing Subsequence variations
5. **String DP**: Palindrome, subsequence problems

### Template Code:

**1. 1D DP (Fibonacci/Climbing Stairs)**
```python
def climb_stairs(n):
    if n <= 2:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2

    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]

# Space optimized
def climb_stairs_optimized(n):
    if n <= 2:
        return n

    prev2, prev1 = 1, 2

    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1
```

**2. 2D DP (Unique Paths)**
```python
def unique_paths(m, n):
    dp = [[0] * n for _ in range(m)]

    # Initialize first row and column
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    return dp[m - 1][n - 1]
```

**3. Knapsack (0/1 Knapsack)**
```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],  # Don't take item
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]  # Take item
                )
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]
```

**4. Unbounded Knapsack (Coin Change)**
```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

**5. LCS (Longest Common Subsequence)**
```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
```

**6. Edit Distance**
```python
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )

    return dp[m][n]
```

### Example Problems:
- "Climbing Stairs" → 1D DP
- "Coin Change" → Unbounded knapsack
- "Longest Common Subsequence" → 2D DP
- "Word Break" → 1D DP with string
- "Edit Distance" → 2D DP

---

## Greedy Algorithms

### Keywords to Look For:
- "maximize/minimize"
- "optimal"
- "earliest/latest"
- "interval scheduling"
- "activity selection"
- "local optimal leads to global optimal"

### When to Use:
- Making locally optimal choices leads to global optimum
- No need to reconsider previous choices
- Often involves sorting first

### How to Identify:
- Greedy choice property exists
- Problem exhibits optimal substructure
- Counterexample doesn't exist for greedy approach

### Template Code:

**1. Activity Selection / Interval Scheduling**
```python
def max_activities(start, end):
    # Create list of (start, end) pairs
    activities = list(zip(start, end))

    # Sort by end time (greedy choice)
    activities.sort(key=lambda x: x[1])

    count = 1
    last_end = activities[0][1]

    for i in range(1, len(activities)):
        if activities[i][0] >= last_end:
            count += 1
            last_end = activities[i][1]

    return count
```

**2. Jump Game (Greedy)**
```python
def can_jump(nums):
    max_reach = 0

    for i in range(len(nums)):
        if i > max_reach:
            return False

        max_reach = max(max_reach, i + nums[i])

        if max_reach >= len(nums) - 1:
            return True

    return True
```

**3. Fractional Knapsack**
```python
def fractional_knapsack(weights, values, capacity):
    # Calculate value per weight
    items = [(values[i] / weights[i], weights[i], values[i])
             for i in range(len(weights))]

    # Sort by value per weight (descending)
    items.sort(reverse=True)

    total_value = 0
    remaining_capacity = capacity

    for value_per_weight, weight, value in items:
        if remaining_capacity >= weight:
            # Take entire item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take fraction of item
            total_value += value_per_weight * remaining_capacity
            break

    return total_value
```

**4. Huffman Coding / Merge Operations**
```python
import heapq

def min_cost_to_connect(sticks):
    heapq.heapify(sticks)
    total_cost = 0

    while len(sticks) > 1:
        # Take two smallest
        first = heapq.heappop(sticks)
        second = heapq.heappop(sticks)

        cost = first + second
        total_cost += cost

        # Add back combined stick
        heapq.heappush(sticks, cost)

    return total_cost
```

### Example Problems:
- "Activity Selection" → Sort by end time, greedy select
- "Jump Game" → Greedy track furthest reachable
- "Gas Station" → Greedy accumulation
- "Minimum Arrows to Burst Balloons" → Interval scheduling

---

## Backtracking

### Keywords to Look For:
- "all possible"
- "generate all"
- "combination"
- "permutation"
- "subset"
- "N-Queens"
- "Sudoku"
- "word search"
- "partition"

### When to Use:
- Need to explore ALL possibilities
- Build solution incrementally
- Abandon path when constraints violated (prune)

### Template Code:

**1. Permutations**
```python
def permute(nums):
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return

        for i in range(len(remaining)):
            # Make choice
            path.append(remaining[i])

            # Recurse with remaining elements
            backtrack(path, remaining[:i] + remaining[i+1:])

            # Undo choice
            path.pop()

    backtrack([], nums)
    return result
```

**2. Combinations**
```python
def combine(n, k):
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)  # Move to next number
            path.pop()

    backtrack(1, [])
    return result
```

**3. Subsets**
```python
def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])  # Add current subset

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

**4. N-Queens**
```python
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]

    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check diagonal (top-left)
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # Check diagonal (top-right)
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return

        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'

    backtrack(0)
    return result
```

**5. Word Search (Grid Backtracking)**
```python
def exist(board, word):
    def backtrack(i, j, index):
        if index == len(word):
            return True

        if (i < 0 or i >= len(board) or
            j < 0 or j >= len(board[0]) or
            board[i][j] != word[index]):
            return False

        # Mark as visited
        temp = board[i][j]
        board[i][j] = '#'

        # Explore 4 directions
        found = (backtrack(i+1, j, index+1) or
                backtrack(i-1, j, index+1) or
                backtrack(i, j+1, index+1) or
                backtrack(i, j-1, index+1))

        # Restore cell
        board[i][j] = temp

        return found

    for i in range(len(board)):
        for j in range(len(board[0])):
            if backtrack(i, j, 0):
                return True

    return False
```

### Example Problems:
- "Generate Parentheses" → Backtracking with constraints
- "Permutations" → Classic backtracking
- "N-Queens" → Backtracking with pruning
- "Word Search" → 2D grid backtracking

---

## Heap / Priority Queue

### Keywords to Look For:
- "kth largest/smallest"
- "top K"
- "median"
- "priority"
- "merge K sorted"
- "closest points"
- "sliding window median"

### When to Use:
- Need quick access to min/max element
- K-way merge operations
- Maintaining top K elements
- Dynamic median finding

### Common Patterns:
1. **Min Heap**: Find K largest (keep K largest in min heap)
2. **Max Heap**: Find K smallest (keep K smallest in max heap)
3. **Two Heaps**: Median finding (max heap + min heap)

### Template Code:

**1. Kth Largest Element**
```python
import heapq

def find_kth_largest(nums, k):
    # Use min heap of size k
    min_heap = []

    for num in nums:
        heapq.heappush(min_heap, num)

        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return min_heap[0]
```

**2. Find Median from Data Stream (Two Heaps)**
```python
import heapq

class MedianFinder:
    def __init__(self):
        # Max heap for lower half (negate values)
        self.small = []
        # Min heap for upper half
        self.large = []

    def addNum(self, num):
        # Add to max heap (small)
        heapq.heappush(self.small, -num)

        # Balance: move largest from small to large
        if self.small and self.large and (-self.small[0] > self.large[0]):
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)

        # Balance sizes
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)

    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0
```

**3. Merge K Sorted Lists**
```python
import heapq

def merge_k_lists(lists):
    min_heap = []
    result = []

    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst[0], i, 0))

    while min_heap:
        val, list_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)

        # Add next element from same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(min_heap, (next_val, list_idx, elem_idx + 1))

    return result
```

**4. Top K Frequent Elements**
```python
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    freq = Counter(nums)

    # Use min heap of size k
    min_heap = []

    for num, count in freq.items():
        heapq.heappush(min_heap, (count, num))

        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return [num for count, num in min_heap]
```

**5. K Closest Points to Origin**
```python
import heapq

def k_closest(points, k):
    # Use max heap to keep k closest points
    max_heap = []

    for x, y in points:
        dist = -(x*x + y*y)  # Negate for max heap

        if len(max_heap) < k:
            heapq.heappush(max_heap, (dist, [x, y]))
        elif dist > max_heap[0][0]:
            heapq.heapreplace(max_heap, (dist, [x, y]))

    return [point for dist, point in max_heap]
```

### Example Problems:
- "Kth Largest Element" → Min heap of size K
- "Find Median from Data Stream" → Two heaps
- "Merge K Sorted Lists" → Min heap
- "Top K Frequent Elements" → Heap with frequency

---

## Bit Manipulation

### Keywords to Look For:
- "binary representation"
- "XOR"
- "single number"
- "power of two"
- "count bits"
- "bitwise operations"
- "subset generation using bits"

### Common Techniques:
- `x & (x-1)`: Remove rightmost set bit
- `x & -x`: Isolate rightmost set bit
- `x ^ x`: Always 0
- `x ^ 0`: Always x
- `a ^ b ^ b = a`: XOR cancellation

### When to Use:
- Space optimization (bitmask instead of boolean array)
- Fast operations (checking power of 2)
- Finding unique elements
- Subset enumeration

### Template Code:

**1. Single Number (XOR)**
```python
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
```

**2. Number of 1 Bits (Hamming Weight)**
```python
def hamming_weight(n):
    count = 0
    while n:
        n &= (n - 1)  # Remove rightmost 1 bit
        count += 1
    return count

# Alternative
def hamming_weight_v2(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count
```

**3. Power of Two**
```python
def is_power_of_two(n):
    if n <= 0:
        return False
    return (n & (n - 1)) == 0
```

**4. Generate Subsets Using Bitmask**
```python
def subsets_bitmask(nums):
    n = len(nums)
    result = []

    # Iterate through all 2^n possibilities
    for mask in range(1 << n):  # 0 to 2^n - 1
        subset = []

        for i in range(n):
            # Check if i-th bit is set
            if mask & (1 << i):
                subset.append(nums[i])

        result.append(subset)

    return result
```

**5. Count Set Bits (Brian Kernighan's Algorithm)**
```python
def count_bits(n):
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        # i & (i - 1) removes rightmost 1
        # So count[i] = count[i & (i-1)] + 1
        dp[i] = dp[i & (i - 1)] + 1

    return dp
```

**6. Reverse Bits**
```python
def reverse_bits(n):
    result = 0

    for i in range(32):
        # Get the i-th bit from right
        bit = (n >> i) & 1
        # Set it at (31-i) position
        result |= (bit << (31 - i))

    return result
```

**7. Find Missing Number**
```python
def missing_number(nums):
    n = len(nums)
    result = n  # Start with n

    for i in range(n):
        result ^= i ^ nums[i]

    return result
```

### Example Problems:
- "Single Number" → XOR all elements
- "Number of 1 Bits" → Bit counting
- "Power of Two" → `n & (n-1) == 0`
- "Subsets" → Bitmask for subset generation

---

## Quick Decision Tree

```
START
  |
  ├─ Need fast lookups/counts? → HASH MAP
  |
  ├─ Contiguous subarray/substring? → SLIDING WINDOW
  |
  ├─ Sorted array + search? → BINARY SEARCH or TWO POINTERS
  |
  ├─ All possible solutions? → BACKTRACKING
  |
  ├─ Overlapping subproblems? → DYNAMIC PROGRAMMING
  |
  ├─ Tree/Graph traversal? → DFS/BFS
  |
  ├─ Kth largest/smallest? → HEAP
  |
  ├─ Matching pairs/nesting? → STACK
  |
  └─ Optimization with local choices? → GREEDY or DP
```

---

## Combination Patterns

Many hard problems combine multiple techniques:

| Problem Type | Primary | Secondary |
|-------------|---------|-----------|
| Minimum Window Substring | Sliding Window | Hash Map |
| Substring with Concatenation | Sliding Window | Hash Map |
| Longest Substring K Distinct | Sliding Window | Hash Map |
| 3Sum | Two Pointers | Sorting + Hash Set |
| Trapping Rain Water | Two Pointers | Stack (alternative) |
| Top K Frequent | Hash Map | Heap |
| Word Ladder | BFS (Graph) | Hash Set |
| Course Schedule | Graph (DFS) | Topological Sort |

---

## Practice Strategy

1. **Identify the pattern** from keywords
2. **Ask clarifying questions** about constraints
3. **Consider time/space complexity** requirements
4. **Start with brute force**, then optimize
5. **Practice recognizing combinations** of patterns

---

## Common Time Complexities by Algorithm

- **Hash Map Operations**: O(1) average
- **Sliding Window**: O(n)
- **Two Pointers**: O(n)
- **Binary Search**: O(log n)
- **DFS/BFS**: O(V + E) for graphs, O(n) for trees
- **Sorting**: O(n log n)
- **Dynamic Programming**: Often O(n²) or O(n × m)
- **Backtracking**: Often O(2ⁿ) or O(n!)
- **Heap Operations**: O(log n) insert/delete, O(1) peek

---

## Tips for Pattern Recognition

1. **Read the problem twice** - understand what's being asked
2. **Identify the input type** - array, string, tree, graph?
3. **Look for constraint clues** - sorted? distinct? size limits?
4. **Check the output** - single value? all solutions? optimization?
5. **Estimate complexity** - can you afford O(n²)? Need O(n)?
6. **Draw examples** - visualize small test cases
7. **Think about edge cases** - empty input? duplicates? negative values?

---

## Conclusion

Pattern recognition improves with practice. Start by solving classic problems for each pattern, then move to problems that combine multiple patterns. Over time, you'll develop intuition for recognizing which algorithm to apply based on the problem description alone.

**Key Takeaway**: The keywords and structure of the problem statement are your biggest clues. Train yourself to spot them quickly!
