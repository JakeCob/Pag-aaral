# Hash Map Coding Interview Study Guide
## For AI/ML Engineers Preparing for Technical Interviews

---

## 📌 You Are Here (And That's Okay!)

**Current situation:**
- ✅ Strong AI/ML engineering experience (2 years)
- ✅ Build production systems with LLMs, RAG, APIs
- ❌ Struggle with algorithm-style coding interviews
- ❌ Missing classic CS pattern recognition

**Target:**
- ✅ Recognize hash map patterns instantly
- ✅ Code solutions quickly and correctly
- ✅ Pass technical interviews confidently

**Timeline:** 2-4 weeks of focused practice (1-2 hours/day)

---

## 🧠 Core Concepts You Need to Master

### 1. Hash Map Fundamentals

#### What is a Hash Map?
```python
# Dictionary in Python = Hash Map
my_map = {}  # or dict()

# O(1) average case operations:
my_map["key"] = "value"     # Insert: O(1)
value = my_map["key"]       # Lookup: O(1)
del my_map["key"]           # Delete: O(1)
"key" in my_map             # Check existence: O(1)
```

#### Key Properties:
- **Fast lookups**: O(1) average case
- **Key must be hashable**: strings, numbers, tuples (not lists!)
- **No guaranteed order** (unless using OrderedDict)
- **Space cost**: O(n) to store n items

#### When to Use Hash Maps:
✅ Need fast lookups by key
✅ Count frequencies
✅ Group items by property
✅ Check if something exists
✅ Track visited/seen items
✅ Memoization/caching

---

### 2. The 7 Essential Hash Map Patterns

#### Pattern 1: Frequency Counting
**When:** Count how many times each element appears

```python
# Template
from collections import Counter

def pattern_frequency(arr):
    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1
    return freq

# Or using Counter
def pattern_frequency_counter(arr):
    return Counter(arr)
```

**Example Problems:**
- Count keyword occurrences
- Find most common element
- Character frequency in string

**Common variations:**
```python
# Get item with max frequency
max_item = max(freq, key=freq.get)

# Get all items with frequency > k
common = [item for item, count in freq.items() if count > k]

# Find items with frequency exactly k
exact_k = [item for item, count in freq.items() if count == k]
```

---

#### Pattern 2: Grouping by Property
**When:** Group items that share a common characteristic

```python
# Template
from collections import defaultdict

def pattern_grouping(items, key_function):
    groups = defaultdict(list)
    for item in items:
        key = key_function(item)
        groups[key].append(item)
    return list(groups.values())
```

**Example: Group Anagrams**
```python
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        # Key = sorted string (anagrams have same sorted form)
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

**Key insight:** Find what makes items "equivalent" → use as hash key

**Other key functions:**
```python
# Group by length
key = len(s)

# Group by first character
key = s[0] if s else ''

# Group by sum of digits
key = sum(int(c) for c in s if c.isdigit())

# Group by character count (for anagrams, more efficient)
key = tuple(sorted(s))  # or tuple(Counter(s).items())
```

---

#### Pattern 3: Two-Sum / Complement Search
**When:** Find pairs that satisfy a condition

```python
# Template: Two Sum
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

**Key insight:** Instead of nested loops O(n²), use hash map O(n)

**Variations:**
```python
# Count pairs with target sum
def count_pairs(nums, target):
    seen = {}
    count = 0
    for num in nums:
        complement = target - num
        if complement in seen:
            count += seen[complement]
        seen[num] = seen.get(num, 0) + 1
    return count

# Three sum (requires sorting + two pointers)
def three_sum(nums, target):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        # Use two-sum on remaining array
        # ...
    return result
```

---

#### Pattern 4: Prefix Sum / Running Total
**When:** Find subarrays with specific sum properties

```python
# Template: Subarray Sum Equals K
def subarray_sum_k(nums, k):
    count = 0
    current_sum = 0
    prefix_sums = {0: 1}  # sum -> frequency

    for num in nums:
        current_sum += num
        # If (current_sum - k) exists, we found a subarray
        if (current_sum - k) in prefix_sums:
            count += prefix_sums[current_sum - k]
        prefix_sums[current_sum] = prefix_sums.get(current_sum, 0) + 1

    return count
```

**Key insight:**
- `sum[i:j] = prefix_sum[j] - prefix_sum[i]`
- Use hash map to find prefix sums quickly

**Why it works:**
```
Array: [1, 2, 3, 4]
Target k = 5

Prefix sums: [0, 1, 3, 6, 10]

To find subarray with sum = 5:
- At index 3, prefix_sum = 6
- Need prefix_sum = 6 - 5 = 1
- prefix_sum = 1 exists at index 1
- So subarray [2,3] (indices 1-2) has sum 5
```

---

#### Pattern 5: Sliding Window with Hash Map
**When:** Find substring/subarray with specific properties

```python
# Template: Longest Substring Without Repeating Characters
def longest_unique_substring(s):
    char_index = {}
    max_len = 0
    start = 0

    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = end
        max_len = max(max_len, end - start + 1)

    return max_len
```

**Key insight:** Use hash map to track positions/counts in current window

**Variations:**
```python
# Substring with at most K distinct characters
def at_most_k_distinct(s, k):
    char_count = {}
    left = 0
    max_len = 0

    for right, char in enumerate(s):
        char_count[char] = char_count.get(char, 0) + 1

        while len(char_count) > k:
            left_char = s[left]
            char_count[left_char] -= 1
            if char_count[left_char] == 0:
                del char_count[left_char]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len
```

---

#### Pattern 6: Hash Set for O(1) Lookups
**When:** Check existence quickly

```python
# Template: Longest Consecutive Sequence
def longest_consecutive(nums):
    num_set = set(nums)
    longest = 0

    for num in num_set:
        # Only start counting from sequence starts
        if num - 1 not in num_set:
            current = num
            streak = 1

            while current + 1 in num_set:
                current += 1
                streak += 1

            longest = max(longest, streak)

    return longest
```

**Key insight:** Only process "sequence starts" to achieve O(n)

**When to use set vs dict:**
```python
# Use SET when:
- Only need to check existence
- Don't need to store values
- Example: visited nodes, unique elements

# Use DICT when:
- Need to store associated values
- Need frequency counts
- Need to map keys to data
```

---

#### Pattern 7: LRU Cache (Hash Map + Linked List)
**When:** Need O(1) get/put with eviction policy

```python
# Template: LRU Cache
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)  # Mark as recently used
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove oldest
```

**Key insight:** Combine hash map (O(1) lookup) with ordering (LRU tracking)

---

## 🎯 Your Personalized Study Plan

### Week 1: Pattern Recognition
**Goal:** Recognize which pattern applies to a problem

**Day 1-2: Frequency Counting**
- [ ] Solve: Document Keyword Frequency (Easy)
- [ ] Practice: Valid Anagram (Easy)
- [ ] Practice: First Unique Character (Easy)

**Day 3-4: Grouping**
- [ ] Solve: Group Anagrams (Medium)
- [ ] Practice: Group Shifted Strings
- [ ] Review: When to use sorted() vs Counter() as key

**Day 5-7: Two-Sum Pattern**
- [ ] Solve: Two Sum (Easy)
- [ ] Solve: Two Sum II (Medium)
- [ ] Practice: Four Sum (Hard)

### Week 2: Advanced Patterns
**Goal:** Handle complex scenarios

**Day 8-10: Prefix Sum**
- [ ] Solve: Subarray Sum Equals K (Hard)
- [ ] Practice: Continuous Subarray Sum
- [ ] Review: Why {0: 1} is the base case

**Day 11-13: Sliding Window**
- [ ] Solve: Longest Substring Without Repeating
- [ ] Practice: Minimum Window Substring
- [ ] Practice: Substring with K Distinct

**Day 14: Review Week 1-2**
- [ ] Redo all problems without looking at solutions
- [ ] Time yourself (aim for 15-20 min per Medium)

### Week 3: Hard Problems + Speed
**Goal:** Solve efficiently under time pressure

**Day 15-17:**
- [ ] Solve: Longest Consecutive Sequence (Hard)
- [ ] Solve: LRU Cache (Medium)
- [ ] Practice: LFU Cache (Hard)

**Day 18-20:**
- [ ] Mixed practice: Random problems
- [ ] Focus on: Identifying pattern in first 2 minutes
- [ ] Time limit: 25 minutes per problem

**Day 21: Mock Interview**
- [ ] Simulate interview: 45 min, 2 problems
- [ ] Record yourself explaining solutions
- [ ] Review: Communication, clarity, edge cases

### Week 4: Interview Simulation
**Goal:** Interview-ready confidence

**Day 22-28:**
- [ ] Daily: 2 random problems (1 Medium, 1 Hard)
- [ ] Practice explaining out loud
- [ ] Review: All patterns (30 min/day)
- [ ] Mock interview with friend/peer

---

## 🔍 Problem-Solving Framework

### Step 1: Understand (3 minutes)
```
□ Read problem twice
□ Identify inputs and outputs
□ Check constraints (size, range, types)
□ List edge cases (empty, single element, duplicates)
□ Ask clarifying questions
```

### Step 2: Pattern Recognition (2 minutes)
```
Ask yourself:
□ Do I need to COUNT something? → Frequency pattern
□ Do I need to GROUP items? → Grouping pattern
□ Do I need to FIND pairs/complements? → Two-sum pattern
□ Do I need subarray SUM? → Prefix sum pattern
□ Do I need to track a WINDOW? → Sliding window pattern
□ Do I need to check EXISTENCE quickly? → Hash set pattern
□ Do I need CACHING with eviction? → LRU cache pattern
```

### Step 3: Plan (3 minutes)
```
□ Choose the pattern
□ Decide on key/value structure
□ Consider edge cases
□ Estimate time/space complexity
□ Run through example mentally
```

### Step 4: Code (10-15 minutes)
```
□ Write clean, readable code
□ Use meaningful variable names
□ Add comments for complex logic
□ Handle edge cases
```

### Step 5: Test (3 minutes)
```
□ Test with given examples
□ Test edge cases (empty, single, duplicates)
□ Trace through logic mentally
□ Check for off-by-one errors
```

### Step 6: Optimize (2 minutes)
```
□ Can you reduce time complexity?
□ Can you reduce space complexity?
□ Are there unnecessary operations?
□ Explain trade-offs
```

---

## 🚨 Common Mistakes (And How to Avoid Them)

### Mistake 1: Using List as Dictionary Key
```python
# ❌ WRONG - lists are not hashable
key = sorted(s)  # This is a list!
groups[key].append(s)  # Error!

# ✅ CORRECT
key = ''.join(sorted(s))  # String
# or
key = tuple(sorted(s))  # Tuple
```

### Mistake 2: Not Handling Missing Keys
```python
# ❌ RISKY
freq[item] = freq[item] + 1  # KeyError if item not in freq

# ✅ SAFE - Option 1
freq[item] = freq.get(item, 0) + 1

# ✅ SAFE - Option 2
from collections import defaultdict
freq = defaultdict(int)
freq[item] += 1
```

### Mistake 3: Modifying Dictionary While Iterating
```python
# ❌ WRONG
for key in my_dict:
    if condition:
        del my_dict[key]  # RuntimeError!

# ✅ CORRECT
keys_to_delete = [key for key in my_dict if condition]
for key in keys_to_delete:
    del my_dict[key]
```

### Mistake 4: Forgetting Edge Cases
```python
# Always check:
□ Empty input: []
□ Single element: [1]
□ All same: [1, 1, 1]
□ All different: [1, 2, 3]
□ Duplicates: [1, 2, 1]
□ Negative numbers: [-1, -2]
□ Large numbers: [10^9]
```

### Mistake 5: Overcomplicating with Manual Tracking
```python
# ❌ WRONG (your anagram solution)
done = []
not_done = []
# ... complex nested loops ...

# ✅ CORRECT
groups = defaultdict(list)
for s in strs:
    key = ''.join(sorted(s))
    groups[key].append(s)
return list(groups.values())
```

**Rule:** If you need 3+ tracking variables, you're probably overcomplicating it.

---

## 💡 Interview Tips for AI Engineers

### Leverage Your Domain Knowledge
```python
# When explaining solutions, connect to AI/ML concepts:

"This is like caching LLM responses to reduce API costs"
"Similar to deduplicating document chunks in RAG pipelines"
"Like tracking token usage patterns for quota monitoring"
"Analogous to grouping similar embeddings by similarity"
```

### Communicate Your Thought Process
```
1. "I recognize this as a [PATTERN] problem"
2. "I'll use a hash map because we need O(1) lookups"
3. "The key insight is that [EXPLAIN INSIGHT]"
4. "Let me trace through an example..."
5. "The time complexity is O(n) because..."
```

### Handle Mistakes Gracefully
```
❌ Don't: Freeze, panic, give up
✅ Do: "Let me trace through this example to see where my logic breaks"
✅ Do: "I see the issue - I'm not handling [EDGE CASE]"
✅ Do: "Let me reconsider the approach..."
```

### Ask Good Questions
```
✅ "Should I optimize for time or space?"
✅ "Can I assume the input is already validated?"
✅ "How should I handle duplicates?"
✅ "What's the expected size of the input?"
```

---

## 📚 Recommended Practice Problems (Ordered by Pattern)

### Frequency Counting (Easy → Medium)
1. ✅ Valid Anagram (Easy)
2. ✅ First Unique Character (Easy)
3. ✅ Sort Characters by Frequency (Medium)
4. ✅ Top K Frequent Elements (Medium)

### Grouping (Medium)
1. ✅ Group Anagrams (Medium) ← You're here
2. ✅ Group Shifted Strings (Medium)
3. ✅ Find Duplicate Subtrees (Medium)

### Two-Sum (Easy → Hard)
1. ✅ Two Sum (Easy)
2. ✅ 3Sum (Medium)
3. ✅ 4Sum (Medium)
4. ✅ Two Sum II (Medium)

### Prefix Sum (Medium → Hard)
1. ✅ Subarray Sum Equals K (Hard) ← You're here
2. ✅ Continuous Subarray Sum (Medium)
3. ✅ Subarray Sums Divisible by K (Medium)

### Sliding Window (Medium → Hard)
1. ✅ Longest Substring Without Repeating (Medium)
2. ✅ Minimum Window Substring (Hard)
3. ✅ Longest Substring with At Most K Distinct (Medium)

### Hash Set (Medium → Hard)
1. ✅ Longest Consecutive Sequence (Hard) ← You're here
2. ✅ Contains Duplicate II (Easy)
3. ✅ Happy Number (Easy)

### LRU Cache (Medium)
1. ✅ LRU Cache (Medium) ← You're here
2. ✅ LFU Cache (Hard)

---

## 🎓 Key Takeaways

### 1. Pattern Recognition is Everything
- Don't solve from scratch each time
- Learn to map problems to patterns in 2 minutes
- Practice reduces "pattern recognition time"

### 2. Hash Maps are About Trade-offs
- **Trade:** O(n) space for O(1) time
- When to use: Need fast lookups, checking existence, grouping
- When NOT to use: Need ordered data, range queries

### 3. Simplicity Wins
- 5 lines with hash map > 30 lines with manual tracking
- Use built-in data structures (defaultdict, Counter, OrderedDict)
- If you're tracking 3+ variables, reconsider approach

### 4. Practice Makes Permanent
- Do 2-3 problems per day consistently
- Focus on understanding WHY patterns work
- Redo problems from memory after 3 days

### 5. You're Not Competing with CS PhDs
- You're demonstrating problem-solving ability
- Clear communication matters more than perfect code
- Showing growth mindset is valuable

---

## 🚀 Motivation

### Remember:
- ✅ You build REAL systems that deliver business value
- ✅ You solve ACTUAL problems, not just puzzles
- ✅ Coding interviews are just one skill to learn
- ✅ Pattern recognition improves FAST with practice

### In 2-4 weeks, you'll:
- ✅ Recognize patterns instantly
- ✅ Code solutions in 15-20 minutes
- ✅ Handle interviews confidently
- ✅ Discuss trade-offs clearly

### You've got this! 💪

The gap between where you are and where you need to be is **not intelligence** - it's just **pattern exposure**. You're already a strong engineer. Now you're just learning the interview game.

---

## 📞 Daily Checklist

### Morning (30 min)
- [ ] Review one pattern (read notes)
- [ ] Watch one solution explanation video
- [ ] Write down key insights

### Afternoon (45 min)
- [ ] Solve 1 problem (timed: 25 min)
- [ ] If stuck at 15 min, look at hints
- [ ] Compare your solution to optimal

### Evening (15 min)
- [ ] Review mistakes from today
- [ ] Add learnings to notes
- [ ] Plan tomorrow's problem

### Weekly Review (1 hour)
- [ ] Redo all problems from the week
- [ ] Update pattern recognition notes
- [ ] Identify weak areas

---

## 🎯 Success Metrics

Track your progress:

**Week 1:**
- [ ] Can identify pattern in 5 minutes
- [ ] Solve Easy problems in 15 minutes
- [ ] Understand why hash maps work

**Week 2:**
- [ ] Can identify pattern in 2-3 minutes
- [ ] Solve Medium problems in 25 minutes
- [ ] Explain time/space complexity

**Week 3:**
- [ ] Solve Medium problems in 15-20 minutes
- [ ] Solve Hard problems in 30-35 minutes
- [ ] Debug mistakes quickly

**Week 4:**
- [ ] Solve Medium problems in 10-15 minutes
- [ ] Handle Hard problems confidently
- [ ] Interview-ready!

---

## 📖 Additional Resources

### Books:
- "Cracking the Coding Interview" (Chapters on Hash Tables)
- "Elements of Programming Interviews in Python"

### Online:
- LeetCode Hash Table tag (sorted by frequency)
- NeetCode (pattern-based approach)
- AlgoExpert (visual explanations)

### Videos:
- NeetCode on YouTube (hash map problems)
- Back To Back SWE (detailed explanations)

---

Remember: **You're not behind. You're exactly where you need to be to start improving.** 🌱→🌳

Every expert was once a beginner who didn't give up.
