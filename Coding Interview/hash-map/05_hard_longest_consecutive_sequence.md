# Hard: Longest Consecutive Sequence (with AI Context)

## Problem Statement

You are analyzing a stream of LLM API response times (in milliseconds) and want to find the longest consecutive sequence of response time values to identify performance patterns.

Given an **unsorted** array of integers, find the length of the longest consecutive elements sequence. The sequence must be contiguous in value (e.g., 1, 2, 3, 4), but the elements can appear in any order in the input array.

**Important:** Your algorithm must run in **O(n)** time complexity.

## Function Signature

```python
def longest_consecutive(nums: list[int]) -> int:
    """
    Args:
        nums: Unsorted array of integers (e.g., response times)

    Returns:
        Length of the longest consecutive sequence
    """
    pass
```

## Examples

### Example 1:
```python
nums = [100, 4, 200, 1, 3, 2]

result = longest_consecutive(nums)
print(result)  # Output: 4
```

**Explanation:**
- The longest consecutive sequence is `[1, 2, 3, 4]`
- Length = 4

### Example 2:
```python
nums = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]

result = longest_consecutive(nums)
print(result)  # Output: 9
```

**Explanation:**
- The longest consecutive sequence is `[0, 1, 2, 3, 4, 5, 6, 7, 8]`
- Length = 9

### Example 3:
```python
nums = [9, 1, 4, 7, 3, 2, 8, 5, 6]

result = longest_consecutive(nums)
print(result)  # Output: 9
```

**Explanation:**
- The longest consecutive sequence is `[1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Length = 9

### Example 4:
```python
nums = [1, 2, 0, 1]

result = longest_consecutive(nums)
print(result)  # Output: 3
```

**Explanation:**
- The longest consecutive sequence is `[0, 1, 2]`
- Duplicates don't affect the result
- Length = 3

### Example 5:
```python
nums = []

result = longest_consecutive(nums)
print(result)  # Output: 0
```

### Example 6:
```python
nums = [10]

result = longest_consecutive(nums)
print(result)  # Output: 1
```

## Constraints

- `0 <= nums.length <= 10^5`
- `-10^9 <= nums[i] <= 10^9`
- Array may contain duplicates
- **Must achieve O(n) time complexity**

## Solution

### ❌ Approach 0: Sorting (Does NOT meet O(n) requirement)

First, let's see the obvious solution that **doesn't meet the time complexity requirement**:

```python
def longest_consecutive_sorting(nums: list[int]) -> int:
    """
    Sorting approach - DOES NOT meet O(n) requirement!

    Time Complexity: O(n log n) - too slow!
    Space Complexity: O(1) or O(n) depending on sorting algorithm
    """
    if not nums:
        return 0

    nums.sort()
    longest = 1
    current_streak = 1

    for i in range(1, len(nums)):
        if nums[i] == nums[i-1]:
            # Skip duplicates
            continue
        elif nums[i] == nums[i-1] + 1:
            # Consecutive
            current_streak += 1
            longest = max(longest, current_streak)
        else:
            # Break in sequence
            current_streak = 1

    return longest
```

**Why this doesn't work:** Sorting is O(n log n), but we need O(n).

### ✅ Approach 1: Hash Set (Optimal Solution)

The key insight: **Use a hash set for O(1) lookups to build consecutive sequences**.

**Strategy:**
1. Put all numbers in a hash set for O(1) lookup
2. For each number, check if it's the **start** of a sequence (i.e., `num - 1` is not in the set)
3. If it's the start, count how long the sequence goes

**Why this works in O(n):**
- Each number is visited at most twice (once to check if it's a start, once when counting)
- Hash set lookups are O(1)

```python
def longest_consecutive(nums: list[int]) -> int:
    """
    Hash set approach - achieves O(n) time complexity.

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not nums:
        return 0

    num_set = set(nums)  # O(n) to create set
    longest_streak = 0

    for num in num_set:  # O(n) iterations
        # Check if this number is the start of a sequence
        # (i.e., num - 1 is not in the set)
        if num - 1 not in num_set:  # O(1) lookup
            # This is the start of a sequence
            current_num = num
            current_streak = 1

            # Count how long the sequence goes
            while current_num + 1 in num_set:  # O(1) lookup per iteration
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak)

    return longest_streak
```

**Detailed Trace for Example 1:**

Input: `nums = [100, 4, 200, 1, 3, 2]`

```
Step 1: Create hash set
num_set = {100, 4, 200, 1, 3, 2}

Step 2: Iterate through num_set

Check num = 100:
  Is 99 in num_set? No → This is a sequence start
  Sequence: 100
  Is 101 in num_set? No
  Streak = 1

Check num = 4:
  Is 3 in num_set? Yes → Skip (not a sequence start)

Check num = 200:
  Is 199 in num_set? No → This is a sequence start
  Sequence: 200
  Is 201 in num_set? No
  Streak = 1

Check num = 1:
  Is 0 in num_set? No → This is a sequence start
  Sequence: 1
  Is 2 in num_set? Yes → 1, 2
  Is 3 in num_set? Yes → 1, 2, 3
  Is 4 in num_set? Yes → 1, 2, 3, 4
  Is 5 in num_set? No
  Streak = 4 ← Longest!

Check num = 3:
  Is 2 in num_set? Yes → Skip (not a sequence start)

Check num = 2:
  Is 1 in num_set? Yes → Skip (not a sequence start)

Result: longest_streak = 4
```

**Why the while loop doesn't make it O(n²):**

At first glance, it looks like the while loop could make this O(n²):
- Outer loop: O(n)
- Inner while loop: Could be O(n) in worst case

BUT here's the key: **Each number is only checked in the while loop once across the entire algorithm**.

For the sequence `[1, 2, 3, 4]`:
- When we check `num = 1` (sequence start), we visit 2, 3, 4 in the while loop
- When we later check `num = 2, 3, 4`, we skip them because they're not sequence starts

So even though we iterate through the set (O(n)), each element is only counted once in total across all while loops. Total: O(n) + O(n) = O(n).

### Approach 2: Hash Map with Memoization (Alternative)

```python
def longest_consecutive_memo(nums: list[int]) -> int:
    """
    Hash map with memoization to store sequence lengths.

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not nums:
        return 0

    num_map = {}
    longest_streak = 0

    for num in nums:
        if num in num_map:
            continue  # Skip duplicates

        # Check neighbors
        left_length = num_map.get(num - 1, 0)
        right_length = num_map.get(num + 1, 0)

        # Total length including current number
        current_length = left_length + right_length + 1

        # Update the map
        num_map[num] = current_length

        # Update the boundaries of the sequence
        num_map[num - left_length] = current_length
        num_map[num + right_length] = current_length

        longest_streak = max(longest_streak, current_length)

    return longest_streak
```

**This approach is clever but harder to understand.** The hash set approach (Approach 1) is clearer and recommended for interviews.

## Test Cases

```python
def test_longest_consecutive():
    # Test 1: Basic example
    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4

    # Test 2: Larger sequence
    assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9

    # Test 3: All consecutive
    assert longest_consecutive([9, 1, 4, 7, 3, 2, 8, 5, 6]) == 9

    # Test 4: With duplicates
    assert longest_consecutive([1, 2, 0, 1]) == 3

    # Test 5: Empty array
    assert longest_consecutive([]) == 0

    # Test 6: Single element
    assert longest_consecutive([10]) == 1

    # Test 7: No consecutive
    assert longest_consecutive([1, 3, 5, 7, 9]) == 1

    # Test 8: Two separate sequences
    assert longest_consecutive([1, 2, 3, 10, 11, 12, 13]) == 4

    # Test 9: Negative numbers
    assert longest_consecutive([-1, -2, 0, 1, 2]) == 5

    # Test 10: All duplicates
    assert longest_consecutive([1, 1, 1, 1]) == 1

    # Test 11: Large gap
    assert longest_consecutive([1, 1000000000]) == 1

    print("All tests passed!")


if __name__ == "__main__":
    test_longest_consecutive()
```

## Complexity Analysis

### Approach 1: Hash Set (Optimal)

**Time Complexity: O(n)**
- Creating the hash set: O(n)
- Iterating through the set: O(n)
- While loop: Amortized O(1) per element
  - Each number is checked at most twice:
    1. Once when checking if it's a sequence start
    2. Once when it's part of a sequence (in the while loop)
  - Total across all iterations: O(n)
- Overall: O(n) + O(n) = O(n)

**Space Complexity: O(n)**
- Hash set stores at most n unique numbers

### Why the While Loop is O(n) Total, Not O(n) Per Iteration

This is the **key insight** that makes this problem hard:

```python
for num in num_set:              # Outer loop: n iterations
    if num - 1 not in num_set:   # Only start sequences here
        while current_num + 1 in num_set:  # This looks like O(n)!
            current_num += 1
```

**Amortized Analysis:**

Consider the sequence `[1, 2, 3, 4, 5]`:

- `num = 1`: Is a start (0 not in set)
  - While loop checks: 2, 3, 4, 5 (4 checks)

- `num = 2`: Not a start (1 is in set) → Skip
- `num = 3`: Not a start (2 is in set) → Skip
- `num = 4`: Not a start (3 is in set) → Skip
- `num = 5`: Not a start (4 is in set) → Skip

**Total work:** 1 + 4 = 5 checks for 5 numbers = O(n)

**Key point:** Only sequence starts enter the while loop. Other numbers are skipped immediately. Each number is visited in the while loop **at most once** across the entire algorithm.

## Key Hash Map Concepts Used

1. **Hash Set for O(1) Membership Testing**: Core to achieving O(n) time
2. **Clever Iteration Strategy**: Only process sequence starts to avoid redundant work
3. **Deduplication**: Set automatically handles duplicates
4. **Amortized Analysis**: Understanding why nested loops can still be O(n)

## Why This Problem is Hard

1. **O(n) Requirement**: The obvious sorting solution is O(n log n), disqualifying it
2. **Non-obvious Optimization**: The "only process sequence starts" insight is not intuitive
3. **Amortized Analysis**: Understanding why the while loop doesn't make it O(n²) requires deeper analysis
4. **Edge Cases**: Empty array, duplicates, negative numbers, single elements
5. **Space-Time Tradeoff**: Using O(n) extra space to achieve O(n) time

## AI/Backend Context

This problem has practical applications in AI systems:

### 1. Latency Analysis
```python
# Analyze API response times to find longest consecutive range
response_times = [120, 118, 122, 119, 121, 300, 301]
longest = longest_consecutive(response_times)
# Result: 5 (sequence 118, 119, 120, 121, 122)
# Indicates stable performance period
```

### 2. Token Usage Patterns
```python
# Find longest consecutive token usage for quota analysis
daily_tokens = [1000, 1001, 1002, 5000, 1003, 1004]
consecutive_days = longest_consecutive(daily_tokens)
# Identify periods of consistent usage
```

### 3. Request ID Sequencing
```python
# Detect gaps in request ID sequences for debugging
request_ids = [1001, 1002, 1003, 1010, 1011, 1012]
max_sequence = longest_consecutive(request_ids)
# Result: 3 (two separate sequences)
# Gap between 1003 and 1010 indicates missing requests
```

### 4. Version Tracking
```python
# Find longest consecutive deployment versions
deployed_versions = [15, 13, 14, 16, 20, 21, 22]
consecutive_deployments = longest_consecutive(deployed_versions)
# Result: 4 (versions 13, 14, 15, 16)
```

### 5. Cache Hit Analysis
```python
# Analyze consecutive cache hit timestamps
cache_hits = [100, 102, 101, 103, 200, 201]
longest_hit_streak = longest_consecutive(cache_hits)
# Result: 4 (timestamps 100, 101, 102, 103)
```

## Follow-up Questions

1. **What if we need to return the actual sequence, not just its length?**
   ```python
   def longest_consecutive_sequence(nums):
       # Track the start of the longest sequence
       # Return the actual sequence as a list
   ```

2. **What if we have a stream of numbers and need to update the longest sequence dynamically?**
   - Use a Union-Find (Disjoint Set) data structure
   - Or maintain a hash map with dynamic updates

3. **What if numbers can be negative or very large?**
   - Hash set approach handles this naturally
   - No special modifications needed

4. **What if we want the k longest consecutive sequences?**
   - Use a heap to track top k sequences
   - Modify the algorithm to store all sequence lengths

5. **How would you solve this in a distributed system?**
   - MapReduce approach
   - Partition numbers into ranges
   - Merge results from different partitions

6. **What if the array is read-only and we can't use extra space?**
   - Cannot achieve O(n) without extra space
   - Would need to use sorting (O(n log n)) with O(1) space

## Common Mistakes to Avoid

1. **Using sorting** (violates O(n) requirement)

2. **Not handling the "sequence start" check:**
   ```python
   # WRONG - checks every number as a potential start
   for num in num_set:
       current_num = num
       while current_num + 1 in num_set:
           current_num += 1
   # This is O(n²) in worst case!

   # CORRECT - only check sequence starts
   for num in num_set:
       if num - 1 not in num_set:  # ← Critical optimization
           while current_num + 1 in num_set:
               current_num += 1
   ```

3. **Not handling duplicates:**
   ```python
   # Using a set automatically handles duplicates
   num_set = set(nums)  # Good!
   ```

4. **Off-by-one errors in the while loop:**
   ```python
   current_streak = 1  # Start at 1, not 0 (current number counts)
   ```

5. **Not handling empty array:**
   ```python
   if not nums:
       return 0
   ```

## Summary

This problem demonstrates:
- ✅ Hash set for O(1) lookups to achieve O(n) overall complexity
- ✅ Clever optimization: only process sequence starts
- ✅ Amortized analysis: understanding why nested loops can still be O(n)
- ✅ Space-time tradeoff: O(n) space for O(n) time
- ✅ Practical applications in performance monitoring and pattern analysis

**The key insight:** By only processing sequence starts (`num - 1 not in set`), we ensure each number is visited at most twice total, achieving O(n) time complexity despite the nested while loop.

Master this problem, and you'll understand advanced hash map optimization techniques crucial for technical interviews and real-world system design!
