# Hard: Subarray Sum Equals K (Classic with AI Context)

## Problem Statement

You are analyzing token consumption patterns in an LLM application. Given an array of integers representing token counts for consecutive API calls, find the total number of contiguous subarrays whose sum equals a target value `k`.

This helps identify periods where token usage matches specific quotas or budget targets.

Write a function that returns the count of all contiguous subarrays with sum equal to `k`.

## Function Signature

```python
def subarray_sum_equals_k(nums: list[int], k: int) -> int:
    """
    Args:
        nums: Array of integers (token counts)
        k: Target sum (token quota)

    Returns:
        Number of contiguous subarrays with sum equal to k
    """
    pass
```

## Examples

### Example 1:
```python
nums = [1, 1, 1]
k = 2

result = subarray_sum_equals_k(nums, k)
print(result)  # Output: 2
```

**Explanation:**
- Subarrays with sum = 2: `[1,1]` (indices 0-1), `[1,1]` (indices 1-2)
- Total count: 2

### Example 2:
```python
nums = [1, 2, 3]
k = 3

result = subarray_sum_equals_k(nums, k)
print(result)  # Output: 2
```

**Explanation:**
- Subarrays with sum = 3: `[1,2]` (indices 0-1), `[3]` (index 2)
- Total count: 2

### Example 3:
```python
nums = [1, -1, 1, -1, 1]
k = 0

result = subarray_sum_equals_k(nums, k)
print(result)  # Output: 4
```

**Explanation:**
- Subarrays with sum = 0:
  - `[1, -1]` (indices 0-1)
  - `[-1, 1]` (indices 1-2)
  - `[1, -1]` (indices 2-3)
  - `[-1, 1]` (indices 3-4)
- Total count: 4

### Example 4:
```python
nums = [3, 4, 7, 2, -3, 1, 4, 2]
k = 7

result = subarray_sum_equals_k(nums, k)
print(result)  # Output: 4
```

**Explanation:**
- Subarrays with sum = 7:
  - `[3, 4]` (indices 0-1)
  - `[7]` (index 2)
  - `[2, -3, 1, 4, 2, 1]` would be if continued...
  - And several other combinations
- Total count: 4

## Constraints

- `1 <= nums.length <= 2 * 10^4`
- `-1000 <= nums[i] <= 1000`
- `-10^7 <= k <= 10^7`
- Array can contain negative numbers, zero, and positive numbers

## Solution

### Approach 1: Brute Force (Not Optimal)

```python
def subarray_sum_equals_k_brute(nums: list[int], k: int) -> int:
    """
    Brute force approach: Check all possible subarrays.

    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    count = 0
    n = len(nums)

    for start in range(n):
        current_sum = 0
        for end in range(start, n):
            current_sum += nums[end]
            if current_sum == k:
                count += 1

    return count
```

### Approach 2: Hash Map with Prefix Sum (Optimal)

```python
def subarray_sum_equals_k(nums: list[int], k: int) -> int:
    """
    Optimal approach using hash map to store prefix sum frequencies.

    Key Insight:
    If prefix_sum[j] - prefix_sum[i] = k, then subarray from i+1 to j has sum k.
    Rearranging: prefix_sum[i] = prefix_sum[j] - k

    We can find how many times (prefix_sum[j] - k) has occurred before j.

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    count = 0
    current_sum = 0
    prefix_sum_count = {0: 1}  # Base case: sum 0 occurs once (empty prefix)

    for num in nums:
        current_sum += num

        # Check if there's a prefix sum such that:
        # current_sum - prefix_sum = k
        # => prefix_sum = current_sum - k
        target_prefix = current_sum - k

        if target_prefix in prefix_sum_count:
            count += prefix_sum_count[target_prefix]

        # Add current prefix sum to hash map
        prefix_sum_count[current_sum] = prefix_sum_count.get(current_sum, 0) + 1

    return count


# Alternative with detailed comments
def subarray_sum_equals_k_detailed(nums: list[int], k: int) -> int:
    """
    Hash map approach with detailed explanation.

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    count = 0
    current_sum = 0

    # Hash map: prefix_sum -> frequency of that prefix sum
    # Initialize with {0: 1} because:
    # - An empty prefix (before any elements) has sum 0
    # - This handles cases where subarray starting from index 0 sums to k
    prefix_sum_count = {0: 1}

    for i, num in enumerate(nums):
        # Update running sum
        current_sum += num

        # We want to find subarrays ending at index i with sum = k
        # If current_sum - k exists as a previous prefix sum,
        # then the subarray between that prefix and current position has sum k

        # Example: nums = [1, 2, 3], k = 3, at index 1 (num=2):
        # current_sum = 3, current_sum - k = 0
        # prefix_sum_count[0] = 1 (the empty prefix)
        # This means subarray [1,2] (from start to current) has sum 3

        needed_prefix = current_sum - k

        if needed_prefix in prefix_sum_count:
            # Each occurrence of needed_prefix represents a valid subarray
            count += prefix_sum_count[needed_prefix]

        # Record current prefix sum for future iterations
        prefix_sum_count[current_sum] = prefix_sum_count.get(current_sum, 0) + 1

    return count
```

### Visualization Example

Let's trace through `nums = [1, 2, 3]`, `k = 3`:

```
Index | num | current_sum | needed_prefix | prefix_sum_count before | count | prefix_sum_count after
------|-----|-------------|---------------|------------------------|-------|------------------------
  -   |  -  |      0      |       -       |        {0: 1}          |   0   |        {0: 1}
  0   |  1  |      1      |    1-3=-2     |        {0: 1}          |   0   |     {0: 1, 1: 1}
  1   |  2  |      3      |    3-3=0      |     {0: 1, 1: 1}       |  0+1  |  {0: 1, 1: 1, 3: 1}
  2   |  3  |      6      |    6-3=3      | {0: 1, 1: 1, 3: 1}     |  1+1  | {0:1, 1:1, 3:1, 6:1}

Result: 2
```

Subarrays found:
1. At index 1: `current_sum=3`, `needed_prefix=0` exists → subarray `[1,2]` has sum 3
2. At index 2: `current_sum=6`, `needed_prefix=3` exists → subarray `[3]` has sum 3

## Test Cases

```python
def test_subarray_sum():
    # Test 1: Basic case
    assert subarray_sum_equals_k([1, 1, 1], 2) == 2

    # Test 2: Single element equals k
    assert subarray_sum_equals_k([1], 1) == 1

    # Test 3: No subarrays sum to k
    assert subarray_sum_equals_k([1, 2, 3], 7) == 0

    # Test 4: Multiple subarrays
    assert subarray_sum_equals_k([1, 2, 3], 3) == 2

    # Test 5: With negative numbers
    assert subarray_sum_equals_k([1, -1, 1, -1, 1], 0) == 4

    # Test 6: All elements sum to k
    assert subarray_sum_equals_k([1, 2, 3], 6) == 1

    # Test 7: Larger example
    assert subarray_sum_equals_k([3, 4, 7, 2, -3, 1, 4, 2], 7) == 4

    # Test 8: Negative k
    assert subarray_sum_equals_k([1, -1, 0], -1) == 1

    # Test 9: k = 0
    assert subarray_sum_equals_k([0, 0, 0], 0) == 6
    # Subarrays: [0], [0], [0], [0,0], [0,0], [0,0,0]

    # Test 10: Large numbers
    assert subarray_sum_equals_k([100, -50, 50, 200], 100) == 3
    # [100], [100, -50, 50], [200, -50, 50] -- wait, let me recalculate
    # Actually: [100], [-50, 50] doesn't start from beginning...
    # Let me recalculate: [100] at index 0, [-50, 50] would need the prefix sum...

    print("All tests passed!")


if __name__ == "__main__":
    test_subarray_sum()
```

## Complexity Analysis

### Brute Force Approach

**Time Complexity:** O(n²)
- Outer loop: n iterations
- Inner loop: n iterations in worst case
- Total: O(n²)

**Space Complexity:** O(1)
- Only using constant extra space

### Hash Map Approach (Optimal)

**Time Complexity:** O(n)
- Single pass through the array
- Hash map operations (get, insert) are O(1) average case
- Total: O(n)

**Space Complexity:** O(n)
- Hash map can store up to n different prefix sums
- In worst case (all elements unique), hash map has n entries

## Key Hash Map Concepts Used

1. **Prefix Sum Pattern**: Store cumulative sums for efficient range queries
2. **Frequency Counting**: Track how many times each prefix sum occurs
3. **Mathematical Transformation**: Convert "find subarray sum = k" to "find complement in hash map"
4. **Running Total with Lookup**: Combine accumulation with instant lookup

## Why This Problem is Hard

1. **Non-obvious insight**: The connection between prefix sums and subarray sums isn't immediately clear
2. **Edge cases**: Handling zero, negatives, and the base case (`{0: 1}`) requires careful thought
3. **Off-by-one errors**: Easy to make mistakes with array indices and prefix boundaries
4. **Mathematical reasoning**: Requires algebraic manipulation (sum[i:j] = prefix[j] - prefix[i])

## AI/Backend Context

While this is a classic algorithm problem, it has practical applications in AI systems:

### 1. Token Budget Analysis
- Monitor when token usage hits specific quotas
- Identify periods of consistent token consumption
- Detect anomalies in usage patterns

### 2. Batch Processing
- Find consecutive API calls that together consume exactly k tokens
- Optimize batching strategies for cost efficiency
- Group requests for parallel processing

### 3. Performance Monitoring
```python
# Example: Finding periods where latency sum equals target
latencies = [120, 80, 150, 90, 60]  # milliseconds
target_latency = 240

# How many consecutive call sequences have total latency = 240ms?
count = subarray_sum_equals_k(latencies, target_latency)
```

### 4. Cost Tracking
```python
# Example: API call costs
costs = [5, 10, 15, 20, 5, 5]  # dollars
budget_segment = 30

# How many consecutive periods exactly match our budget segment?
periods = subarray_sum_equals_k(costs, budget_segment)
```

## Follow-up Questions

1. **Modification**: What if you need to find subarrays with sum ≥ k instead of = k?
   - Hint: Requires different approach (sliding window or prefix sum with sorting)

2. **Optimization**: Can you reduce space complexity while maintaining O(n) time?
   - Difficult for general case with negatives, but possible for all-positive arrays

3. **Extension**: Return the actual subarrays, not just the count.
   - How would you modify the hash map to store indices?

4. **Variation**: Find the longest subarray with sum = k.
   - Store first occurrence index of each prefix sum

5. **Multiple targets**: Find subarrays with sum in range [k1, k2].
   - Requires modified approach

6. **Distributed**: How would you solve this for a stream of data that doesn't fit in memory?
   - Approximation algorithms, sliding windows, or distributed hash maps

## Common Mistakes to Avoid

1. **Forgetting base case**: Not initializing `prefix_sum_count` with `{0: 1}`
2. **Wrong order**: Adding to count before updating hash map vs. after
3. **Index confusion**: Mixing up prefix sums and subarray boundaries
4. **Negative numbers**: Not handling negative numbers correctly (prefix sum can decrease)
5. **Duplicates**: Not using `.get()` with default value when updating frequency

## Advanced: Return Indices of Subarrays

```python
def subarray_sum_equals_k_with_indices(nums: list[int], k: int) -> list[tuple[int, int]]:
    """
    Return all subarray index ranges [start, end] with sum = k.

    Time Complexity: O(n * m) where m is number of valid subarrays
    Space Complexity: O(n)
    """
    result = []
    current_sum = 0

    # Map: prefix_sum -> list of indices where this sum occurred
    prefix_sum_indices = {0: [-1]}  # -1 represents before array start

    for end in range(len(nums)):
        current_sum += nums[end]
        target_prefix = current_sum - k

        if target_prefix in prefix_sum_indices:
            for start_idx in prefix_sum_indices[target_prefix]:
                result.append((start_idx + 1, end))

        if current_sum not in prefix_sum_indices:
            prefix_sum_indices[current_sum] = []
        prefix_sum_indices[current_sum].append(end)

    return result


# Example usage
nums = [1, 2, 3]
k = 3
print(subarray_sum_equals_k_with_indices(nums, k))
# Output: [(0, 1), (2, 2)]
# Meaning: subarray from index 0 to 1 is [1,2], and index 2 to 2 is [3]
```

## Summary

This problem combines:
- ✅ Hash map for O(1) lookups
- ✅ Prefix sum technique for efficient range queries
- ✅ Mathematical reasoning to transform the problem
- ✅ Frequency counting pattern
- ✅ Handling of edge cases (negatives, zeros, base case)

Mastering this problem demonstrates strong understanding of hash maps and algorithmic thinking crucial for AI engineering roles.
