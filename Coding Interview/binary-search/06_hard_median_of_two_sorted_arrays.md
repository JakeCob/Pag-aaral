# Hard: Median of Two Sorted Arrays

## Problem Statement

Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return **the median** of the two sorted arrays.

The overall run time complexity should be **O(log(min(m,n)))**.

## Function Signature

```python
def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    """
    Find the median of two sorted arrays.

    Args:
        nums1: First sorted array
        nums2: Second sorted array

    Returns:
        Median of the combined sorted arrays
    """
    pass
```

## Examples

### Example 1:
```python
nums1 = [1, 3]
nums2 = [2]

result = find_median_sorted_arrays(nums1, nums2)
print(result)
```

**Output:**
```
2.0
```

**Explanation:**
- Merged array: [1, 2, 3]
- Median is 2.0

### Example 2:
```python
nums1 = [1, 2]
nums2 = [3, 4]

result = find_median_sorted_arrays(nums1, nums2)
print(result)
```

**Output:**
```
2.5
```

**Explanation:**
- Merged array: [1, 2, 3, 4]
- Median is (2 + 3) / 2 = 2.5

### Example 3:
```python
nums1 = [0, 0]
nums2 = [0, 0]

result = find_median_sorted_arrays(nums1, nums2)
print(result)
```

**Output:**
```
0.0
```

### Example 4:
```python
nums1 = []
nums2 = [1]

result = find_median_sorted_arrays(nums1, nums2)
print(result)
```

**Output:**
```
1.0
```

### Example 5:
```python
nums1 = [1, 3, 5, 7, 9]
nums2 = [2, 4, 6, 8, 10]

result = find_median_sorted_arrays(nums1, nums2)
print(result)
```

**Output:**
```
5.5
```

**Explanation:**
- Merged: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
- Length 10 (even), median = (5 + 6) / 2 = 5.5

## Constraints

- `nums1.length == m`
- `nums2.length == n`
- `0 <= m <= 1000`
- `0 <= n <= 1000`
- `1 <= m + n <= 2000`
- `-10^6 <= nums1[i], nums2[i] <= 10^6`

## Key Concepts

- **Median Definition**:
  - If total length is odd: middle element
  - If total length is even: average of two middle elements
- **Partition**: Dividing arrays such that left half ≤ right half
- **Binary Search**: Search for correct partition point
- **Edge Cases**: Empty arrays, arrays of different sizes

## Why This is Hard

This problem is challenging because:
1. **O(log(min(m,n))) requirement** rules out simple O(m+n) merge approach
2. **Partition logic** requires careful handling of edge cases
3. **Binary search on concept** not on values (searching for partition point)
4. **Multiple edge cases** with empty arrays, single elements, etc.

## Approaches

### Approach 1: Merge and Find Median (Not Optimal)

**Algorithm:**
1. Merge both arrays into a single sorted array
2. Find median of merged array

**Time Complexity:** O(m + n) - doesn't meet requirement
**Space Complexity:** O(m + n)

**Code:**
```python
def find_median_sorted_arrays_merge(nums1: list[int], nums2: list[int]) -> float:
    merged = sorted(nums1 + nums2)
    n = len(merged)
    if n % 2 == 1:
        return float(merged[n // 2])
    else:
        return (merged[n // 2 - 1] + merged[n // 2]) / 2.0
```

This works but **doesn't meet the O(log(min(m,n))) requirement**.

### Approach 2: Binary Search on Partition (Optimal)

**Key Insight:**

The median divides the combined array into two equal halves:
```
Left half | Right half
All elements in left ≤ All elements in right
```

We need to find the **partition point** in both arrays such that:
1. `len(left_half) == len(right_half)` (or differ by 1 if odd total length)
2. `max(left_half) <= min(right_half)`

**Algorithm:**

1. **Ensure nums1 is smaller** (optimization):
   ```python
   if len(nums1) > len(nums2):
       nums1, nums2 = nums2, nums1
   ```

2. **Binary search on nums1** for partition point:
   ```python
   left, right = 0, len(nums1)
   ```

3. **For each partition in nums1, calculate partition in nums2**:
   ```python
   partition1 = (left + right) // 2
   partition2 = (m + n + 1) // 2 - partition1
   ```

4. **Get boundary values**:
   ```python
   max_left1 = nums1[partition1 - 1] if partition1 > 0 else -infinity
   min_right1 = nums1[partition1] if partition1 < len(nums1) else +infinity

   max_left2 = nums2[partition2 - 1] if partition2 > 0 else -infinity
   min_right2 = nums2[partition2] if partition2 < len(nums2) else +infinity
   ```

5. **Check partition validity**:
   ```python
   if max_left1 <= min_right2 and max_left2 <= min_right1:
       # Found correct partition!
       if (m + n) % 2 == 0:
           return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
       else:
           return float(max(max_left1, max_left2))
   ```

6. **Adjust search space**:
   ```python
   elif max_left1 > min_right2:
       right = partition1 - 1  # Move partition1 left
   else:
       left = partition1 + 1   # Move partition1 right
   ```

**Time Complexity:** O(log(min(m, n)))
**Space Complexity:** O(1)

## Visual Example

Let's trace Example 2:

```python
nums1 = [1, 2]    # m = 2
nums2 = [3, 4]    # n = 2
```

**Total length:** 4 (even)
**Goal:** Find partition where left half has 2 elements, right half has 2 elements

```
Binary search on nums1:

Iteration 1:
  partition1 = (0 + 2) // 2 = 1
  partition2 = (2 + 2 + 1) // 2 - 1 = 2 - 1 = 1

  nums1: [1 | 2]
          ↑ partition1 = 1
  nums2: [3 | 4]
          ↑ partition2 = 1

  max_left1 = nums1[0] = 1
  min_right1 = nums1[1] = 2
  max_left2 = nums2[0] = 3
  min_right2 = nums2[1] = 4

  Check: max_left1 <= min_right2? (1 <= 4 ✓)
         max_left2 <= min_right1? (3 <= 2 ✗)

  3 > 2, so partition1 is too small
  left = partition1 + 1 = 2

Iteration 2:
  partition1 = (2 + 2) // 2 = 2
  partition2 = (2 + 2 + 1) // 2 - 2 = 2 - 2 = 0

  nums1: [1, 2 | ]
              ↑ partition1 = 2 (all elements in left)
  nums2: [ | 3, 4]
         ↑ partition2 = 0 (all elements in right)

  max_left1 = nums1[1] = 2
  min_right1 = +infinity (no right part)
  max_left2 = -infinity (no left part)
  min_right2 = nums2[0] = 3

  Check: max_left1 <= min_right2? (2 <= 3 ✓)
         max_left2 <= min_right1? (-inf <= +inf ✓)

  Found! Valid partition.

  Left half: [1, 2]
  Right half: [3, 4]

  Total length is even (4):
  Median = (max(2, -inf) + min(+inf, 3)) / 2.0
         = (2 + 3) / 2.0
         = 2.5
```

## Complete Implementation

```python
def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    left, right = 0, m

    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1

        # Get boundary values
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]

        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]

        # Check if we found the correct partition
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found correct partition
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            else:
                return float(max(max_left1, max_left2))
        elif max_left1 > min_right2:
            # partition1 is too large, move left
            right = partition1 - 1
        else:
            # partition1 is too small, move right
            left = partition1 + 1

    raise ValueError("Input arrays are not sorted")
```

## Expected Time Complexity

- **Target:** O(log(min(m, n)))
- **Not Acceptable:** O(m + n) merge approach

## Expected Space Complexity

- **Target:** O(1) constant extra space

## Testing Your Solution

```python
# Test Case 1: Basic case
assert find_median_sorted_arrays([1, 3], [2]) == 2.0

# Test Case 2: Even total length
assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5

# Test Case 3: Duplicates
assert find_median_sorted_arrays([0, 0], [0, 0]) == 0.0

# Test Case 4: Empty first array
assert find_median_sorted_arrays([], [1]) == 1.0

# Test Case 5: Empty second array
assert find_median_sorted_arrays([2], []) == 2.0

# Test Case 6: Larger arrays
assert find_median_sorted_arrays([1, 3, 5, 7, 9], [2, 4, 6, 8, 10]) == 5.5

# Test Case 7: Different sizes
assert find_median_sorted_arrays([1], [2, 3, 4, 5, 6]) == 3.5

# Test Case 8: Negative numbers
assert find_median_sorted_arrays([-5, -3, -1], [0, 2, 4]) == -0.5

# Test Case 9: Single elements
assert find_median_sorted_arrays([1], [2]) == 1.5

# Test Case 10: All in one array
assert find_median_sorted_arrays([1, 2, 3, 4, 5], []) == 3.0

print("✅ All tests passed!")
```

## Common Mistakes

❌ **Not handling empty arrays:**
```python
# Wrong: Will crash on empty array
max_left1 = nums1[partition1 - 1]

# Correct: Use infinity for boundary
max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
```

❌ **Wrong partition calculation:**
```python
# Wrong: Doesn't handle odd lengths correctly
partition2 = (m + n) // 2 - partition1

# Correct: Use (m + n + 1) // 2 to handle both even and odd
partition2 = (m + n + 1) // 2 - partition1
```

❌ **Not ensuring nums1 is smaller:**
```python
# Inefficient: Binary search on larger array
# Should swap to search on smaller array

# Correct:
if len(nums1) > len(nums2):
    nums1, nums2 = nums2, nums1
```

❌ **Wrong median calculation for odd length:**
```python
# Wrong: Taking average even for odd length
return (max_left1 + max_left2) / 2.0

# Correct: Check if total length is odd or even
if (m + n) % 2 == 1:
    return float(max(max_left1, max_left2))
else:
    return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
```

❌ **Off-by-one in binary search:**
```python
# Wrong: Using left < right
while left < right:
    ...

# Correct: Using left <= right
while left <= right:
    ...
```

## Why This Problem is Important

This problem tests multiple advanced concepts:

1. **Binary Search Mastery**: Searching on abstract concept (partition), not values
2. **Edge Case Handling**: Empty arrays, single elements, all elements in one array
3. **Mathematical Reasoning**: Understanding median and partition properties
4. **Complexity Analysis**: Understanding why O(log min(m,n)) is required
5. **Code Quality**: Clean handling of boundary conditions

## Real-World Applications

1. **Database Query Optimization**: Finding median without loading all data
2. **Distributed Systems**: Computing statistics across sorted partitions
3. **Data Streaming**: Maintaining median of incoming sorted streams
4. **Log Analysis**: Finding median latency from multiple sorted log files
5. **A/B Testing**: Comparing median metrics from different sorted datasets

## Variations to Practice

### Variation 1: Kth Element
Instead of median, find the k-th element in merged arrays.

```python
def find_kth_element(nums1: list[int], nums2: list[int], k: int) -> int:
    """
    Find k-th smallest element in two sorted arrays.
    This generalizes the median problem.
    """
    pass
```

### Variation 2: Multiple Arrays
Find median of N sorted arrays.

```python
def find_median_k_arrays(arrays: list[list[int]]) -> float:
    """
    Find median of k sorted arrays.
    Can use heap for efficient merging.
    """
    pass
```

### Variation 3: Stream Median
Maintain median as elements arrive in stream.

```python
class MedianFinder:
    """
    Use two heaps (max heap for left, min heap for right).
    Related LeetCode: 295. Find Median from Data Stream
    """
    pass
```

## Related LeetCode Problems

- **4. Median of Two Sorted Arrays** (This problem - Hard)
- **295. Find Median from Data Stream** (Hard)
- **480. Sliding Window Median** (Hard)
- **719. Find K-th Smallest Pair Distance** (Hard)
- **378. Kth Smallest Element in a Sorted Matrix** (Medium)

## Interview Tips

1. **Start with clarifying questions**:
   - Can arrays be empty?
   - Are arrays guaranteed to be sorted?
   - What if one array is much larger?

2. **Explain the naive approach first**:
   - Merge arrays: O(m+n)
   - Find median: O(1) after merge
   - Total: O(m+n)
   - "But we need better..."

3. **Explain the optimization**:
   - "Instead of merging, we can binary search for partition"
   - "This reduces to O(log min(m,n))"

4. **Walk through a small example**:
   - Draw the partition on paper/whiteboard
   - Show how binary search adjusts partition

5. **Discuss edge cases**:
   - Empty arrays
   - Single element
   - All elements in one array
   - Negative numbers

6. **Code incrementally**:
   - Start with partition logic
   - Add boundary checks
   - Add median calculation
   - Test edge cases

7. **Be prepared to generalize**:
   - "How would you find k-th element?"
   - "What about 3 sorted arrays?"

This is one of the hardest binary search problems. If you can explain and code this clearly, you've mastered binary search!

Good luck! 🚀
