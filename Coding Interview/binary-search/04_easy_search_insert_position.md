# Easy: Search Insert Position

## Problem Statement

Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with **O(log n)** runtime complexity.

## Function Signature

```python
def search_insert(nums: list[int], target: int) -> int:
    """
    Find the index of target or where it should be inserted.

    Args:
        nums: Sorted array of distinct integers
        target: Target value to find or insert

    Returns:
        Index of target if found, otherwise index where it should be inserted
    """
    pass
```

## Examples

### Example 1:
```python
nums = [1, 3, 5, 6]
target = 5

result = search_insert(nums, target)
print(result)
```

**Output:**
```
2
```

**Explanation:** 5 is found at index 2.

### Example 2:
```python
nums = [1, 3, 5, 6]
target = 2

result = search_insert(nums, target)
print(result)
```

**Output:**
```
1
```

**Explanation:** 2 is not found. It should be inserted at index 1 (between 1 and 3).

### Example 3:
```python
nums = [1, 3, 5, 6]
target = 7

result = search_insert(nums, target)
print(result)
```

**Output:**
```
4
```

**Explanation:** 7 is larger than all elements. It should be inserted at the end (index 4).

### Example 4:
```python
nums = [1, 3, 5, 6]
target = 0

result = search_insert(nums, target)
print(result)
```

**Output:**
```
0
```

**Explanation:** 0 is smaller than all elements. It should be inserted at the beginning (index 0).

### Example 5:
```python
nums = [1]
target = 1

result = search_insert(nums, target)
print(result)
```

**Output:**
```
0
```

**Explanation:** Single element array, target found at index 0.

## Constraints

- `1 <= nums.length <= 10^4`
- `-10^4 <= nums[i] <= 10^4`
- `nums` contains **distinct** values sorted in **ascending** order
- `-10^4 <= target <= 10^4`

## Key Concepts

- **Binary Search**: Divide and conquer approach for sorted arrays
- **Search Space Reduction**: Eliminate half of remaining elements each iteration
- **Edge Cases**: Target smaller/larger than all elements, single element array
- **Index Calculation**: Understanding mid-point calculation and boundary conditions

## Approach Hints

### Approach 1: Binary Search (Optimal)

**Algorithm:**
1. Initialize two pointers: `left = 0`, `right = len(nums) - 1`
2. While `left <= right`:
   - Calculate mid: `mid = (left + right) // 2`
   - If `nums[mid] == target`: return `mid` (found!)
   - If `nums[mid] < target`: search right half (`left = mid + 1`)
   - If `nums[mid] > target`: search left half (`right = mid - 1`)
3. If not found, `left` is the insertion position

**Why `left` is the answer?**
- After binary search terminates, `left` points to the first position where `nums[i] >= target`
- This is exactly where we should insert the target

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

### Approach 2: Linear Search (Not Optimal)

Simply iterate through the array until you find the first element >= target.

**Time Complexity:** O(n) - doesn't meet requirement
**Space Complexity:** O(1)

## Step-by-Step Trace

Let's trace through Example 2:

```python
nums = [1, 3, 5, 6]
target = 2
```

**Binary Search Execution:**

```
Initial: left=0, right=3

Iteration 1:
  mid = (0 + 3) // 2 = 1
  nums[1] = 3
  3 > 2, so search left half
  right = mid - 1 = 0

Iteration 2:
  left=0, right=0
  mid = (0 + 0) // 2 = 0
  nums[0] = 1
  1 < 2, so search right half
  left = mid + 1 = 1

Exit condition: left > right (1 > 0)
Return left = 1
```

**Result:** Insert at index 1, giving [1, 2, 3, 5, 6] ✓

## Expected Time Complexity

- **Target:** O(log n) using binary search
- **Not Acceptable:** O(n) linear search

## Expected Space Complexity

- **Target:** O(1) constant extra space

## Testing Your Solution

```python
# Test Case 1: Target found in middle
assert search_insert([1, 3, 5, 6], 5) == 2

# Test Case 2: Target not found, insert in middle
assert search_insert([1, 3, 5, 6], 2) == 1

# Test Case 3: Target larger than all elements
assert search_insert([1, 3, 5, 6], 7) == 4

# Test Case 4: Target smaller than all elements
assert search_insert([1, 3, 5, 6], 0) == 0

# Test Case 5: Single element, target found
assert search_insert([1], 1) == 0

# Test Case 6: Single element, target not found (smaller)
assert search_insert([1], 0) == 0

# Test Case 7: Single element, target not found (larger)
assert search_insert([1], 2) == 1

# Test Case 8: Two elements
assert search_insert([1, 3], 2) == 1

# Test Case 9: Large array
assert search_insert(list(range(0, 100, 2)), 51) == 26

# Test Case 10: Negative numbers
assert search_insert([-10, -5, 0, 5, 10], -3) == 2

print("✅ All tests passed!")
```

## Common Mistakes

❌ **Off-by-one errors:**
```python
# Wrong: Using right = len(nums) instead of len(nums) - 1
right = len(nums)  # This will cause index out of bounds

# Correct:
right = len(nums) - 1
```

❌ **Wrong termination condition:**
```python
# Wrong: while left < right (misses single element case)
while left < right:
    ...

# Correct:
while left <= right:
    ...
```

❌ **Wrong mid calculation (integer overflow in some languages):**
```python
# In Python this is fine, but in Java/C++ this can overflow:
mid = (left + right) // 2

# More defensive (prevents overflow):
mid = left + (right - left) // 2
```

❌ **Returning wrong value when target not found:**
```python
# Wrong: returning -1 or right
return -1  # We need insertion position, not "not found"

# Correct:
return left  # left is the insertion position
```

## Variations to Practice

### Variation 1: Find Last Position
Find the rightmost position to insert target (after all equal elements).

```python
def search_insert_last(nums: list[int], target: int) -> int:
    """
    Find the rightmost position to insert target.
    Example: nums=[1,2,2,2,3], target=2 → return 4
    """
    pass
```

### Variation 2: Insert with Duplicates
What if the array can contain duplicates? Find the leftmost position.

```python
def search_insert_with_duplicates(nums: list[int], target: int) -> int:
    """
    Find leftmost position to insert target in array with duplicates.
    Example: nums=[1,2,2,2,3], target=2 → return 1
    """
    pass
```

### Variation 3: Search in Rotated Array
Array is rotated at some pivot point. Find insert position.

```python
def search_insert_rotated(nums: list[int], target: int) -> int:
    """
    Example: nums=[4,5,6,7,0,1,2], target=3 → return 4
    (This is much harder - combines rotation detection with binary search)
    """
    pass
```

## Real-World Applications

This algorithm is fundamental to many applications:

1. **Database Indexing**: Finding where to insert new records in B-trees
2. **Memory Management**: Inserting into sorted free-block lists
3. **Event Scheduling**: Finding insertion point for new events in timeline
4. **Auto-complete**: Finding position in sorted suggestion list
5. **Version Control**: Finding commit position in sorted commit history

## Related LeetCode Problems

- **34. Find First and Last Position of Element in Sorted Array** (Medium)
- **704. Binary Search** (Easy)
- **69. Sqrt(x)** (Easy)
- **278. First Bad Version** (Easy)
- **33. Search in Rotated Sorted Array** (Medium)

## Interview Tips

When asked this problem in an interview:

1. **Clarify requirements**:
   - Are there duplicates? (No in this version)
   - What to return if not found? (Insertion position)
   - Time complexity requirement? (O(log n))

2. **Start with examples**: Walk through 2-3 examples before coding

3. **Explain the approach**:
   - "I'll use binary search since the array is sorted"
   - "After binary search ends, left pointer will be at the insertion position"

4. **Handle edge cases**:
   - Empty array (not in constraints, but good to mention)
   - Single element
   - Target before/after all elements

5. **Test your solution**: Walk through test cases step-by-step

6. **Discuss complexity**: Clearly state O(log n) time, O(1) space

## Binary Search Template

This is a great problem to master the binary search template:

```python
def binary_search_template(nums, target):
    """
    General binary search template.
    Finds the leftmost position where condition is satisfied.
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if condition(mid):  # Your condition here
            # If found or condition met, you might return here
            # Or continue searching left/right
            pass
        elif some_other_condition(mid):
            left = mid + 1  # Search right half
        else:
            right = mid - 1  # Search left half

    return left  # Or right, depending on problem
```

For this specific problem:
```python
def search_insert(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return left
```

Good luck! This is a foundational binary search problem that appears frequently in interviews.
