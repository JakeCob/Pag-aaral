# Top K Frequent Elements

## Problem Statement
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.

**Constraints:**
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
- k is in the range [1, number of unique elements]
- The answer is guaranteed to be unique

## Examples

### Example 1:
```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

### Example 2:
```
Input: nums = [1], k = 1
Output: [1]
```

### Example 3:
```
Input: nums = [4,4,4,2,2,5,5,5,5], k = 2
Output: [5,4]
```

## Approach: Hash Map + Bucket Sort

### Key Insight
Instead of sorting by frequency (O(n log n)), we can use **bucket sort** since frequencies are bounded by the array length.

### Algorithm Steps
1. **Count frequencies** using a hash map
2. **Create buckets** where index = frequency
3. **Fill buckets** with numbers that have that frequency
4. **Collect from buckets** starting from highest frequency until we have k elements

### Why Bucket Sort?
- Frequencies range from 1 to n (where n = len(nums))
- We can create n+1 buckets and place each number in bucket[frequency]
- Then iterate from highest frequency to lowest to get top k

## Solution

```python
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """
    Find k most frequent elements using hash map and bucket sort.

    Time Complexity: O(n)
        - Counting frequencies: O(n)
        - Creating buckets: O(n)
        - Filling buckets: O(n)
        - Collecting results: O(n) worst case
        - Total: O(n)

    Space Complexity: O(n)
        - Frequency map: O(n) for unique elements
        - Buckets array: O(n) with n buckets
        - Total: O(n)
    """
    # Step 1: Count frequencies
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1

    # Step 2: Create buckets where index = frequency
    # bucket[i] contains all numbers with frequency i
    buckets = [[] for _ in range(len(nums) + 1)]

    # Step 3: Fill buckets
    for num, freq in freq_map.items():
        buckets[freq].append(num)

    # Step 4: Collect k elements from highest frequency buckets
    result = []
    for freq in range(len(buckets) - 1, 0, -1):  # Iterate backwards
        for num in buckets[freq]:
            result.append(num)
            if len(result) == k:
                return result

    return result


# Test cases
print(top_k_frequent([1,1,1,2,2,3], 2))           # [1, 2]
print(top_k_frequent([1], 1))                      # [1]
print(top_k_frequent([4,4,4,2,2,5,5,5,5], 2))     # [5, 4]
```

## Complexity Analysis with Example

**Input: nums = [1,1,1,2,2,3], k = 2**

### Time: O(n) where n=6

1. **Count frequencies** - O(6):
   ```
   freq_map = {1: 3, 2: 2, 3: 1}
   ```

2. **Create buckets** - O(7):
   ```
   buckets = [[], [], [], [], [], [], []]
   indices:    0   1   2   3   4   5   6
   ```

3. **Fill buckets** - O(3 unique elements):
   ```
   buckets[3].append(1)  → buckets[3] = [1]
   buckets[2].append(2)  → buckets[2] = [2]
   buckets[1].append(3)  → buckets[1] = [3]

   Result:
   buckets = [[], [3], [2], [1], [], [], []]
   indices:    0    1    2    3   4   5   6
   ```

4. **Collect top k=2** - O(k):
   ```
   freq=3: add 1 → result = [1]
   freq=2: add 2 → result = [1, 2]  ✓ len == k, return
   ```

### Space: O(n) where n=6
- `freq_map`: 3 entries
- `buckets`: 7 lists (most empty)
- Total: O(6)

## Alternative: Heap Solution

```python
import heapq

def top_k_frequent_heap(nums: list[int], k: int) -> list[int]:
    """
    Alternative using min-heap.

    Time Complexity: O(n log k)
    Space Complexity: O(n)
    """
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1

    # Min-heap of size k (stores tuples of (frequency, number))
    heap = []
    for num, freq in freq_map.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)

    return [num for freq, num in heap]
```

**Trade-offs:**
- Bucket sort: O(n) time, better for small datasets
- Heap: O(n log k) time, better when k << n (only need top few)

## Key Takeaways

1. **Bucket sort** is perfect when values are bounded (frequencies ≤ n)
2. **Reverse iteration** from high to low frequency gets top k efficiently
3. **Hash map** is essential for O(1) frequency counting
4. Can achieve **O(n) time** without full sorting!
