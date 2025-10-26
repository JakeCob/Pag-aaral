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


# Alternative: Heap Solution
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


# Test cases
if __name__ == "__main__":
    print("Bucket Sort Solution:")
    print(top_k_frequent([1,1,1,2,2,3], 2))           # [1, 2]
    print(top_k_frequent([1], 1))                      # [1]
    print(top_k_frequent([4,4,4,2,2,5,5,5,5], 2))     # [5, 4]

    print("\nHeap Solution:")
    print(top_k_frequent_heap([1,1,1,2,2,3], 2))      # [1, 2] or [2, 1]
    print(top_k_frequent_heap([1], 1))                 # [1]
    print(top_k_frequent_heap([4,4,4,2,2,5,5,5,5], 2)) # [5, 4] or [4, 5]
