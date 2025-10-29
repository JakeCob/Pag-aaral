def search_insert(nums: list[int], target: int) -> int:
    """
    Find the index of target or where it should be inserted.

    Args:
        nums: Sorted array of distinct integers
        target: Target value to find or insert

    Returns:
        Index of target if found, otherwise index where it should be inserted
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return left

# Test Case 1: Target found in middle
nums = [1, 3, 5, 6]
target = 5

result = search_insert(nums, target)
print(result)

# Test Case 2: Target not found, insert in middle
nums = [1, 3, 5, 6]
target = 2

result = search_insert(nums, target)
print(result)

# Test Case 3: Target larger than all elements
nums = [1, 3, 5, 6]
target = 7

result = search_insert(nums, target)
print(result)

# Test Case 4: Target smaller than all elements
nums = [1, 3, 5, 6]
target = 0

result = search_insert(nums, target)
print(result)

# Test Case 5: Single element, target found
nums = [1]
target = 1

result = search_insert(nums, target)
print(result)

# Test Case 6: Single element, target not found (smaller)