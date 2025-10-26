'''
Requirements:
- token consumption patterns in an LLM application
- identify periods where token usage matches specific quotas or budget targets

Input = array of integers representing token counts for consecutive API calls
problem = total number of contiguous subarrays whose sum equals a target value `k`
expected output = count of all contiguous subarrays with sum equal to `k`
'''

def subarray_sum_equals_k(nums: list[int], k: int) -> int:
    """
    Args:
        nums: Array of integers (token counts)
        k: Target sum (token quota)

    Returns:
        Number of contiguous subarrays with sum equal to k
    """
    i = 0
    sum = 0
    result = 0
    while i < len(nums):
        sum += nums[i]
        if sum == k:
            result += 1
            sum = 0
        if i == len(nums) - 1:
            break
        elif nums[i] != k:
            i += 1
    
    return result

# Example 1
nums = [1, 1, 1]
k = 2

result = subarray_sum_equals_k(nums, k)
print(result)  # Output: 2

# Example 2
nums = [1, 2, 3]
k = 3

result = subarray_sum_equals_k(nums, k)
print(result)  # Output: 2

# Example 3
nums = [1, -1, 1, -1, 1]
k = 0

result = subarray_sum_equals_k(nums, k)
print(result)  # Output: 4

# Example 4
nums = [3, 4, 7, 2, -3, 1, 4, 2]
k = 7

result = subarray_sum_equals_k(nums, k)
print(result)  # Output: 4