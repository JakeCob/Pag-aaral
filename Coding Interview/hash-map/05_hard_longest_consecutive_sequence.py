'''
Requirements=
- Analyze a stream of LLM API response times (in milliseconds)
- The sequence must be contiguous in value (e.g., 1, 2, 3, 4), but the elements can appear in any order in the input array.

Expected Output = longest consecutive sequence of response time values

Input = **unsorted** array of integers, find the length of the longest consecutive elements sequence.
'''
def longest_consecutive(nums: list[int]) -> int:
    """
    Args:
        nums: Unsorted array of integers (e.g., response times)

    Returns:
        Length of the longest consecutive sequence

    nums = [100, 4, 200, 1, 3, 2]

    result = longest_consecutive(nums)
    print(result)  # Output: 4
    """
    sorted_nums = sorted(list(set(nums)))
    consecutives = []
    temp_index = 0

    for i, num in enumerate(sorted_nums):
        prev_num = sorted_nums[i - 1] if i > 0 else None
        
        if prev_num is not None:
            difference = num - prev_num
            if difference != 1:
                consecutives.append([num])
                temp_index += 1                
            else:
                consecutives[temp_index].append(num)
        else:
            consecutives.append([num])


    sorted_consecutives = sorted(consecutives, key=len, reverse=True)
    return len(sorted_consecutives[0]) if sorted_consecutives else 0
				

# Example 1:
nums = [100, 4, 200, 1, 3, 2]
result = longest_consecutive(nums)
print(result)

# Example 2:
nums = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
result = longest_consecutive(nums)
print(result)

# Example 3:
nums = [9, 1, 4, 7, 3, 2, 8, 5, 6]
result = longest_consecutive(nums)
print(result)

# Example 4:
nums = [1, 2, 0, 1]
result = longest_consecutive(nums)
print(result)

# Example 5:
nums = []
result = longest_consecutive(nums)
print(result)

# Example 6:
nums = [10]
result = longest_consecutive(nums)
print(result)
# Example 8: