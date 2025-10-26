'''
document deduplication system for a RAG pipeline

Input: a list of document chunk identifiers (strings)

Requirements: group together all identifiers that are anagrams of each other. Anagrams are strings that contain the same characters in different orders.

Expected Output: Return the groups as a list of lists.
'''
def get_most_freq(nums: list[int], k: int):
    count_freq = {}
    for num in nums:
        count_freq[num] = count_freq.get(num, 0) + 1

    sorted_count = sorted(count_freq.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_count[:k]]

nums = [1,1,1,2,2,3]
k = 2
result = get_most_freq(nums, k)
print(result)

nums = [1]
k = 1
result = get_most_freq(nums, k)
print(result)

nums = [4,4,4,2,2,5,5,5,5]
k = 2
result = get_most_freq(nums, k)
print(result)