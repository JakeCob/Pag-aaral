'''
document deduplication system for a RAG pipeline

Input: a list of document chunk identifiers (strings)

Requirements: group together all identifiers that are anagrams of each other. Anagrams are strings that contain the same characters in different orders.

Expected Output: Return the groups as a list of lists.
'''

def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Args:
        strs: List of strings (document chunk identifiers)

    Returns:
        List of lists, where each inner list contains anagrams grouped together
    """
    anagrams = []
    anagram_index = {}
    not_done = [0]
    done = []
    i = 0
    start = 0
    while start < len(strs):
        current_str = strs[start]
        anagrams.append([current_str])
        letters = list(current_str)
        for end in range (start + 1, len(strs)):
            other_str = strs[end]
            if end in done:
                continue
            for letter in letters:
                if letter not in other_str and not anagram_index.get(other_str, None) and not end in not_done:
                    not_done.append(end)
                    break
            if not end in not_done:                
                anagrams[i].append(other_str)
                done.append(end)
            
        try:
            i += 1
            start = not_done[i]
            if start == len(strs):
                start += 1
        except Exception:
            break

    return anagrams
				
	
# Example 1
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]

result = group_anagrams(strs)
print(result)

strs = [""]

result = group_anagrams(strs)
print(result)

strs = ["a"]

result = group_anagrams(strs)
print(result)

strs = ["abc", "bca", "cab", "xyz", "zyx", "yxz"]

result = group_anagrams(strs)
print(result)