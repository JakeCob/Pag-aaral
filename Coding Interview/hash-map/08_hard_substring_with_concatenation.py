"""
Substring with Concatenation of All Words - LeetCode Hard

Given a string s and an array of strings words (all of the same length),
find all starting indices of substring(s) in s that is a concatenation of
each word in words exactly once, in any order, and without any intervening characters.
"""

from typing import List
from collections import Counter


def findSubstring(s: str, words: List[str]) -> List[int]:
    """
    Find all starting indices where concatenation of all words appears.

    Approach: Sliding Window with Hash Map
    - Use hash map to track word frequencies
    - Try different starting positions (0 to word_length-1)
    - Slide window word by word, maintaining word counts

    Time: O(n * word_len) where n is length of s
    Space: O(m) where m is number of words
    """
    if not s or not words:
        return []

    word_len = len(words[0])
    word_count = len(words)
    total_len = word_len * word_count

    if len(s) < total_len:
        return []

    # Create frequency map of words
    word_freq = Counter(words)
    result = []

    # Try starting from each position in first word
    for i in range(word_len):
        left = i
        right = i
        current_freq = Counter()
        count = 0  # Number of valid words in current window

        # Slide window through string
        while right + word_len <= len(s):
            # Get word from right pointer
            word = s[right:right + word_len]
            right += word_len

            if word in word_freq:
                current_freq[word] += 1
                count += 1

                # If word appears too many times, shrink window from left
                while current_freq[word] > word_freq[word]:
                    left_word = s[left:left + word_len]
                    current_freq[left_word] -= 1
                    count -= 1
                    left += word_len

                # Check if we have a valid concatenation
                if count == word_count:
                    result.append(left)
                    # Move left pointer to find next potential match
                    left_word = s[left:left + word_len]
                    current_freq[left_word] -= 1
                    count -= 1
                    left += word_len
            else:
                # Word not in words list, reset window
                current_freq.clear()
                count = 0
                left = right

    return result


def findSubstring_naive(s: str, words: List[str]) -> List[int]:
    """
    Naive approach: Check every possible position.

    Time: O(n * m * word_len) - Much slower for large inputs
    Space: O(m)
    """
    if not s or not words:
        return []

    word_len = len(words[0])
    word_count = len(words)
    total_len = word_len * word_count
    word_freq = Counter(words)
    result = []

    # Try each possible starting position
    for i in range(len(s) - total_len + 1):
        seen = Counter()
        j = 0

        # Check each word in potential concatenation
        while j < word_count:
            word_start = i + j * word_len
            word = s[word_start:word_start + word_len]

            if word not in word_freq:
                break

            seen[word] += 1

            if seen[word] > word_freq[word]:
                break

            j += 1

        if j == word_count:
            result.append(i)

    return result


# Test cases
def test_findSubstring():
    # Test case 1: Basic example
    s1 = "barfoothefoobarman"
    words1 = ["foo", "bar"]
    assert sorted(findSubstring(s1, words1)) == [0, 9]
    print("✓ Test 1 passed")

    # Test case 2: No valid concatenation
    s2 = "wordgoodgoodgoodbestword"
    words2 = ["word", "good", "best", "word"]
    assert findSubstring(s2, words2) == []
    print("✓ Test 2 passed")

    # Test case 3: Multiple valid positions
    s3 = "barfoofoobarthefoobarman"
    words3 = ["bar", "foo", "the"]
    assert sorted(findSubstring(s3, words3)) == [6, 9, 12]
    print("✓ Test 3 passed")

    # Test case 4: Duplicate words in array
    s4 = "lingmindraboofooowingdingbarrwingmonkeypoundcake"
    words4 = ["fooo", "barr", "wing", "ding", "wing"]
    assert findSubstring(s4, words4) == [13]
    print("✓ Test 4 passed")

    # Test case 5: All same words
    s5 = "aaaaaaaa"
    words5 = ["aa", "aa", "aa"]
    assert sorted(findSubstring(s5, words5)) == [0, 1, 2]
    print("✓ Test 5 passed")

    # Test case 6: Single word
    s6 = "wordgoodgoodgoodbestword"
    words6 = ["word"]
    assert sorted(findSubstring(s6, words6)) == [0, 20]
    print("✓ Test 6 passed")

    # Test case 7: String too short
    s7 = "abc"
    words7 = ["abcd", "efgh"]
    assert findSubstring(s7, words7) == []
    print("✓ Test 7 passed")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_findSubstring()

    # Example usage
    print("\n--- Example Usage ---")
    s = "barfoothefoobarman"
    words = ["foo", "bar"]
    result = findSubstring(s, words)
    print(f"String: {s}")
    print(f"Words: {words}")
    print(f"Starting indices: {result}")

    # Verify results
    print("\nVerification:")
    for idx in result:
        word_len = len(words[0])
        total_len = word_len * len(words)
        substring = s[idx:idx + total_len]
        print(f"  Index {idx}: '{substring}'")
