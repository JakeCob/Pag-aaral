def min_window(s: str, t: str) -> str:
    """
    Find minimum window substring using sliding window + hash maps.

    Time Complexity: O(m + n)
        - Build target_count: O(n) where n = len(t)
        - Sliding window: O(m) where m = len(s)
            - Each character added once (right pointer)
            - Each character removed once (left pointer)
        - Total: O(m + n)

    Space Complexity: O(m + n)
        - target_count: O(n) - unique chars in t
        - window_count: O(m) - unique chars in s
        - In practice: O(1) since only 52 possible letters (a-z, A-Z)
    """
    if not s or not t:
        return ""

    # Step 1: Build target frequency map
    target_count = {}
    for char in t:
        target_count[char] = target_count.get(char, 0) + 1

    # Tracking variables
    left = 0
    min_len = float('inf')
    min_start = 0

    # How many unique characters in t we need to satisfy
    required = len(target_count)
    # How many unique characters we currently satisfy
    formed = 0

    # Current window character counts
    window_count = {}

    # Step 2: Expand window with right pointer
    for right in range(len(s)):
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1

        # Check if current character satisfies the requirement
        if char in target_count and window_count[char] == target_count[char]:
            formed += 1

        # Step 3: Contract window with left pointer when valid
        while formed == required and left <= right:
            # Update minimum window
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left

            # Remove leftmost character
            left_char = s[left]
            window_count[left_char] -= 1

            # Check if we broke a requirement
            if left_char in target_count and window_count[left_char] < target_count[left_char]:
                formed -= 1

            left += 1

    # Step 4: Return result
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]


# Alternative: Using Counter from collections
from collections import Counter

def min_window_v2(s: str, t: str) -> str:
    """
    Cleaner version using Counter.
    Same complexity: O(m + n) time, O(1) space
    """
    if not s or not t:
        return ""

    target_count = Counter(t)
    required = len(target_count)
    formed = 0

    window_count = {}
    left = 0
    min_len = float('inf')
    min_start = 0

    for right in range(len(s)):
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1

        if char in target_count and window_count[char] == target_count[char]:
            formed += 1

        while formed == required and left <= right:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left

            left_char = s[left]
            window_count[left_char] -= 1

            if left_char in target_count and window_count[left_char] < target_count[left_char]:
                formed -= 1

            left += 1

    return "" if min_len == float('inf') else s[min_start:min_start + min_len]


if __name__ == "__main__":
    print("Basic Solution:")
    print(min_window("ADOBECODEBANC", "ABC"))    # "BANC"
    print(min_window("a", "a"))                   # "a"
    print(min_window("a", "aa"))                  # ""
    print(min_window("ADOBECODEBANC", "AABC"))   # "ADOBEC"

    print("\nCounter Solution:")
    print(min_window_v2("ADOBECODEBANC", "ABC")) # "BANC"
    print(min_window_v2("a", "a"))                # "a"
    print(min_window_v2("a", "aa"))               # ""
    print(min_window_v2("ADOBECODEBANC", "AABC"))# "ADOBEC"
