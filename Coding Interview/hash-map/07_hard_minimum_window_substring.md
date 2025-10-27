# Minimum Window Substring

## Problem Statement
Given two strings `s` and `t`, return the **minimum window substring** of `s` such that every character in `t` (including duplicates) is included in the window. If there is no such substring, return an empty string `""`.

**Constraints:**
- 1 <= s.length, t.length <= 10^5
- s and t consist of uppercase and lowercase English letters
- The testcases will be generated such that the answer is unique

**Follow-up:** Could you find an algorithm that runs in O(m + n) time, where m = len(s) and n = len(t)?

## Examples

### Example 1:
```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
```

### Example 2:
```
Input: s = "a", t = "a"
Output: "a"
```

### Example 3:
```
Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
```

### Example 4:
```
Input: s = "ADOBECODEBANC", t = "AABC"
Output: "ADOBEC"
Explanation: Need two 'A's, one 'B', and one 'C'
```

## Approach: Sliding Window + Two Hash Maps

### Key Insight
Use the **sliding window technique** with two pointers:
1. **Expand** the right pointer to include characters
2. **Contract** the left pointer when we have a valid window
3. Track the **minimum valid window** seen so far

### Algorithm Steps

1. **Build target frequency map** - count characters needed from `t`
2. **Use two pointers** (left, right) to create a sliding window
3. **Expand window** - move right pointer, add characters to window
4. **Contract window** - when valid, move left pointer to minimize
5. **Track minimum** - update result when we find smaller valid window

### Why Two Hash Maps?

- `target_count`: What we need (from string `t`)
- `window_count`: What we currently have in our window
- Compare them to know if window is valid

## Solution

```python
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


# Test cases
print(min_window("ADOBECODEBANC", "ABC"))   # "BANC"
print(min_window("a", "a"))                  # "a"
print(min_window("a", "aa"))                 # ""
print(min_window("ADOBECODEBANC", "AABC"))  # "ADOBEC"
```

## Detailed Walkthrough

**Input: s = "ADOBECODEBANC", t = "ABC"**

### Initial State
```
target_count = {'A': 1, 'B': 1, 'C': 1}
required = 3 (need to satisfy 3 unique chars)
formed = 0
window_count = {}
```

### Step-by-Step Execution

```
Step 1: right=0, s[0]='A'
        window_count = {'A': 1}
        formed = 1 (A satisfied)
        Window: "A" (not valid yet, formed < required)

Step 2: right=1, s[1]='D'
        window_count = {'A': 1, 'D': 1}
        Window: "AD" (not valid)

Step 3: right=2, s[2]='O'
        window_count = {'A': 1, 'D': 1, 'O': 1}
        Window: "ADO" (not valid)

Step 4: right=3, s[3]='B'
        window_count = {'A': 1, 'D': 1, 'O': 1, 'B': 1}
        formed = 2 (A, B satisfied)
        Window: "ADOB" (not valid)

Step 5: right=4, s[4]='E'
        window_count = {'A': 1, 'D': 1, 'O': 1, 'B': 1, 'E': 1}
        Window: "ADOBE" (not valid)

Step 6: right=5, s[5]='C'
        window_count = {'A': 1, 'D': 1, 'O': 1, 'B': 1, 'E': 1, 'C': 1}
        formed = 3 (A, B, C all satisfied) ✓ VALID!
        Window: "ADOBEC" - length 6, update min_len=6, min_start=0

        NOW CONTRACT (while formed == required):

        Contract 1: Remove 'A' at left=0
                   window_count['A'] = 0
                   formed = 2 (broke A requirement)
                   left = 1, stop contracting

Step 7: right=6, s[6]='O'
        window_count = {..., 'O': 2}
        Window: "DOBECO" (not valid, need A)

... continue until right=12 ...

Step 13: right=12, s[12]='C'
         After expansions and contractions:
         Window: "BANC" - length 4 ✓ NEW MINIMUM!
         min_len=4, min_start=9
```

**Final Result: s[9:13] = "BANC"**

## Complexity Analysis

### Time: O(m + n)

**Example: s = "ADOBECODEBANC" (m=13), t = "ABC" (n=3)**

1. **Build target_count**: O(3)
   ```
   {'A': 1, 'B': 1, 'C': 1}
   ```

2. **Sliding window**: O(13 + 13) = O(26)
   - **Right pointer**: Visits each of 13 characters once
   - **Left pointer**: Visits each of 13 characters at most once
   - Total character visits: 26

**Total: O(3 + 26) = O(16) = O(m + n)**

### Space: O(k) where k = unique characters

- `target_count`: At most 52 entries (a-z, A-Z)
- `window_count`: At most 52 entries
- **Effectively O(1)** for English letters

## Key Insights

1. **Sliding window pattern**: Expand right, contract left when valid
2. **"Formed" counter trick**: Avoid comparing full hash maps every iteration
3. **Two hash maps**: Separate what we need vs what we have
4. **Greedy contraction**: Always try to minimize when valid
5. **O(m + n) is possible**: Each character visited at most twice

## Common Mistakes

### ❌ Mistake 1: Comparing entire hash maps
```python
# This is O(k) every iteration! Too slow!
if window_count == target_count:
    # ...
```

### ✅ Correct: Use "formed" counter
```python
# O(1) comparison!
if formed == required:
    # ...
```

### ❌ Mistake 2: Not handling duplicates
```python
target_count = set(t)  # Wrong! Loses duplicate info
```

### ✅ Correct: Count frequencies
```python
target_count = {}
for char in t:
    target_count[char] = target_count.get(char, 0) + 1
```

## Variations

This pattern applies to many problems:
- **Longest substring with K distinct characters**
- **Substring with concatenation of all words**
- **Permutation in string**
- **Find all anagrams in a string**

## Key Takeaways

1. **Sliding window** is powerful for substring problems
2. **Two pointers** can achieve O(n) instead of O(n²)
3. **Hash maps** enable O(1) character tracking
4. **"Formed" optimization** avoids expensive map comparisons
5. **Contraction** is just as important as expansion!
