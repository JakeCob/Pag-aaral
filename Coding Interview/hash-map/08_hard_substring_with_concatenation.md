# Substring with Concatenation of All Words

**Difficulty:** Hard

## Problem Description

You are given a string `s` and an array of strings `words` of the same length. Return all starting indices of substring(s) in `s` that is a concatenation of each word in `words` exactly once, in any order, and without any intervening characters.

You can return the answer in any order.

## Examples

### Example 1:
```
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation:
Substrings starting at index 0 and 9 are "barfoo" and "foobar" respectively.
The output order does not matter, returning [9,0] is fine too.
```

### Example 2:
```
Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
Output: []
Explanation:
Since words has duplicate "word", we need exactly two "word" in the concatenated substring.
```

### Example 3:
```
Input: s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
Output: [6,9,12]
Explanation:
The substring starting at 6 is "foobarthe".
The substring starting at 9 is "barthefoo".
The substring starting at 12 is "thefoobar".
```

### Example 4:
```
Input: s = "lingmindraboofooowingdingbarrwingmonkeypoundcake", words = ["fooo","barr","wing","ding","wing"]
Output: [13]
Explanation:
Note that "wing" appears twice in words array.
```

## Constraints

- `1 <= s.length <= 10^4`
- `1 <= words.length <= 5000`
- `1 <= words[i].length <= 30`
- `s` and `words[i]` consist of lowercase English letters

## Key Concepts

- **Hash Map**: Track word frequencies and match them in sliding window
- **Sliding Window**: Check each possible starting position with a fixed-size window
- **String Matching**: Efficient substring comparison and validation

## Approach Hints

1. **Hash Map for Frequencies**: Create a hash map to store the frequency of each word in `words`
2. **Fixed Window Size**: Calculate total concatenation length = `len(words) * word_length`
3. **Word-by-Word Sliding**: Slide through `s` one word at a time (not one character)
4. **Multiple Starting Points**: Try starting positions from 0 to `word_length - 1` to cover all possibilities
5. **Optimization**: Use a sliding window that maintains word counts to avoid recounting

## Edge Cases

- Words array contains duplicates
- No valid concatenation exists
- Multiple valid starting positions
- String `s` is shorter than total concatenation length
- All words are the same

## Time Complexity Target

O(n * m * len) where:
- n = length of string s
- m = number of words
- len = length of each word

With optimization using sliding window: O(n * len)

## Space Complexity Target

O(m) for storing word frequencies in hash map
