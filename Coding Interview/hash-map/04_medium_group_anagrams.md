# Medium: Group Anagrams (Classic with AI Context)

## Problem Statement

You are building a document deduplication system for a RAG pipeline. Given a list of document chunk identifiers (strings), group together all identifiers that are anagrams of each other. Anagrams are strings that contain the same characters in different orders.

This helps identify duplicate or semantically similar document chunks that might have been processed in different orders.

Return the groups as a list of lists.

## Function Signature

```python
def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Args:
        strs: List of strings (document chunk identifiers)

    Returns:
        List of lists, where each inner list contains anagrams grouped together
    """
    pass
```

## Examples

### Example 1:
```python
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]

result = group_anagrams(strs)
print(result)
```

**Output:**
```python
[
    ["eat", "tea", "ate"],
    ["tan", "nat"],
    ["bat"]
]
```

**Note:** The order of groups and order within groups doesn't matter.

### Example 2:
```python
strs = [""]

result = group_anagrams(strs)
print(result)
```

**Output:**
```python
[[""]]
```

### Example 3:
```python
strs = ["a"]

result = group_anagrams(strs)
print(result)
```

**Output:**
```python
[["a"]]
```

### Example 4:
```python
strs = ["abc", "bca", "cab", "xyz", "zyx", "yxz"]

result = group_anagrams(strs)
print(result)
```

**Output:**
```python
[
    ["abc", "bca", "cab"],
    ["xyz", "zyx", "yxz"]
]
```

## Constraints

- `1 <= strs.length <= 10^4`
- `0 <= strs[i].length <= 100`
- `strs[i]` consists of lowercase English letters only

## Solution

### Approach 1: Sorted String as Key

The key insight: **Anagrams produce the same string when sorted**.

For example:
- `"eat"` → sorted → `"aet"`
- `"tea"` → sorted → `"aet"`
- `"ate"` → sorted → `"aet"`

All three have the same sorted form, so they're anagrams!

```python
def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Group anagrams using sorted string as hash map key.

    Time Complexity: O(n * k log k)
        - n = number of strings
        - k = maximum length of a string
        - Sorting each string takes O(k log k)
        - Total: O(n * k log k)

    Space Complexity: O(n * k)
        - Hash map stores all n strings
        - Each string has average length k
    """
    anagram_groups = {}

    for s in strs:
        # Sort the string to create a key
        # Anagrams will have the same sorted key
        sorted_key = ''.join(sorted(s))

        # Add the original string to the group with this key
        if sorted_key not in anagram_groups:
            anagram_groups[sorted_key] = []
        anagram_groups[sorted_key].append(s)

    # Return all groups as a list of lists
    return list(anagram_groups.values())
```

**Alternative with defaultdict:**

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Cleaner version using defaultdict.

    Time Complexity: O(n * k log k)
    Space Complexity: O(n * k)
    """
    anagram_groups = defaultdict(list)

    for s in strs:
        sorted_key = ''.join(sorted(s))
        anagram_groups[sorted_key].append(s)

    return list(anagram_groups.values())
```

### Approach 2: Character Count as Key (Optimized)

Instead of sorting (O(k log k)), we can count character frequencies (O(k)).

The key insight: **Anagrams have the same character frequency distribution**.

For example:
- `"eat"` → `{e:1, a:1, t:1}`
- `"tea"` → `{e:1, a:1, t:1}`
- `"ate"` → `{e:1, a:1, t:1}`

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Group anagrams using character count as hash map key.

    Time Complexity: O(n * k)
        - n = number of strings
        - k = maximum length of a string
        - Counting characters in each string takes O(k)
        - Total: O(n * k)

    Space Complexity: O(n * k)
    """
    anagram_groups = defaultdict(list)

    for s in strs:
        # Create a character count array (26 letters)
        char_count = [0] * 26

        for char in s:
            char_count[ord(char) - ord('a')] += 1

        # Convert to tuple to use as dictionary key
        # (lists can't be dictionary keys, but tuples can)
        key = tuple(char_count)

        anagram_groups[key].append(s)

    return list(anagram_groups.values())
```

**Why this is faster:**
- Sorting: O(k log k) per string
- Counting: O(k) per string
- For large k, counting is significantly faster

### Approach 3: Prime Number Hashing (Advanced)

Assign each letter a unique prime number, multiply them together.

**Why it works:** The fundamental theorem of arithmetic states that every integer has a unique prime factorization.

```python
def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Group anagrams using prime number product as key.

    WARNING: This can cause integer overflow for long strings!
    Use only for educational purposes or with overflow handling.

    Time Complexity: O(n * k)
    Space Complexity: O(n * k)
    """
    # Assign prime numbers to each letter
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]

    anagram_groups = defaultdict(list)

    for s in strs:
        key = 1
        for char in s:
            key *= primes[ord(char) - ord('a')]

        anagram_groups[key].append(s)

    return list(anagram_groups.values())
```

**Caveat:** This approach can cause integer overflow for long strings with many characters. Approach 2 (character count) is generally better.

## Test Cases

```python
def test_group_anagrams():
    # Test 1: Basic example
    result1 = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    # Check that we have 3 groups
    assert len(result1) == 3
    # Check that "eat", "tea", "ate" are in the same group
    eat_group = [group for group in result1 if "eat" in group][0]
    assert set(eat_group) == {"eat", "tea", "ate"}

    # Test 2: Empty string
    result2 = group_anagrams([""])
    assert result2 == [[""]]

    # Test 3: Single character
    result3 = group_anagrams(["a"])
    assert result3 == [["a"]]

    # Test 4: All different
    result4 = group_anagrams(["a", "b", "c"])
    assert len(result4) == 3

    # Test 5: All anagrams
    result5 = group_anagrams(["abc", "bca", "cab"])
    assert len(result5) == 1
    assert len(result5[0]) == 3

    # Test 6: Duplicate strings
    result6 = group_anagrams(["eat", "eat", "tea"])
    assert len(result6) == 1
    assert len(result6[0]) == 3

    # Test 7: Empty input (edge case if allowed)
    # result7 = group_anagrams([])
    # assert result7 == []

    print("All tests passed!")


if __name__ == "__main__":
    test_group_anagrams()
```

## Complexity Analysis

### Approach 1: Sorted String as Key

**Time Complexity:** O(n * k log k)
- n = number of strings in input
- k = maximum length of a string
- For each string, we sort it: O(k log k)
- Total: O(n * k log k)

**Space Complexity:** O(n * k)
- Hash map stores all n strings
- Each key (sorted string) has length k
- Total: O(n * k)

### Approach 2: Character Count as Key

**Time Complexity:** O(n * k)
- n = number of strings
- k = maximum length of a string
- For each string, we count characters: O(k)
- Creating the tuple key: O(26) = O(1) (constant)
- Total: O(n * k)

**Space Complexity:** O(n * k)
- Hash map stores all n strings
- Each key is a tuple of length 26 (constant)
- Total: O(n * k)

**Approach 2 is faster** because O(n * k) < O(n * k log k).

## Key Hash Map Concepts Used

1. **Hash Map for Grouping**: Using hash map to group items by a common property
2. **Creative Key Design**: Using sorted strings or character counts as keys
3. **Tuple as Key**: Converting mutable data (list) to immutable (tuple) for use as dictionary key
4. **defaultdict**: Simplifying code by auto-initializing missing keys
5. **values() method**: Extracting all groups from the hash map

## AI/Backend Context

This problem relates to the AI Engineer role in several ways:

### 1. Document Deduplication in RAG Systems
```python
# Example: Grouping similar document chunk identifiers
chunk_ids = [
    "user_query_v1",
    "query_v1_user",  # Anagram of above
    "document_123",
    "321_document",   # Different order, same content
]

groups = group_anagrams(chunk_ids)
# Helps identify potential duplicates to remove before indexing
```

### 2. Caching Similar Queries
```python
# Normalize queries by grouping anagrams
queries = ["AI ML", "ML AI", "deep learning", "learning deep"]
grouped = group_anagrams(queries)

# All anagrams can share the same cache entry
for group in grouped:
    canonical_form = sorted(group)[0]  # Pick one as canonical
    # Cache all variations under the same key
```

### 3. Prompt Optimization
- Identify variations of the same prompt that differ only in word order
- Deduplicate prompt templates before storing in a prompt library
- Analyze user query patterns to find common variations

### 4. Token Optimization
- Group similar text chunks to avoid redundant embeddings
- Reduce vector database storage by identifying duplicate content
- Optimize batch processing by grouping similar inputs

## Follow-up Questions

1. **What if the strings contain Unicode characters instead of just lowercase letters?**
   - Sorted string approach still works
   - Character count approach needs modification (can't use fixed array of 26)

2. **How would you handle case-insensitivity?**
   - Convert all strings to lowercase before processing

3. **What if you need to return groups sorted by size (largest first)?**
   ```python
   return sorted(anagram_groups.values(), key=len, reverse=True)
   ```

4. **How would you find the largest anagram group?**
   ```python
   return max(anagram_groups.values(), key=len)
   ```

5. **What if anagrams should ignore spaces and punctuation?**
   - Filter characters before creating the key:
   ```python
   filtered = ''.join(c for c in s if c.isalpha())
   sorted_key = ''.join(sorted(filtered.lower()))
   ```

6. **How would you implement this in a distributed system with billions of strings?**
   - Use MapReduce: Map each string to (sorted_key, string), then Reduce by grouping
   - Or use a distributed hash table (like Redis Cluster)

## Common Mistakes to Avoid

1. **Using a list as dictionary key**
   ```python
   # WRONG - lists are not hashable
   key = sorted(s)  # This is a list
   anagram_groups[key].append(s)  # Error!

   # CORRECT - convert to string or tuple
   key = ''.join(sorted(s))  # String
   # or
   key = tuple(sorted(s))  # Tuple
   ```

2. **Forgetting to handle empty strings**
   ```python
   # Make sure your solution works for [""]
   ```

3. **Not considering duplicates**
   ```python
   # Input: ["eat", "eat", "tea"]
   # Should return: [["eat", "eat", "tea"]] (all three in one group)
   ```

4. **Modifying input while iterating**
   ```python
   # Don't do this:
   for i, s in enumerate(strs):
       strs[i] = sorted(s)  # Bad practice
   ```

## Visual Example

Let's trace through `["eat", "tea", "tan", "ate", "nat", "bat"]`:

```
Process "eat":
  sorted_key = "aet"
  anagram_groups = {"aet": ["eat"]}

Process "tea":
  sorted_key = "aet"
  anagram_groups = {"aet": ["eat", "tea"]}

Process "tan":
  sorted_key = "ant"
  anagram_groups = {"aet": ["eat", "tea"], "ant": ["tan"]}

Process "ate":
  sorted_key = "aet"
  anagram_groups = {"aet": ["eat", "tea", "ate"], "ant": ["tan"]}

Process "nat":
  sorted_key = "ant"
  anagram_groups = {"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"]}

Process "bat":
  sorted_key = "abt"
  anagram_groups = {"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"], "abt": ["bat"]}

Final result:
[
    ["eat", "tea", "ate"],
    ["tan", "nat"],
    ["bat"]
]
```

## Summary

This problem demonstrates:
- ✅ Using hash maps for grouping by common properties
- ✅ Creative key design (sorted strings or character counts)
- ✅ Trade-offs between different approaches (sorting vs counting)
- ✅ Converting between data types (list → tuple for hashability)
- ✅ Practical applications in RAG systems and document processing

Master this pattern, and you'll be ready for many similar grouping problems in interviews and real-world AI systems!
