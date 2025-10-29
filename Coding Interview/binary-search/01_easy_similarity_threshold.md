# Easy: Find Minimum Similarity Threshold for Vector Search

## Problem Statement

You are building a RAG (Retrieval-Augmented Generation) system for ELGO AI. After retrieving document chunks based on vector similarity, you need to filter results by a minimum similarity threshold to ensure quality.

Given a sorted array of similarity scores (in descending order) and a target number of results `k`, find the minimum similarity threshold that would give you at least `k` results.

## Real-World Context

In production RAG systems, you often want to return a minimum number of context chunks (e.g., 3-5 chunks) while maintaining quality. This requires finding the threshold value that includes exactly `k` chunks, ensuring you get enough context without including too many low-quality results.

## Function Signature

```python
def find_similarity_threshold(scores: list[float], k: int) -> float:
    """
    Find the minimum similarity threshold to get at least k results.

    Args:
        scores: List of similarity scores sorted in descending order (0.0 to 1.0)
        k: Minimum number of results needed

    Returns:
        The minimum similarity threshold (the k-th highest score)
        Returns 0.0 if k > len(scores)
    """
    pass
```

## Examples

### Example 1:
```python
scores = [0.95, 0.87, 0.82, 0.76, 0.65, 0.52, 0.41, 0.33]
k = 3

result = find_similarity_threshold(scores, k)
print(result)
```

**Output:**
```
0.82
```

**Explanation:** To get at least 3 results, the threshold should be 0.82 (the 3rd highest score). This includes scores [0.95, 0.87, 0.82].

### Example 2:
```python
scores = [0.99, 0.95, 0.92, 0.88, 0.85]
k = 5

result = find_similarity_threshold(scores, k)
print(result)
```

**Output:**
```
0.85
```

**Explanation:** To get 5 results, we need threshold of 0.85 (5th score).

### Example 3:
```python
scores = [0.78, 0.65, 0.54]
k = 5

result = find_similarity_threshold(scores, k)
print(result)
```

**Output:**
```
0.0
```

**Explanation:** We only have 3 scores but need 5 results. Return 0.0 to indicate we should accept all available scores.

### Example 4:
```python
scores = [0.91, 0.91, 0.88, 0.88, 0.88, 0.75]
k = 3

result = find_similarity_threshold(scores, k)
print(result)
```

**Output:**
```
0.88
```

**Explanation:** Even with duplicate scores, the 3rd element is 0.88.

## Constraints

- `1 <= len(scores) <= 10^4`
- `0.0 <= scores[i] <= 1.0`
- `1 <= k <= 10^4`
- Scores are sorted in descending order
- Scores may contain duplicates

## Key Concepts

- **Array Access**: Direct index-based retrieval
- **Edge Cases**: Handle k larger than array size
- **Sorted Data**: Leverage pre-sorted order

## Approach Hints

1. **Simple Solution**: Since the array is sorted, you can directly access the k-th element
2. **Edge Case**: Check if k exceeds the array length
3. **Time Complexity**: O(1) - constant time access
4. **Space Complexity**: O(1) - no extra space needed

## Why This Matters for AI Engineering

In production RAG systems at ELGO AI:
- You need to balance **recall** (getting enough context) with **precision** (quality of results)
- Dynamic thresholding ensures consistent result counts across varying query similarities
- This is a building block for more complex relevance ranking systems

## Follow-up Questions

1. What if the scores were NOT sorted? How would your approach change?
2. How would you modify this to return all scores above a fixed threshold (e.g., 0.7)?
3. In a distributed system with multiple vector databases, how would you find the global k-th highest score?

## Expected Time Complexity

- **Target**: O(1)
- **Acceptable**: O(log n) if you want to use binary search for practice

## Expected Space Complexity

- **Target**: O(1)

## Testing Your Solution

```python
# Test Case 1: Normal case
assert find_similarity_threshold([0.95, 0.87, 0.82, 0.76], 3) == 0.82

# Test Case 2: k equals array length
assert find_similarity_threshold([0.9, 0.8, 0.7], 3) == 0.7

# Test Case 3: k exceeds array length
assert find_similarity_threshold([0.8, 0.6], 5) == 0.0

# Test Case 4: k = 1 (highest score)
assert find_similarity_threshold([0.99, 0.95, 0.90], 1) == 0.99

# Test Case 5: Duplicate scores
assert find_similarity_threshold([0.9, 0.9, 0.8, 0.8], 2) == 0.9

# Test Case 6: All same scores
assert find_similarity_threshold([0.85, 0.85, 0.85], 2) == 0.85

print("✅ All tests passed!")
```

## Bonus Challenge

Extend your solution to handle the case where you want to return at most `max_results` chunks, but only those above a minimum quality threshold `min_quality`. Return the actual threshold to use.

```python
def find_adaptive_threshold(
    scores: list[float],
    k: int,
    min_quality: float,
    max_results: int
) -> float:
    """
    Find threshold considering both quantity and quality constraints.

    Returns:
        The highest threshold that satisfies all constraints
    """
    pass
```
