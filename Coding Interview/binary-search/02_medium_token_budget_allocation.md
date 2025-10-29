# Medium: Token Budget Allocation for LLM Context Window

## Problem Statement

You are optimizing ELGO AI's RAG system to efficiently use the LLM's context window. Given an array of document chunk sizes (in tokens) sorted in ascending order, and a maximum token budget, find the maximum number of chunks you can include in the context without exceeding the budget.

Additionally, implement a function to find the minimum chunk size you should filter out to fit at least `k` chunks within the budget.

## Real-World Context

LLMs have fixed context windows (e.g., GPT-4: 8K, 32K, 128K tokens). In production RAG systems:
- You retrieve multiple relevant chunks
- Each chunk has a token count
- You need to maximize context while staying within the token budget
- System messages, prompts, and response buffer also consume tokens

This is a variant of the "capacity problem" where you need to optimize context inclusion.

## Function Signature

```python
def max_chunks_within_budget(chunk_sizes: list[int], budget: int) -> int:
    """
    Find the maximum number of chunks that fit within token budget.

    Args:
        chunk_sizes: List of chunk sizes in tokens (sorted ascending)
        budget: Maximum token budget available

    Returns:
        Maximum number of chunks that fit within budget
    """
    pass


def find_min_size_threshold(chunk_sizes: list[int], budget: int, k: int) -> int:
    """
    Find the minimum chunk size threshold to fit at least k chunks in budget.

    Args:
        chunk_sizes: List of chunk sizes in tokens (sorted ascending)
        budget: Maximum token budget available
        k: Minimum number of chunks needed

    Returns:
        Minimum chunk size threshold (filter out chunks smaller than this)
        Returns -1 if impossible to fit k chunks
    """
    pass
```

## Examples

### Example 1: Basic Budget Allocation
```python
chunk_sizes = [50, 100, 150, 200, 250, 300, 350]
budget = 600

result = max_chunks_within_budget(chunk_sizes, budget)
print(result)
```

**Output:**
```
4
```

**Explanation:** We can include chunks [50, 100, 150, 200] = 500 tokens (4 chunks). Adding the next chunk (250) would exceed 600.

### Example 2: Finding Size Threshold
```python
chunk_sizes = [80, 120, 150, 200, 250, 300, 400, 500]
budget = 1000
k = 4

result = find_min_size_threshold(chunk_sizes, budget, k)
print(result)
```

**Output:**
```
200
```

**Explanation:**
- If we include all chunks ≥ 200: [200, 250, 300, 400, 500] = 1650 tokens (too much)
- If we include chunks [200, 250, 300] = 750 tokens (only 3 chunks, need 4)
- We need to check: can we fit 4 chunks?
- Actually, chunks [80, 120, 150, 200] = 550 tokens (4 chunks)
- But if threshold is 200, we only include [200, 250, 300, 400, 500]
- We need to find the threshold where exactly k chunks fit
- Threshold 200 means we include [200, 250, 300, 400] = 1150 (too much)
- Threshold 250 means we include [250, 300, 400, 500] = 1450 (too much)
- We need a different interpretation...

**Corrected Interpretation:**
Find minimum size such that if we take chunks >= that size, we can fit at least k of them within budget.

Let me rewrite this more clearly...

### Example 2 (Revised): Finding Size Threshold
```python
chunk_sizes = [50, 100, 150, 200, 250, 300, 350, 400]
budget = 600
k = 3

result = find_min_size_threshold(chunk_sizes, budget, k)
print(result)
```

**Output:**
```
150
```

**Explanation:**
- We want at least 3 chunks within budget of 600
- Try threshold 50: chunks [50, 100, 150, 200, 250, 300, 350, 400]
  - Take 3 smallest: 50 + 100 + 150 = 300 ✓ (fits)
- Try threshold 100: chunks [100, 150, 200, 250, 300, 350, 400]
  - Take 3 smallest: 100 + 150 + 200 = 450 ✓ (fits)
- Try threshold 150: chunks [150, 200, 250, 300, 350, 400]
  - Take 3 smallest: 150 + 200 + 250 = 600 ✓ (fits)
- Try threshold 200: chunks [200, 250, 300, 350, 400]
  - Take 3 smallest: 200 + 250 + 300 = 750 ✗ (doesn't fit)
- Minimum threshold is 150

### Example 3: Impossible Case
```python
chunk_sizes = [400, 500, 600]
budget = 1000
k = 3

result = find_min_size_threshold(chunk_sizes, budget, k)
print(result)
```

**Output:**
```
-1
```

**Explanation:** Even with the smallest chunks [400, 500, 600] = 1500 tokens, we exceed the budget of 1000. Impossible to fit 3 chunks.

### Example 4: Exact Fit
```python
chunk_sizes = [100, 200, 300, 400]
budget = 600

result = max_chunks_within_budget(chunk_sizes, budget)
print(result)
```

**Output:**
```
3
```

**Explanation:** Chunks [100, 200, 300] = 600 tokens exactly.

## Constraints

- `1 <= len(chunk_sizes) <= 10^5`
- `1 <= chunk_sizes[i] <= 10^4`
- `1 <= budget <= 10^6`
- `1 <= k <= len(chunk_sizes)`
- Array is sorted in ascending order
- All values are positive integers

## Key Concepts

- **Binary Search**: Search for optimal threshold value
- **Prefix Sum**: Calculate cumulative token usage
- **Greedy Selection**: Always take smallest chunks first to maximize count
- **Feasibility Check**: Verify if k chunks can fit within budget

## Approach Hints

### For `max_chunks_within_budget`:
1. Use **prefix sum** or **cumulative sum** approach
2. Binary search on the number of chunks (answer space)
3. For each candidate count, check if sum of that many smallest chunks <= budget
4. Time Complexity: O(log n) with O(1) feasibility check if using prefix sum

### For `find_min_size_threshold`:
1. Binary search on the threshold value (chunk size)
2. For each candidate threshold:
   - Filter chunks >= threshold
   - Take k smallest chunks from filtered list
   - Check if sum <= budget
3. Find the minimum threshold where this is possible
4. Time Complexity: O(n log(max_size))

## Why This Matters for AI Engineering

At ELGO AI, this problem represents real production challenges:
- **Context window optimization**: Maximizing relevant information within token limits
- **Cost optimization**: Each token costs money in API calls
- **Quality vs Quantity tradeoff**: More chunks ≠ better answers
- **Dynamic chunk selection**: Adapting to different model context sizes
- **Hybrid search strategies**: Combining multiple ranking signals

## Step-by-Step Example

Let's trace through finding `max_chunks_within_budget`:

```python
chunk_sizes = [50, 100, 150, 200, 250]
budget = 500
```

**Approach with Prefix Sum:**
```
Prefix sums: [50, 150, 300, 500, 750]

Binary search on answer (number of chunks):
- left = 0, right = 5
- mid = 2: prefix_sum[2] = 150 <= 500 ✓ (try more)
- left = 3, right = 5
- mid = 4: prefix_sum[4] = 500 <= 500 ✓ (try more)
- left = 5, right = 5
- mid = 5: prefix_sum[5] = 750 > 500 ✗ (too much)
- left = 5, right = 4 (stop)

Answer: 4 chunks
```

## Expected Time Complexity

- **`max_chunks_within_budget`**:
  - O(n) with prefix sum + linear scan
  - O(n log n) with binary search on answer
  - O(n) preprocessing, O(log n) per query with prefix sum array

- **`find_min_size_threshold`**:
  - O(n log(max_size)) with binary search on threshold
  - Can optimize to O(n log n) with better feasibility check

## Expected Space Complexity

- O(n) if using prefix sum array
- O(1) if calculating sums on-the-fly

## Testing Your Solution

```python
# Test max_chunks_within_budget
assert max_chunks_within_budget([50, 100, 150, 200, 250, 300], 600) == 4
assert max_chunks_within_budget([100, 200, 300], 600) == 3
assert max_chunks_within_budget([100, 200, 300], 50) == 0
assert max_chunks_within_budget([50], 100) == 1
assert max_chunks_within_budget([100, 100, 100], 250) == 2

# Test find_min_size_threshold
assert find_min_size_threshold([50, 100, 150, 200, 250], 400, 3) == 100
assert find_min_size_threshold([100, 200, 300], 700, 3) == 100
assert find_min_size_threshold([400, 500, 600], 1000, 3) == -1
assert find_min_size_threshold([50, 100, 150], 500, 2) == 50
assert find_min_size_threshold([100, 200, 300, 400], 600, 2) == 100

print("✅ All tests passed!")
```

## Bonus Challenges

### Challenge 1: Weighted Chunks
Each chunk has both a size (tokens) and a relevance score. Find the maximum total relevance score within the budget.

```python
def max_relevance_within_budget(
    chunks: list[tuple[int, float]],  # (size, relevance_score)
    budget: int
) -> float:
    """
    This is a variant of the 0/1 knapsack problem.
    Requires dynamic programming, not binary search.
    """
    pass
```

### Challenge 2: Multi-Model Support
Given multiple LLM options with different context windows and costs, determine the optimal model selection and chunk allocation.

```python
def optimize_multi_model(
    chunk_sizes: list[int],
    models: list[dict],  # [{"context": 8000, "cost_per_token": 0.001}, ...]
    k: int
) -> dict:
    """
    Return: {"model": model_index, "chunks": num_chunks, "cost": total_cost}
    """
    pass
```

## Common Pitfalls

❌ **Don't:**
- Forget that array is sorted (leverage this property!)
- Use O(n²) brute force when binary search is possible
- Ignore edge cases (budget < smallest chunk, k > array length)
- Forget to handle empty result cases

✅ **Do:**
- Use prefix sums for efficient range sum queries
- Binary search on the answer or threshold value
- Handle edge cases explicitly
- Consider integer overflow for very large budgets

## Related LeetCode Problems

- Binary Search (Easy)
- Capacity To Ship Packages Within D Days (Medium)
- Koko Eating Bananas (Medium)
- Minimum Speed to Arrive on Time (Medium)
