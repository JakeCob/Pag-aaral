# Easy: Document Keyword Frequency Counter

## Problem Statement

You are building a simple keyword extraction feature for an LLM-powered document analysis system. Given a list of processed document chunks (strings) and a target keyword, count how many chunks contain the keyword (case-insensitive).

Additionally, return a mapping of each unique chunk to the number of times it appears in the list, but only for chunks that contain the target keyword.

## Function Signature

```python
def count_keyword_chunks(chunks: list[str], keyword: str) -> dict:
    """
    Args:
        chunks: List of document chunk strings
        keyword: Target keyword to search for (case-insensitive)

    Returns:
        Dictionary with:
        - 'total_matching_chunks': int - number of chunks containing keyword
        - 'chunk_frequency': dict - mapping of matching chunks to their frequencies
    """
    pass
```

## Examples

### Example 1:
```python
chunks = [
    "The AI model generates responses",
    "Machine learning requires data",
    "The AI model is trained on data",
    "The AI model generates responses",
    "Deep learning is a subset of AI"
]
keyword = "ai"

result = count_keyword_chunks(chunks, keyword)
print(result)
```

**Output:**
```python
{
    'total_matching_chunks': 4,
    'chunk_frequency': {
        'The AI model generates responses': 2,
        'The AI model is trained on data': 1,
        'Deep learning is a subset of AI': 1
    }
}
```

### Example 2:
```python
chunks = ["hello world", "goodbye world", "hello there"]
keyword = "world"

result = count_keyword_chunks(chunks, keyword)
print(result)
```

**Output:**
```python
{
    'total_matching_chunks': 2,
    'chunk_frequency': {
        'hello world': 1,
        'goodbye world': 1
    }
}
```

### Example 3:
```python
chunks = ["python programming", "java programming", "c++ coding"]
keyword = "rust"

result = count_keyword_chunks(chunks, keyword)
print(result)
```

**Output:**
```python
{
    'total_matching_chunks': 0,
    'chunk_frequency': {}
}
```

## Constraints

- `1 <= len(chunks) <= 1000`
- `1 <= len(chunks[i]) <= 200`
- `1 <= len(keyword) <= 50`
- Chunks may contain any ASCII characters
- Keyword matching should be case-insensitive
- Keyword matching should be word-boundary aware (e.g., "AI" should match "AI model" but not "AIrplane")

## Solution

```python
def count_keyword_chunks(chunks: list[str], keyword: str) -> dict:
    """
    Count chunks containing a keyword and track their frequencies.

    Time Complexity: O(n * m) where n is number of chunks, m is average chunk length
    Space Complexity: O(k) where k is number of unique matching chunks
    """
    chunk_frequency = {}
    total_matching = 0

    keyword_lower = keyword.lower()

    for chunk in chunks:
        # Case-insensitive keyword matching
        if keyword_lower in chunk.lower():
            total_matching += 1
            # Track frequency of this specific chunk
            chunk_frequency[chunk] = chunk_frequency.get(chunk, 0) + 1

    return {
        'total_matching_chunks': total_matching,
        'chunk_frequency': chunk_frequency
    }


# Alternative solution with word boundary checking
import re

def count_keyword_chunks_word_boundary(chunks: list[str], keyword: str) -> dict:
    """
    Enhanced version with word boundary checking.

    Time Complexity: O(n * m) where n is number of chunks, m is average chunk length
    Space Complexity: O(k) where k is number of unique matching chunks
    """
    chunk_frequency = {}
    total_matching = 0

    # Create regex pattern for word boundary matching
    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)

    for chunk in chunks:
        # Check if keyword exists as a complete word
        if pattern.search(chunk):
            total_matching += 1
            chunk_frequency[chunk] = chunk_frequency.get(chunk, 0) + 1

    return {
        'total_matching_chunks': total_matching,
        'chunk_frequency': chunk_frequency
    }
```

## Test Cases

```python
def test_count_keyword_chunks():
    # Test 1: Basic functionality
    chunks1 = [
        "The AI model generates responses",
        "Machine learning requires data",
        "The AI model is trained on data",
        "The AI model generates responses",
        "Deep learning is a subset of AI"
    ]
    result1 = count_keyword_chunks(chunks1, "ai")
    assert result1['total_matching_chunks'] == 4
    assert result1['chunk_frequency']['The AI model generates responses'] == 2

    # Test 2: No matches
    chunks2 = ["python programming", "java programming"]
    result2 = count_keyword_chunks(chunks2, "rust")
    assert result2['total_matching_chunks'] == 0
    assert result2['chunk_frequency'] == {}

    # Test 3: All chunks match
    chunks3 = ["AI is great", "AI models", "AI systems"]
    result3 = count_keyword_chunks(chunks3, "AI")
    assert result3['total_matching_chunks'] == 3
    assert len(result3['chunk_frequency']) == 3

    # Test 4: Case insensitivity
    chunks4 = ["OpenAI develops models", "openai is a company"]
    result4 = count_keyword_chunks(chunks4, "OpenAI")
    assert result4['total_matching_chunks'] == 2

    print("All tests passed!")

if __name__ == "__main__":
    test_count_keyword_chunks()
```

## Complexity Analysis

### Time Complexity: O(n * m)
- n = number of chunks
- m = average length of each chunk
- For each chunk, we perform a substring search which takes O(m) time
- Total: O(n * m)

### Space Complexity: O(k)
- k = number of unique chunks that match the keyword
- In worst case, k = n (all chunks are unique and match)
- The hash map stores at most k entries

## Key Hash Map Concepts Used

1. **Frequency Counting**: Using hash map to count occurrences
2. **Key Lookup**: Fast O(1) average-case lookup and insertion
3. **get() method**: Safe retrieval with default values
4. **Dictionary Comprehension**: (Optional optimization)

## AI/Backend Context

This problem simulates a common task in RAG (Retrieval-Augmented Generation) systems:
- **Document Chunking**: Breaking documents into processable segments
- **Keyword Search**: Finding relevant chunks for retrieval
- **Deduplication Awareness**: Tracking chunk frequencies helps identify repeated content
- **Cache Optimization**: Frequency data can inform which chunks to cache

## Follow-up Questions

1. How would you modify this to search for multiple keywords?
2. How would you handle very large document collections that don't fit in memory?
3. How could you optimize this if the same chunks list is queried with different keywords repeatedly?
4. What if you needed to return chunks sorted by relevance (keyword frequency within chunk)?
