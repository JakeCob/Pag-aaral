# Complete Text Chunking Strategies for RAG Systems

**Based on "The 5 Levels Of Text Splitting For Retrieval" by Greg Kamradt**

**For companies building custom systems (No LangChain required)**

---

## 🎯 The Chunking Commandment

> "Your goal is not to chunk for chunking's sake, but to get data in a format where it can be retrieved for value later."

---

## Quick Reference Table

| Level | Method | Time Complexity | API Calls | Best For | Interview Frequency |
|-------|--------|-----------------|-----------|----------|---------------------|
| 1 | Character | O(n) | 0 | Prototyping | ⭐⭐ |
| 2 | Recursive | O(n log n) | 0 | **Production default** | ⭐⭐⭐⭐⭐ |
| 3 | Document-Specific | O(n) - O(n log n) | 0 | Code, Markdown, JSON | ⭐⭐⭐⭐ |
| 4 | Semantic | O(s * e) | s embeddings | High-quality retrieval | ⭐⭐⭐ |
| 5 | Agentic | O(p * LLM_time) | Many LLMs | Research only | ⭐ |

---

## Level 1: Character Splitting

### Algorithm

```
1. Initialize: chunks = [], current_chunk = ""
2. For each character in text:
   a. If len(current_chunk) + char > chunk_size:
      - Save current_chunk
      - Start new chunk with overlap
   b. Else: current_chunk += char
3. Save final chunk
4. Return chunks
```

### Vanilla Python Implementation

```python
class CharacterTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap

        return chunks
```

### When to Use
- ✅ Quick prototyping
- ✅ Uniform text without structure
- ❌ Don't use for production (too naive)

### Interview Tip
*"Character splitting is O(n) and simple, but it doesn't respect document structure. I'd only use it for initial testing, then switch to recursive splitting for production."*

---

## Level 2: Recursive Character Text Splitting ⭐

### Algorithm

```
RecursiveSplit(text, separators):
    1. If len(text) <= chunk_size:
       return [text]

    2. Find first separator that exists in text:
       for sep in ["\n\n", "\n", " ", ""]:
           if sep in text:
               use this separator
               break

    3. Split by separator:
       splits = text.split(separator)

    4. Process splits:
       good_splits = []
       for split in splits:
           if len(split) <= chunk_size:
               good_splits.append(split)
           else:
               # Too large, needs further splitting
               if good_splits:
                   merge and save good_splits
                   good_splits = []

               # Recursively split with remaining separators
               recursive_chunks = RecursiveSplit(split, remaining_separators)
               add recursive_chunks to final

    5. Merge remaining good_splits with overlap
    6. Return all chunks
```

### Vanilla Python Implementation

```python
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200,
                 separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Default: paragraph → line → space → character
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, seps: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        # Find first separator that exists
        separator = seps[-1]  # Default to last (empty string)
        remaining_seps = []

        for i, sep in enumerate(seps):
            if sep == "" or sep in text:
                separator = sep
                remaining_seps = seps[i+1:]
                break

        # Split by separator
        splits = text.split(separator) if separator else list(text)

        # Process splits
        final_chunks = []
        good_splits = []

        for split in splits:
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                # Save accumulated good splits
                if good_splits:
                    merged = self._merge_splits(good_splits)
                    final_chunks.extend(merged)
                    good_splits = []

                # Recursively split large piece
                if remaining_seps:
                    recursive = self._split_recursive(split, remaining_seps)
                    final_chunks.extend(recursive)
                else:
                    final_chunks.append(split)

        # Merge remaining good splits
        if good_splits:
            merged = self._merge_splits(good_splits)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge small splits with overlap."""
        chunks = []
        current = []
        current_len = 0

        for split in splits:
            if current_len + len(split) > self.chunk_size and current:
                # Save current chunk
                chunks.append("".join(current))

                # Start new with overlap
                overlap_size = 0
                overlap_parts = []
                for part in reversed(current):
                    if overlap_size + len(part) <= self.chunk_overlap:
                        overlap_parts.insert(0, part)
                        overlap_size += len(part)
                    else:
                        break

                current = overlap_parts
                current_len = overlap_size

            current.append(split)
            current_len += len(split)

        if current:
            chunks.append("".join(current))

        return chunks
```

### Why This is the Default

1. **Respects structure**: Keeps paragraphs, sentences, words together
2. **No API costs**: Pure algorithm, no external calls
3. **Fast enough**: O(n log n) is acceptable for most cases
4. **Flexible**: Works for most document types

### Interview Answer Template

*"For production RAG, I'd use recursive character text splitting with separators ["\n\n", "\n", " ", ""]. This respects natural text boundaries - it tries to keep paragraphs together, then sentences, then words. Only splits mid-word as a last resort. It's O(n log n), has no API costs, and works well for 90% of cases."*

---

## Level 3: Document-Specific Splitting

### Python Code Splitter

**Key Insight**: Python has natural boundaries - classes and functions. Keep them together!

```python
class PythonCodeSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Python-specific separator hierarchy
        self.separators = [
            "\nclass ",     # Class definitions
            "\ndef ",       # Top-level functions
            "\n    def ",   # Methods (indented)
            "\n\n",         # Blank lines
            "\n",           # Single lines
            " ",            # Words
            ""              # Characters
        ]

    def split_text(self, code: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        return splitter.split_text(code)
```

### Markdown Splitter

**Key Insight**: Markdown has hierarchical structure via headers. Track context!

```python
class MarkdownSplitter:
    def split_text(self, markdown: str) -> List[Dict]:
        """Split by headers, preserving hierarchy."""
        lines = markdown.split('\n')
        chunks = []
        current_chunk = []
        current_headers = {}  # Track header hierarchy

        for line in lines:
            # Check for header: # Title, ## Subtitle, etc.
            header_match = re.match(r'^(#{1,6})\s+(.*)', line)

            if header_match:
                # Save previous chunk
                if current_chunk:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'metadata': current_headers.copy()
                    })
                    current_chunk = []

                # Update header context
                level = len(header_match.group(1))
                header_text = header_match.group(2)

                # Clear lower-level headers
                keys_to_remove = [k for k in current_headers
                                 if int(k.split()[1]) >= level]
                for key in keys_to_remove:
                    del current_headers[key]

                current_headers[f'Header {level}'] = header_text
                current_chunk.append(line)
            else:
                current_chunk.append(line)

        # Save final chunk
        if current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'metadata': current_headers.copy()
            })

        return chunks
```

**Example Output**:
```python
{
    'content': '### Introduction\nThis is intro text...',
    'metadata': {
        'Header 1': 'Main Title',
        'Header 2': 'Section A',
        'Header 3': 'Introduction'
    }
}
```

### JSON Splitter

**Key Insight**: JSON has object/array structure. Keep logical units together!

```python
class JSONSplitter:
    def split_text(self, json_str: str) -> List[str]:
        data = json.loads(json_str)

        if isinstance(data, list):
            # Split by items
            chunks = []
            current_chunk = []
            current_size = 0

            for item in data:
                item_str = json.dumps(item, indent=2)
                item_size = len(item_str)

                if current_size + item_size > self.chunk_size and current_chunk:
                    chunks.append(json.dumps(current_chunk, indent=2))
                    current_chunk = []
                    current_size = 0

                current_chunk.append(item)
                current_size += item_size

            if current_chunk:
                chunks.append(json.dumps(current_chunk, indent=2))

            return chunks
```

### Interview Tip

*"For code, I'd use language-specific separators to keep functions and classes together. For Markdown, I'd track the header hierarchy to provide context. The key is respecting the document's natural structure."*

---

## Level 4: Semantic Splitting

### Algorithm

```
SemanticSplit(text):
    1. Split into sentences: O(n)
       sentences = split_sentences(text)

    2. Generate embeddings: O(s * embedding_time)
       embeddings = [embed(sent) for sent in sentences]

    3. Calculate distances: O(s * d)
       distances = []
       for i in range(len(embeddings) - 1):
           dist = cosine_distance(emb[i], emb[i+1])
           distances.append(dist)

    4. Find breakpoints: O(s)
       threshold = calculate_threshold(distances, method)
       breakpoints = [i for i, d in enumerate(distances)
                     if d > threshold]

    5. Create chunks from breakpoints

    6. Return chunks
```

### Vanilla Python Implementation

```python
class SemanticTextSplitter:
    def __init__(self, embedding_function,
                 threshold_type="percentile",
                 threshold_amount=95):
        self.embed = embedding_function
        self.threshold_type = threshold_type
        self.threshold_amount = threshold_amount

    def split_text(self, text: str) -> List[str]:
        # 1. Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [text]

        # 2. Generate embeddings
        embeddings = [self.embed(sent) for sent in sentences]

        # 3. Calculate cosine distances
        distances = []
        for i in range(len(embeddings) - 1):
            dist = self._cosine_distance(embeddings[i], embeddings[i+1])
            distances.append(dist)

        # 4. Find breakpoints
        breakpoints = self._find_breakpoints(distances)

        # 5. Create chunks
        chunks = []
        start = 0
        for bp in breakpoints:
            chunk = " ".join(sentences[start:bp+1])
            chunks.append(chunk)
            start = bp + 1

        # Final chunk
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        # Pattern: . ! ? followed by space and capital
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_distance(self, vec1, vec2) -> float:
        """Calculate cosine distance between vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

        # Distance = 1 - similarity
        return 1 - similarity

    def _find_breakpoints(self, distances: List[float]) -> List[int]:
        """Find where to split based on threshold."""
        if not distances:
            return []

        distances_array = np.array(distances)

        if self.threshold_type == "percentile":
            # Split at 95th percentile (default)
            threshold = np.percentile(distances_array, self.threshold_amount)

        elif self.threshold_type == "standard_deviation":
            # Split at mean + k*std
            mean = np.mean(distances_array)
            std = np.std(distances_array)
            threshold = mean + (self.threshold_amount * std)

        elif self.threshold_type == "interquartile":
            # IQR method (robust to outliers)
            q1 = np.percentile(distances_array, 25)
            q3 = np.percentile(distances_array, 75)
            iqr = q3 - q1
            threshold = q3 + (1.5 * iqr)

        # Find indices where distance exceeds threshold
        breakpoints = [i for i, d in enumerate(distances)
                      if d > threshold]

        return breakpoints
```

### Threshold Methods Explained

**1. Percentile (Default)**
```python
# Split when distance > 95th percentile
threshold = np.percentile(distances, 95)
# Top 5% of distances become breakpoints
```

**2. Standard Deviation**
```python
# Split when distance > mean + 1.5*σ
threshold = mean(distances) + 1.5 * std(distances)
# Good for normally distributed data
```

**3. Interquartile Range (Robust)**
```python
# Split using IQR (resistant to outliers)
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
threshold = Q3 + 1.5*IQR  # Same as boxplot outliers
```

### When to Use

✅ **Use semantic splitting when:**
- Budget allows (~$0.001 per sentence for embeddings)
- Retrieval accuracy is critical
- Topics change within documents
- Can cache embeddings

❌ **Don't use when:**
- Need real-time processing
- Budget is tight
- Documents are already well-structured

### Time & Cost Analysis

```
Example: 1000-word document
- ~50 sentences
- 50 embedding API calls
- Cost: 50 * $0.001 = $0.05 per document
- Time: 50 * 100ms = 5 seconds

vs. Recursive Splitting:
- Cost: $0 (no API)
- Time: <10ms
```

### Interview Tip

*"Semantic splitting uses embeddings to find topic boundaries. I'd calculate cosine distance between consecutive sentences and split where distance exceeds the 95th percentile. It's O(s * embedding_time) and costs ~$0.001 per sentence, so only use it when retrieval quality justifies the cost. For most production systems, recursive splitting is sufficient."*

---

## Level 5: Agentic Splitting (Research Only)

### Algorithm

```
AgenticSplit(text, llm):
    1. Extract Propositions:
       propositions = llm("Break this into atomic ideas: {text}")

    2. Group Propositions:
       chunks = []
       current_chunk = [propositions[0]]

       for prop in propositions[1:]:
           should_add = llm(f"Does '{prop}' belong with {current_chunk}?")

           if should_add:
               current_chunk.append(prop)
           else:
               # Finalize chunk
               title = llm(f"Title for: {current_chunk}")
               summary = llm(f"Summary for: {current_chunk}")
               chunks.append({
                   'content': current_chunk,
                   'title': title,
                   'summary': summary
               })
               current_chunk = [prop]

    3. Return chunks with rich metadata
```

### Cost Analysis

```
Example: 1000-word document
- ~100 propositions (1 LLM call)
- ~100 grouping decisions (100 LLM calls)
- ~20 chunks × 2 metadata calls (40 LLM calls)
- Total: ~141 LLM calls

At $0.01 per call = $1.41 per document
At 2 seconds per call = ~5 minutes per document

vs. Semantic: $0.05, 5 seconds
vs. Recursive: $0, 10ms
```

### When to Use

✅ **Only use when:**
- Research/experimentation
- Maximum quality needed
- Cost is not a concern
- Time is not a concern

❌ **Never use for:**
- Production systems
- Real-time processing
- Batch processing
- Cost-sensitive applications

### Interview Tip

*"Agentic splitting uses LLM to make all chunking decisions. It's the highest quality but costs ~$1-2 per document and takes minutes. Only practical for research. In production, I'd use recursive or semantic splitting."*

---

## Bonus: Parent-Document Retrieval

### The Problem

**Small chunks**: Better semantic matching, but lack context for LLM
**Large chunks**: More context, but diluted semantic matching

### The Solution

**Two-stage approach**:
1. **Index small chunks** (400 chars) → Better retrieval
2. **Return large chunks** (2000 chars) → More context

```python
class ParentDocumentSplitter:
    def __init__(self, parent_size=2000, child_size=400, overlap=50):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=overlap
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=overlap
        )

    def split_text(self, text: str) -> List[Dict]:
        """Returns parent-child relationships."""
        parent_chunks = self.parent_splitter.split_text(text)

        result = []
        for parent_id, parent in enumerate(parent_chunks):
            children = self.child_splitter.split_text(parent)

            for child_id, child in enumerate(children):
                result.append({
                    'parent_id': parent_id,
                    'child_id': child_id,
                    'search_text': child,      # Use for embeddings
                    'context_text': parent,    # Return to LLM
                    'metadata': {
                        'parent_size': len(parent),
                        'child_size': len(child)
                    }
                })

        return result
```

### Retrieval Flow

```
User Query → Embed Query
            ↓
Search using child embeddings (precise matching)
            ↓
Return parent chunks (more context)
            ↓
Send to LLM for answer generation
```

### Interview Tip

*"Parent-document retrieval solves the small vs large chunk dilemma. Index small 400-char chunks for precise semantic matching, but return large 2000-char parent chunks for LLM context. Best of both worlds."*

---

## Decision Matrix

### Quick Decision Tree

```
START
  │
  ├─ Real-time requirement? (<100ms)
  │   Yes → Recursive ✓
  │   No → Continue
  │
  ├─ Is it code?
  │   Yes → Document-Specific (Python/JS/etc.) ✓
  │   No → Continue
  │
  ├─ Budget for embeddings? (~$0.05/doc)
  │   No → Recursive ✓
  │   Yes → Continue
  │
  ├─ Need highest quality?
  │   Yes → Semantic ✓
  │   No → Recursive ✓
```

### Comparison Table

| Criteria | Character | Recursive | Doc-Specific | Semantic | Agentic |
|----------|-----------|-----------|--------------|----------|---------|
| **Speed** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡ | ⚡ |
| **Quality** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Cost** | Free | Free | Free | $0.05/doc | $1-2/doc |
| **Setup** | Easy | Easy | Medium | Hard | Hard |
| **Production** | ❌ | ✅ | ✅ | ✅ | ❌ |

---

## Implementation Best Practices

### 1. Start Simple, Iterate

```python
# Phase 1: Start with recursive (Day 1)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Phase 2: Tune parameters (Week 1)
# Test different chunk sizes: 500, 1000, 1500
# Test different overlaps: 10%, 20%, 30%

# Phase 3: Try semantic if needed (Week 2)
# Only if retrieval metrics show room for improvement
```

### 2. Always Measure

```python
def evaluate_chunking(chunks, test_queries):
    """Measure retrieval quality."""
    metrics = {
        'avg_chunk_size': np.mean([len(c) for c in chunks]),
        'chunk_count': len(chunks),
        'retrieval_accuracy': calculate_accuracy(chunks, test_queries),
        'context_preservation': check_context(chunks)
    }
    return metrics
```

### 3. Handle Edge Cases

```python
def robust_split(text, chunk_size):
    # Empty text
    if not text or not text.strip():
        return []

    # Text smaller than chunk_size
    if len(text) <= chunk_size:
        return [text]

    # Invalid chunk_size
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    # Unicode handling
    try:
        text = text.encode('utf-8').decode('utf-8')
    except UnicodeError:
        # Handle encoding issues
        pass

    # Your splitting logic
    return chunks
```

### 4. Optimize for Common Cases

```python
def split_text_optimized(text, chunk_size):
    # Fast path: already small enough
    if len(text) <= chunk_size:
        return [text]

    # Fast path: clean paragraph splits
    if '\n\n' in text:
        paragraphs = text.split('\n\n')
        if all(len(p) <= chunk_size for p in paragraphs):
            return paragraphs

    # Slow path: complex splitting
    return recursive_split(text, chunk_size)
```

---

## Interview Questions & Answers

### Q1: "How would you chunk a 10,000-word document for RAG?"

**Good Answer**:
```
1. Clarify requirements:
   - Use case? (QA, search, summarization)
   - Budget? (free vs paid API)
   - Latency? (real-time vs batch)
   - Document type? (prose, code, structured)

2. My approach:
   - Start: RecursiveCharacterTextSplitter
   - Chunk size: 1000 characters (~200 tokens)
   - Overlap: 200 characters (20%)
   - Separators: ["\n\n", "\n", " ", ""]

3. Why:
   - No API costs
   - Fast: O(n log n), ~10ms for 10k words
   - Respects structure (paragraphs → sentences)
   - Works for 90% of cases

4. Alternative if budget allows:
   - Try semantic splitting
   - Compare retrieval metrics
   - Only adopt if significant improvement

5. Evaluation:
   - Measure chunk size distribution
   - Test retrieval accuracy on sample queries
   - Monitor chunk coherence
```

### Q2: "Implement recursive text splitting from scratch"

```python
def recursive_split(text, chunk_size, separators=["\n\n", "\n", " ", ""]):
    """
    Time: O(n log n) average
    Space: O(n)
    """
    if len(text) <= chunk_size:
        return [text]

    # Find first separator that exists
    for i, sep in enumerate(separators):
        if sep in text or sep == "":
            splits = text.split(sep) if sep else list(text)
            remaining_seps = separators[i+1:]
            break

    # Process splits
    chunks = []
    current = []

    for split in splits:
        if len(split) <= chunk_size:
            current.append(split)
        else:
            # Save accumulated
            if current:
                chunks.append(sep.join(current))
                current = []

            # Recurse on large split
            if remaining_seps:
                chunks.extend(recursive_split(split, chunk_size, remaining_seps))
            else:
                chunks.append(split)

    # Save remaining
    if current:
        chunks.append(sep.join(current))

    return chunks
```

### Q3: "When would you use semantic vs recursive splitting?"

**Good Answer**:
```
Recursive Splitting:
✓ Default choice for production
✓ When: cost-sensitive, real-time, well-structured docs
✓ Performance: O(n log n), no API, ~10ms
✓ Quality: 7/10

Semantic Splitting:
✓ When retrieval quality is critical
✓ When: budget allows, topics change, can batch process
✓ Performance: O(s * embed_time), ~$0.05/doc, ~5 seconds
✓ Quality: 9/10

Decision Process:
1. Start with recursive
2. Measure retrieval accuracy
3. If accuracy < target AND budget allows → try semantic
4. Compare metrics to justify cost
5. Cache embeddings to amortize cost

Example:
- Customer support KB: Use semantic (high value, static)
- Real-time chat: Use recursive (latency critical)
- Code search: Use document-specific (structure matters)
```

### Q4: "Calculate time complexity of semantic splitting"

**Answer**:
```
Given: document with n characters, s sentences

Semantic Splitting Steps:
1. Sentence splitting: O(n)
   - Regex scan through text

2. Embedding generation: O(s * e)
   - s sentences
   - Each embedding takes e time
   - For API: e ≈ 100-200ms
   - Total: 50 sentences × 100ms = 5 seconds

3. Distance calculation: O(s * d)
   - s-1 distance calculations
   - Each distance: O(d) for cosine similarity
   - d = embedding dimensions (384-1536)
   - Total: 50 × 384 = ~20k operations ≈ 1ms

4. Threshold calculation: O(s)
   - Percentile: O(s log s) for sorting
   - Standard deviation: O(s)

5. Chunk creation: O(s)

Overall: O(n + s*e + s*d + s log s)
Dominated by: O(s * e) - embedding generation

Example:
- 1000 words ≈ 50 sentences
- 50 × 100ms = 5 seconds
- vs Recursive: <10ms (500x faster)
```

### Q5: "Design a chunking system for a code search engine"

**Answer**:
```
Requirements:
- Index millions of code files
- Fast search (<100ms)
- Preserve function/class context
- Support multiple languages

Architecture:

1. Language Detection:
   - Use file extension or detect from content
   - Support: Python, JS, Java, Go, etc.

2. Code-Specific Chunking:
   - Don't use character splitting!
   - Use language-aware separators:

   Python: ["\nclass ", "\ndef ", "\n    def ", "\n\n"]
   JS: ["\nclass ", "\nfunction ", "\nexport ", "\n\n"]
   Java: ["\npublic class ", "\nprivate ", "\n\n"]

3. Chunk Strategy:
   - Keep functions complete (don't split mid-function)
   - Include docstrings/comments with function
   - Target: 500-2000 characters per chunk
   - Include 2-3 lines of context before/after

4. Metadata Extraction:
   metadata = {
       'language': 'python',
       'type': 'function',  # or 'class', 'method'
       'name': 'calculate_total',
       'file_path': 'src/utils.py',
       'line_start': 42,
       'line_end': 58
   }

5. Indexing:
   - Use hybrid search:
     * BM25 for exact name matching
     * Semantic embeddings for concept matching
   - Store both in vector DB with metadata

6. Retrieval:
   - Query: "function to calculate user discount"
   - BM25 matches: functions with "calculate" + "discount"
   - Semantic matches: pricing, payment, checkout code
   - Combine with alpha=0.5 (50/50)
   - Return top 10 with metadata

Optimization:
- Cache parsed ASTs
- Batch embed similar files
- Incremental updates (only changed files)
- Multi-language support via abstract interface
```

---

## Key Takeaways for Interview

1. **Know the default**: Recursive character text splitting is the production standard

2. **Explain trade-offs**: Speed vs quality, cost vs accuracy

3. **Think in Big-O**: Be ready to analyze complexity

4. **Consider costs**: Embeddings aren't free ($0.001 per sentence)

5. **Real-world mindset**: "I'd start with recursive, measure, then optimize if needed"

6. **Ask questions**: "What's the budget? Latency requirements? Document type?"

7. **Show depth**: Explain why recursive uses ["\n\n", "\n", " ", ""] hierarchy

8. **Bonus points**: Mention parent-document retrieval pattern

---

## Resources

- **Video**: [The 5 Levels Of Text Splitting For Retrieval](https://www.youtube.com/watch?v=8OJC21T2SL4)
- **GitHub**: [FullStackRetrieval-com/RetrievalTutorials](https://github.com/FullStackRetrieval-com/RetrievalTutorials)
- **Tool**: [ChunkViz](https://www.chunkviz.com/) - Visualize chunking strategies

---

**Created for**: AI Engineer Interview Prep (Jan 8, 2025)
**Focus**: Companies building custom systems (No LangChain)
**Based on**: Greg Kamradt's comprehensive tutorial
**Last Updated**: October 31, 2025
