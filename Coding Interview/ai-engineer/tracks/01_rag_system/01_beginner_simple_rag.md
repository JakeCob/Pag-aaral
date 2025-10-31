# Challenge: Simple RAG System (Beginner)

**Track**: RAG System
**Difficulty**: Beginner
**Time Estimate**: 45-60 minutes

---

## Problem Statement

Build a basic RAG (Retrieval-Augmented Generation) system that:
1. Accepts a single text file as input
2. Chunks the text into manageable pieces
3. Stores chunks in a vector database (ChromaDB)
4. Retrieves relevant chunks for a given query
5. Generates an answer using the retrieved context

---

## Requirements

### Part A: Document Processing (20 points)

Implement a `SimpleDocumentProcessor` class with:

```python
class SimpleDocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor

        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        pass

    def process_text_file(self, file_path: str) -> List[str]:
        """
        Read and chunk a text file

        Args:
            file_path: Path to text file

        Returns:
            List of text chunks
        """
        pass
```

**Must Handle**:
- Read text file (UTF-8 encoding)
- Chunk text using RecursiveCharacterTextSplitter from LangChain
- Return list of chunks

### Part B: Vector Storage (20 points)

Implement a `VectorStore` class with:

```python
class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize vector store using ChromaDB

        Args:
            collection_name: Name of the collection
        """
        pass

    def add_documents(self, chunks: List[str]) -> None:
        """
        Add document chunks to vector store

        Args:
            chunks: List of text chunks
        """
        pass

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Search for similar chunks

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of most similar chunks
        """
        pass
```

**Must Handle**:
- Initialize ChromaDB client
- Create/get collection
- Add documents with auto-generated IDs
- Query collection and return top k results

### Part C: Answer Generation (20 points)

Implement a `SimpleRAG` class with:

```python
class SimpleRAG:
    def __init__(self, vectorstore: VectorStore, llm):
        """
        Initialize RAG system

        Args:
            vectorstore: Vector store instance
            llm: Language model (OpenAI)
        """
        pass

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer question using RAG

        Args:
            question: User question

        Returns:
            Dictionary with 'answer' and 'sources'
        """
        pass
```

**Must Handle**:
- Retrieve relevant chunks (top 3)
- Build prompt with context
- Generate answer using LLM
- Return answer and source chunks

---

## Constraints

- Use LangChain's RecursiveCharacterTextSplitter
- Use ChromaDB for vector storage
- Use OpenAI for embeddings and LLM (or mock for testing)
- No error handling required (beginner level)
- No conversation memory needed

---

## Example Usage

```python
# Initialize components
processor = SimpleDocumentProcessor(chunk_size=500, chunk_overlap=50)
vectorstore = VectorStore(collection_name="my_docs")

# Process document
chunks = processor.process_text_file("document.txt")
print(f"Created {len(chunks)} chunks")

# Store in vector database
vectorstore.add_documents(chunks)

# Initialize RAG
llm = OpenAI(temperature=0.7)
rag = SimpleRAG(vectorstore, llm)

# Query
result = rag.query("What is the main topic of the document?")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} chunks used")
```

---

## Expected Output

```
Created 15 chunks

Answer: The main topic of the document is...
Sources: 3 chunks used
```

---

## Evaluation Criteria

| Criteria | Points | Description |
|----------|--------|-------------|
| **Functionality** | 40% | All methods work correctly |
| **Code Quality** | 30% | Clean code, proper naming, docstrings |
| **Correctness** | 20% | Handles basic cases, no crashes |
| **Testing** | 10% | Includes basic test (optional) |

---

## Hints

1. **Chunking**: Use `RecursiveCharacterTextSplitter` with `separators=["\n\n", "\n", ".", " "]`
2. **ChromaDB**: Use `chromadb.Client()` for in-memory storage
3. **Prompt Template**: Format as:
   ```
   Context: [chunk1, chunk2, chunk3]

   Question: {question}

   Answer based on the context above:
   ```
4. **OpenAI API**: Use `OpenAI(temperature=0.7)` from LangChain

---

## Sample Test Case

```python
def test_simple_rag():
    # Create test document
    with open("test_doc.txt", "w") as f:
        f.write("""
        Machine learning is a subset of artificial intelligence.
        It involves training algorithms on data to make predictions.
        Deep learning is a type of machine learning using neural networks.
        """)

    # Process
    processor = SimpleDocumentProcessor(chunk_size=100, chunk_overlap=20)
    chunks = processor.process_text_file("test_doc.txt")

    assert len(chunks) > 0, "Should create at least one chunk"

    # Store
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks)

    # Query
    results = vectorstore.search("What is machine learning?", k=2)
    assert len(results) == 2, "Should return 2 results"
    assert "machine learning" in results[0].lower(), "Should find relevant chunk"

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_simple_rag()
```

---

## Extension Ideas (If Time Permits)

- Add error handling for file not found
- Support multiple file formats (txt, md)
- Add chunk metadata (index, source file)
- Display similarity scores

---

## Resources

- **LangChain Docs**: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
- **ChromaDB Quickstart**: https://docs.trychroma.com/getting-started
- **Reference Guide**: `ai-engineer/reference/rag_architecture_guide.md`

---

**Time Yourself!** Try to complete in 45-60 minutes.

Good luck! 🚀
