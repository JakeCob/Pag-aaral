"""
Simple RAG System - DIY Version (Beginner)

Fill in the TODOs to complete the implementation.
"""

from typing import List, Dict, Any
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI


class SimpleDocumentProcessor:
    """Process text documents and chunk them"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor

        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        # TODO: Initialize RecursiveCharacterTextSplitter
        # Hint: Use separators=["\n\n", "\n", ".", " ", ""]
        self.text_splitter = None  # Replace with actual implementation

    def process_text_file(self, file_path: str) -> List[str]:
        """
        Read and chunk a text file

        Args:
            file_path: Path to text file

        Returns:
            List of text chunks
        """
        # TODO: Read file content
        text = ""  # Read from file_path

        # TODO: Split text into chunks
        chunks = []  # Use self.text_splitter.split_text(text)

        return chunks


class VectorStore:
    """Vector store using ChromaDB"""

    def __init__(self, collection_name: str = "documents"):
        """
        Initialize vector store

        Args:
            collection_name: Name of the collection
        """
        # TODO: Initialize ChromaDB client
        self.client = None  # Create chromadb.Client()

        # TODO: Create or get collection
        self.collection = None  # Use client.get_or_create_collection()

    def add_documents(self, chunks: List[str]) -> None:
        """
        Add document chunks to vector store

        Args:
            chunks: List of text chunks
        """
        # TODO: Generate IDs for chunks (e.g., "chunk_0", "chunk_1", ...)
        ids = []

        # TODO: Add documents to collection
        # Hint: self.collection.add(documents=chunks, ids=ids)
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
        # TODO: Query the collection
        # Hint: results = self.collection.query(query_texts=[query], n_results=k)

        # TODO: Extract documents from results
        # Hint: results['documents'][0] contains the list of documents

        return []  # Return list of chunks


class SimpleRAG:
    """Simple RAG system"""

    def __init__(self, vectorstore: VectorStore, llm):
        """
        Initialize RAG system

        Args:
            vectorstore: Vector store instance
            llm: Language model
        """
        self.vectorstore = vectorstore
        self.llm = llm

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer question using RAG

        Args:
            question: User question

        Returns:
            Dictionary with 'answer' and 'sources'
        """
        # TODO: Retrieve relevant chunks
        chunks = []  # Use self.vectorstore.search(question, k=3)

        # TODO: Build prompt with context
        prompt = f"""
Context: {chunks}

Question: {question}

Answer based on the context above:
"""

        # TODO: Generate answer using LLM
        answer = ""  # Use self.llm(prompt)

        return {
            "answer": answer,
            "sources": chunks
        }


# Example usage (after completing TODOs)
if __name__ == "__main__":
    # Create test document
    with open("test_doc.txt", "w") as f:
        f.write("""
        Machine learning is a subset of artificial intelligence.
        It involves training algorithms on data to make predictions.
        Deep learning is a type of machine learning using neural networks.
        Neural networks consist of layers of interconnected nodes.
        Each node performs a simple computation.
        """)

    # Initialize components
    processor = SimpleDocumentProcessor(chunk_size=100, chunk_overlap=20)
    vectorstore = VectorStore(collection_name="test_docs")

    # Process document
    chunks = processor.process_text_file("test_doc.txt")
    print(f"Created {len(chunks)} chunks")

    # Store in vector database
    vectorstore.add_documents(chunks)
    print("Documents stored in vector database")

    # Initialize RAG (comment out if OpenAI API not available)
    # llm = OpenAI(temperature=0.7)
    # rag = SimpleRAG(vectorstore, llm)

    # Query
    # result = rag.query("What is machine learning?")
    # print(f"Answer: {result['answer']}")
    # print(f"Sources: {len(result['sources'])} chunks used")
