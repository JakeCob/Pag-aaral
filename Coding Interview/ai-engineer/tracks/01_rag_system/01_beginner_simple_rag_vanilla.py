"""
Simple RAG System - VANILLA PYTHON (No LangChain)
Build RAG from scratch using raw APIs

This version shows you how to build RAG without frameworks.
Perfect for companies that build custom systems!

Dependencies:
- openai (for LLM)
- chromadb (for vector store)
- sentence-transformers (for embeddings)
"""

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
import os


class VanillaDocumentProcessor:
    """
    Document processor without LangChain.
    Chunks text using pure Python.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_text_file(self, file_path: str) -> str:
        """Load text from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text with overlap using pure Python.
        No LangChain TextSplitter needed!
        """
        chunks = []
        start = 0

        while start < len(text):
            # Get chunk
            end = start + self.chunk_size
            chunk = text[start:end]

            # Add to list if not empty
            if chunk.strip():
                chunks.append(chunk.strip())

            # Move start position (with overlap)
            start = end - self.chunk_overlap

        return chunks

    def process_file(self, file_path: str) -> List[str]:
        """Load and chunk a file"""
        text = self.load_text_file(file_path)
        chunks = self.chunk_text(text)
        return chunks


class VanillaEmbeddings:
    """
    Create embeddings without LangChain.
    Uses sentence-transformers directly.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.

        Popular models:
        - all-MiniLM-L6-v2 (fast, good for most tasks)
        - all-mpnet-base-v2 (slower, higher quality)
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Create embedding for a single query"""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


class VanillaVectorStore:
    """
    Vector store without LangChain wrappers.
    Uses ChromaDB directly.
    """

    def __init__(self, collection_name: str = "documents"):
        """Initialize ChromaDB client"""
        # Create client (in-memory for simplicity)
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=".chromadb"
        ))

        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        except:
            # Collection exists, get it
            self.client.delete_collection(collection_name)
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str] = None
    ):
        """Add documents with their embeddings"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search for similar documents.

        Returns:
            {
                "documents": List[str],
                "distances": List[float],
                "ids": List[str]
            }
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return {
            "documents": results["documents"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0]
        }


class VanillaRAG:
    """
    Complete RAG system built from scratch.
    No LangChain, just raw APIs!
    """

    def __init__(self, openai_api_key: str = None):
        """
        Initialize RAG system.

        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        # Set API key
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")

        # Initialize components
        self.processor = VanillaDocumentProcessor()
        self.embeddings = VanillaEmbeddings()
        self.vector_store = VanillaVectorStore()

    def index_document(self, file_path: str):
        """
        Index a document for retrieval.

        Steps:
        1. Load and chunk document
        2. Create embeddings
        3. Store in vector database
        """
        print(f"📄 Loading document: {file_path}")

        # Step 1: Chunk document
        chunks = self.processor.process_file(file_path)
        print(f"✓ Created {len(chunks)} chunks")

        # Step 2: Create embeddings
        print(f"🔢 Creating embeddings...")
        chunk_embeddings = self.embeddings.embed_documents(chunks)
        print(f"✓ Created {len(chunk_embeddings)} embeddings")

        # Step 3: Store in vector DB
        print(f"💾 Storing in vector database...")
        self.vector_store.add_documents(chunks, chunk_embeddings)
        print(f"✓ Indexed successfully!")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User question
            top_k: Number of documents to retrieve

        Returns:
            List of relevant document chunks
        """
        # Create query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Search vector store
        results = self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k
        )

        return results["documents"]

    def generate_answer(self, query: str, context_docs: List[str]) -> str:
        """
        Generate answer using OpenAI API directly.
        No LangChain prompts!

        Args:
            query: User question
            context_docs: Retrieved context documents

        Returns:
            Generated answer
        """
        # Format context
        context = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(context_docs)
        ])

        # Build prompt manually
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
If the answer cannot be found in the context, say "I don't have enough information to answer that."
Always cite which document(s) you used."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        # Call OpenAI API directly
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Extract answer
        answer = response.choices[0].message.content

        return answer

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve + generate.

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            {
                "question": str,
                "answer": str,
                "sources": List[str]
            }
        """
        print(f"\n🔍 Question: {question}")

        # Step 1: Retrieve relevant documents
        print(f"📚 Retrieving top {top_k} documents...")
        context_docs = self.retrieve(question, top_k=top_k)
        print(f"✓ Retrieved {len(context_docs)} documents")

        # Step 2: Generate answer
        print(f"🤖 Generating answer...")
        answer = self.generate_answer(question, context_docs)
        print(f"✓ Answer generated")

        return {
            "question": question,
            "answer": answer,
            "sources": context_docs
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """
    Example of using vanilla RAG system.
    Compare this with LangChain version - same result, more control!
    """

    print("="*70)
    print("Vanilla Python RAG System (No LangChain)")
    print("="*70)

    # Initialize RAG system
    # Make sure to set OPENAI_API_KEY environment variable!
    rag = VanillaRAG()

    # Index a document
    doc_path = "../../data/test_documents/langchain_overview.txt"
    rag.index_document(doc_path)

    # Ask questions
    questions = [
        "What is LangChain?",
        "What are the core components of LangChain?",
        "How does LCEL work?"
    ]

    for question in questions:
        result = rag.query(question, top_k=3)

        print("\n" + "="*70)
        print(f"Q: {result['question']}")
        print("-"*70)
        print(f"A: {result['answer']}")
        print("\n📄 Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source[:200]}...")

    print("\n" + "="*70)
    print("✅ Done! Built entirely without LangChain!")
    print("="*70)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nFor testing without OpenAI, you can:")
        print("1. Use a local LLM (Ollama, llama.cpp)")
        print("2. Mock the generate_answer() function")
        print("3. Focus on the retrieval part only")
    else:
        main()
