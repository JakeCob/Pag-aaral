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
        self.text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
			length_function=len,
			separators=["\n\n", "\n", ".", " ", ""]
		)

    def process_text_file(self, file_path: str) -> List[str]:
        """
        Read and chunk a text file

        Args:
            file_path: Path to text file

        Returns:
            List of text chunks
        """
        with open(file_path, "r") as file:
            text = file.read()

        chunks = self.text_splitter.split_text(text)

        return chunks


if __name__ == "__main__":
    langchain_test_document = "../data/test_documents/langchain_overview.txt"
    # Process test document
    processor = SimpleDocumentProcessor(chunk_size=100, chunk_overlap=20)
    chunks = processor.process_text_file(langchain_test_document)
    print(f"Created {len(chunks)} chunks")