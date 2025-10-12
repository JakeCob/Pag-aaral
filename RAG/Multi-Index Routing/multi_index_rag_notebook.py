"""
Multi-Index Routing in RAG - Complete Tutorial
==============================================

This notebook demonstrates how to implement Multi-Index Routing in a RAG system.
We'll build a system that intelligently routes queries to different domain-specific indexes.

Requirements:
pip install openai chromadb langchain langchain-openai langchain-community
"""

# ============================================================================
# Part 1: Setup and Dependencies
# ============================================================================

import os
from typing import List, Dict, Literal
from pydantic import BaseModel, Field

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_core.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# ============================================================================
# Part 2: Create Sample Documents for Different Domains
# ============================================================================

print("Creating sample documents for different knowledge domains...")

# Technical Documentation
tech_docs = [
    Document(
        page_content="Python is a high-level programming language. To install packages, use pip install package_name. Virtual environments help isolate dependencies.",
        metadata={"source": "tech_docs", "topic": "python"}
    ),
    Document(
        page_content="FastAPI is a modern web framework for building APIs with Python. It uses type hints and provides automatic API documentation via Swagger UI.",
        metadata={"source": "tech_docs", "topic": "fastapi"}
    ),
    Document(
        page_content="Docker containers package applications with their dependencies. Use docker build to create images and docker run to start containers.",
        metadata={"source": "tech_docs", "topic": "docker"}
    ),
]

# Medical Information
medical_docs = [
    Document(
        page_content="Diabetes is a metabolic disorder characterized by high blood sugar levels. Type 1 diabetes is autoimmune, while Type 2 is related to insulin resistance.",
        metadata={"source": "medical", "topic": "diabetes"}
    ),
    Document(
        page_content="Hypertension, or high blood pressure, is often called the 'silent killer' because it may have no symptoms. Regular monitoring and lifestyle changes are important.",
        metadata={"source": "medical", "topic": "hypertension"}
    ),
    Document(
        page_content="Antibiotics treat bacterial infections but are ineffective against viral infections like the common cold or flu. Overuse can lead to antibiotic resistance.",
        metadata={"source": "medical", "topic": "antibiotics"}
    ),
]

# Financial Information
financial_docs = [
    Document(
        page_content="Compound interest is interest calculated on both the principal and accumulated interest. The formula is A = P(1 + r/n)^(nt) where A is final amount.",
        metadata={"source": "financial", "topic": "compound_interest"}
    ),
    Document(
        page_content="Index funds track a market index like the S&P 500. They offer diversification and typically have lower fees than actively managed funds.",
        metadata={"source": "financial", "topic": "index_funds"}
    ),
    Document(
        page_content="A 401(k) is a retirement savings plan offered by employers. Contributions are often tax-deductible, and many employers offer matching contributions.",
        metadata={"source": "financial", "topic": "401k"}
    ),
]

# ============================================================================
# Part 3: Create Separate Vector Indexes
# ============================================================================

print("\nCreating separate vector indexes for each domain...")

embeddings = OpenAIEmbeddings()

# Create three separate vector stores
tech_vectorstore = Chroma.from_documents(
    documents=tech_docs,
    embedding=embeddings,
    collection_name="tech_docs"
)

medical_vectorstore = Chroma.from_documents(
    documents=medical_docs,
    embedding=embeddings,
    collection_name="medical_docs"
)

financial_vectorstore = Chroma.from_documents(
    documents=financial_docs,
    embedding=embeddings,
    collection_name="financial_docs"
)

# Create retrievers from vector stores
tech_retriever = tech_vectorstore.as_retriever(search_kwargs={"k": 2})
medical_retriever = medical_vectorstore.as_retriever(search_kwargs={"k": 2})
financial_retriever = financial_vectorstore.as_retriever(search_kwargs={"k": 2})

print("✓ Vector indexes created successfully!")

# ============================================================================
# Part 4: Define the Router Schema
# ============================================================================

print("\nDefining routing schema...")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: Literal["technical", "medical", "financial"] = Field(
        ...,
        description="Given a user question, choose which datasource would be most relevant for answering their question"
    )

# ============================================================================
# Part 5: Create the LLM-Based Router
# ============================================================================

print("Setting up LLM-based router...")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Convert Pydantic model to OpenAI function format
router_function = convert_pydantic_to_openai_function(RouteQuery)

# Create a chain that uses function calling to route queries
router_chain = (
    llm.bind(
        functions=[router_function],
        function_call={"name": "RouteQuery"}
    )
    | PydanticAttrOutputFunctionsParser(
        pydantic_schema=RouteQuery,
        attr_name="datasource"
    )
)

# ============================================================================
# Part 6: Create the Complete Multi-Index RAG System
# ============================================================================

print("Building complete RAG system with routing...")

# Map datasource names to retrievers
retriever_map = {
    "technical": tech_retriever,
    "medical": medical_retriever,
    "financial": financial_retriever
}

# Create the RAG prompt template
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based on the provided context.

Context: {context}

Question: {question}

Answer: Provide a clear and concise answer based on the context above.
""")

def format_docs(docs: List[Document]) -> str:
    """Format documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def multi_index_rag(question: str) -> Dict[str, str]:
    """
    Complete Multi-Index RAG pipeline:
    1. Route the question to the appropriate index
    2. Retrieve relevant documents
    3. Generate answer using LLM
    """
    
    # Step 1: Route the query
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    
    selected_datasource = router_chain.invoke({"question": question})
    print(f"\n🔀 Routing Decision: '{selected_datasource}' index")
    
    # Step 2: Retrieve from selected index
    selected_retriever = retriever_map[selected_datasource]
    retrieved_docs = selected_retriever.invoke(question)
    print(f"📚 Retrieved {len(retrieved_docs)} documents")
    
    # Step 3: Generate answer
    context = format_docs(retrieved_docs)
    
    rag_chain = rag_prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })
    
    print(f"\n💡 Answer: {answer}")
    
    return {
        "question": question,
        "route": selected_datasource,
        "answer": answer,
        "retrieved_docs": [doc.page_content for doc in retrieved_docs]
    }

# ============================================================================
# Part 7: Test the System with Different Queries
# ============================================================================

print("\n" + "="*60)
print("Testing Multi-Index Routing RAG System")
print("="*60)

# Test queries from different domains
test_queries = [
    "How do I install Python packages?",
    "What is Type 2 diabetes?",
    "Explain compound interest to me",
    "What are the benefits of Docker containers?",
    "Should I invest in index funds?",
]

results = []
for query in test_queries:
    result = multi_index_rag(query)
    results.append(result)
    print("\n" + "-"*60 + "\n")

# ============================================================================
# Part 8: Advanced - Adding a Fallback Index
# ============================================================================

print("\n" + "="*60)
print("BONUS: Adding a General Knowledge Fallback")
print("="*60)

# Create a general knowledge index as fallback
general_docs = [
    Document(
        page_content="The Earth orbits the Sun once every 365.25 days. This is why we have leap years every four years.",
        metadata={"source": "general", "topic": "astronomy"}
    ),
    Document(
        page_content="Water boils at 100°C (212°F) at sea level. The boiling point decreases at higher altitudes due to lower air pressure.",
        metadata={"source": "general", "topic": "physics"}
    ),
]

general_vectorstore = Chroma.from_documents(
    documents=general_docs,
    embedding=embeddings,
    collection_name="general_docs"
)

# Update schema to include general option
class RouteQueryWithFallback(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: Literal["technical", "medical", "financial", "general"] = Field(
        ...,
        description="Choose the most relevant datasource. Use 'general' if the question doesn't fit technical, medical, or financial categories."
    )

print("\n✓ System now supports routing to general knowledge as fallback!")

# ============================================================================
# Part 9: Key Takeaways and Best Practices
# ============================================================================

print("\n" + "="*60)
print("KEY TAKEAWAYS")
print("="*60)
print("""
1. Multi-Index Routing improves relevance by searching domain-specific indexes
2. LLM-based routing is flexible and handles complex classification
3. Separate indexes allow different retrieval strategies per domain
4. Always include a fallback/general index for edge cases
5. Monitor routing decisions to improve classification over time

BEST PRACTICES:
- Create clear, distinct domains for each index
- Provide good descriptions in your routing schema
- Use metadata to track routing performance
- Consider hybrid approaches (semantic + keyword)
- Test with diverse queries to validate routing logic
""")

print("\n✅ Tutorial Complete! You now understand Multi-Index Routing in RAG!")
