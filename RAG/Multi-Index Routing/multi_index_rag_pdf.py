"""
Multi-Index Routing RAG with PDF Documents
==========================================

This notebook shows how to build a Multi-Index RAG system using PDF documents.
We'll load PDFs from different domains and route queries to the appropriate index.

Requirements:
pip install openai chromadb langchain langchain-openai langchain-community pypdf
"""

# ============================================================================
# Part 1: Setup and Dependencies
# ============================================================================

import os
from pathlib import Path
from typing import List, Dict, Literal
from pydantic import BaseModel, Field

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_core.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

print("✓ Dependencies loaded successfully!")

# ============================================================================
# Part 2: Directory Structure for PDF Documents
# ============================================================================

print("\n" + "="*60)
print("RECOMMENDED DIRECTORY STRUCTURE")
print("="*60)
print("""
project_folder/
├── pdf_documents/
│   ├── technical/
│   │   ├── python_guide.pdf
│   │   ├── docker_manual.pdf
│   │   └── api_documentation.pdf
│   ├── medical/
│   │   ├── diabetes_research.pdf
│   │   ├── cardiology_handbook.pdf
│   │   └── pharmacology_guide.pdf
│   └── financial/
│       ├── investment_basics.pdf
│       ├── retirement_planning.pdf
│       └── tax_guide.pdf
└── notebook.ipynb
""")

# ============================================================================
# Part 3: Load PDFs from Different Domains
# ============================================================================

def load_pdfs_from_directory(directory_path: str, domain: str) -> List[Document]:
    """
    Load all PDF files from a directory and add domain metadata.
    
    Args:
        directory_path: Path to directory containing PDFs
        domain: Domain name (technical, medical, financial)
    
    Returns:
        List of Document objects with loaded content
    """
    print(f"\n📂 Loading PDFs from: {directory_path}")
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"⚠️  Directory not found: {directory_path}")
        print(f"Creating sample directory structure...")
        os.makedirs(directory_path, exist_ok=True)
        return []
    
    # Load all PDFs from directory
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    documents = loader.load()
    
    # Add domain metadata to all documents
    for doc in documents:
        doc.metadata["domain"] = domain
        doc.metadata["source_type"] = "pdf"
    
    print(f"✓ Loaded {len(documents)} pages from {domain} PDFs")
    return documents

# ============================================================================
# Part 4: Alternative - Load Individual PDF Files
# ============================================================================

def load_single_pdf(pdf_path: str, domain: str) -> List[Document]:
    """
    Load a single PDF file and add domain metadata.
    
    Args:
        pdf_path: Path to PDF file
        domain: Domain name
    
    Returns:
        List of Document objects (one per page)
    """
    print(f"\n📄 Loading: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  File not found: {pdf_path}")
        return []
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Add metadata
    for doc in documents:
        doc.metadata["domain"] = domain
        doc.metadata["source_type"] = "pdf"
        doc.metadata["filename"] = os.path.basename(pdf_path)
    
    print(f"✓ Loaded {len(documents)} pages")
    return documents

# ============================================================================
# Part 5: Example - Loading PDFs for Each Domain
# ============================================================================

print("\n" + "="*60)
print("LOADING PDF DOCUMENTS")
print("="*60)

# Option 1: Load from directories
technical_docs = load_pdfs_from_directory("pdf_documents/technical", "technical")
medical_docs = load_pdfs_from_directory("pdf_documents/medical", "medical")
financial_docs = load_pdfs_from_directory("pdf_documents/financial", "financial")

# Option 2: Load individual files (use this if you have specific files)
# Uncomment and modify paths as needed:
"""
technical_docs = []
technical_docs.extend(load_single_pdf("path/to/python_guide.pdf", "technical"))
technical_docs.extend(load_single_pdf("path/to/docker_manual.pdf", "technical"))

medical_docs = []
medical_docs.extend(load_single_pdf("path/to/diabetes_study.pdf", "medical"))
medical_docs.extend(load_single_pdf("path/to/cardiology.pdf", "medical"))

financial_docs = []
financial_docs.extend(load_single_pdf("path/to/investing_101.pdf", "financial"))
financial_docs.extend(load_single_pdf("path/to/retirement.pdf", "financial"))
"""

# ============================================================================
# Part 6: Create Sample PDFs (For Demo Purposes)
# ============================================================================

print("\n" + "="*60)
print("DEMO MODE: Using Sample Documents")
print("="*60)
print("Since PDFs may not be available, using sample text documents...")

# Create sample documents for demonstration
if not technical_docs:
    technical_docs = [
        Document(
            page_content="""
            Python Programming Guide
            
            Chapter 1: Introduction to Python
            Python is a high-level, interpreted programming language known for its 
            simplicity and readability. It supports multiple programming paradigms 
            including procedural, object-oriented, and functional programming.
            
            Installing Python:
            1. Download from python.org
            2. Run the installer
            3. Add Python to PATH
            
            Package Management:
            Use pip to install packages: pip install package_name
            Create virtual environments: python -m venv env_name
            """,
            metadata={"domain": "technical", "source": "python_guide.pdf", "page": 1}
        ),
        Document(
            page_content="""
            Docker Containerization Guide
            
            What is Docker?
            Docker is a platform for developing, shipping, and running applications 
            in containers. Containers package an application with all its dependencies.
            
            Basic Commands:
            - docker build -t image_name . : Build an image
            - docker run image_name : Run a container
            - docker ps : List running containers
            - docker stop container_id : Stop a container
            
            Dockerfile Example:
            FROM python:3.9
            WORKDIR /app
            COPY requirements.txt .
            RUN pip install -r requirements.txt
            COPY . .
            CMD ["python", "app.py"]
            """,
            metadata={"domain": "technical", "source": "docker_guide.pdf", "page": 1}
        ),
    ]

if not medical_docs:
    medical_docs = [
        Document(
            page_content="""
            Diabetes Mellitus: A Comprehensive Overview
            
            Definition:
            Diabetes mellitus is a chronic metabolic disorder characterized by 
            elevated blood glucose levels due to defects in insulin secretion, 
            insulin action, or both.
            
            Types of Diabetes:
            1. Type 1 Diabetes: Autoimmune destruction of pancreatic beta cells
            2. Type 2 Diabetes: Insulin resistance and relative insulin deficiency
            3. Gestational Diabetes: Develops during pregnancy
            
            Risk Factors for Type 2 Diabetes:
            - Obesity and sedentary lifestyle
            - Family history
            - Age over 45
            - High blood pressure
            
            Management:
            - Blood glucose monitoring
            - Healthy diet and exercise
            - Medication (oral hypoglycemics or insulin)
            - Regular medical check-ups
            """,
            metadata={"domain": "medical", "source": "diabetes_research.pdf", "page": 1}
        ),
        Document(
            page_content="""
            Cardiovascular Health and Hypertension
            
            Understanding Blood Pressure:
            Blood pressure is the force of blood against artery walls. It's measured 
            in millimeters of mercury (mmHg) with two numbers:
            - Systolic (top number): Pressure when heart beats
            - Diastolic (bottom number): Pressure between beats
            
            Normal vs High Blood Pressure:
            - Normal: Less than 120/80 mmHg
            - Elevated: 120-129/<80 mmHg
            - Stage 1 Hypertension: 130-139/80-89 mmHg
            - Stage 2 Hypertension: ≥140/≥90 mmHg
            
            Complications of Untreated Hypertension:
            - Heart attack and stroke
            - Heart failure
            - Kidney damage
            - Vision problems
            
            Lifestyle Modifications:
            - Reduce sodium intake
            - Regular exercise
            - Maintain healthy weight
            - Limit alcohol consumption
            """,
            metadata={"domain": "medical", "source": "cardiology_handbook.pdf", "page": 1}
        ),
    ]

if not financial_docs:
    financial_docs = [
        Document(
            page_content="""
            Investment Fundamentals
            
            Chapter 3: Understanding Index Funds
            
            What are Index Funds?
            Index funds are mutual funds or ETFs designed to track the performance 
            of a specific market index, such as the S&P 500 or NASDAQ-100.
            
            Advantages of Index Funds:
            1. Low fees: Typically 0.03% to 0.20% expense ratio
            2. Diversification: Instant exposure to hundreds or thousands of stocks
            3. Consistent returns: Match market performance
            4. Tax efficiency: Lower turnover means fewer taxable events
            5. Simplicity: Easy to understand and manage
            
            Popular Index Funds:
            - S&P 500 Index Funds (Large-cap US stocks)
            - Total Stock Market Index Funds
            - International Index Funds
            - Bond Index Funds
            
            How to Invest:
            1. Open a brokerage account
            2. Choose your index fund
            3. Determine investment amount
            4. Set up automatic investments
            """,
            metadata={"domain": "financial", "source": "investment_basics.pdf", "page": 3}
        ),
        Document(
            page_content="""
            Retirement Planning Guide
            
            401(k) Retirement Plans
            
            What is a 401(k)?
            A 401(k) is an employer-sponsored retirement savings plan that allows 
            employees to save and invest for retirement on a tax-deferred basis.
            
            Key Features:
            - Pre-tax contributions (Traditional 401k) or after-tax (Roth 401k)
            - Employer matching contributions (free money!)
            - 2024 contribution limit: $23,000 ($30,500 if age 50+)
            - Tax-deferred growth
            
            Employer Match Example:
            If employer matches 50% up to 6% of salary:
            - You contribute: $100
            - Employer adds: $50
            - Total: $150 in your account
            
            Important Rules:
            - Vesting schedules may apply to employer contributions
            - Early withdrawal penalties (10%) before age 59½
            - Required Minimum Distributions (RMDs) start at age 73
            
            Investment Options:
            Most 401(k) plans offer:
            - Target-date funds
            - Index funds
            - Actively managed mutual funds
            - Company stock (use cautiously)
            """,
            metadata={"domain": "financial", "source": "retirement_planning.pdf", "page": 5}
        ),
    ]

print(f"✓ Using {len(technical_docs)} technical documents")
print(f"✓ Using {len(medical_docs)} medical documents")
print(f"✓ Using {len(financial_docs)} financial documents")

# ============================================================================
# Part 7: Split Documents into Chunks
# ============================================================================

print("\n" + "="*60)
print("SPLITTING DOCUMENTS INTO CHUNKS")
print("="*60)

# PDFs often have long pages, so we need to split them into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum size of each chunk
    chunk_overlap=200,  # Overlap between chunks to maintain context
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Split each domain's documents
technical_chunks = text_splitter.split_documents(technical_docs)
medical_chunks = text_splitter.split_documents(medical_docs)
financial_chunks = text_splitter.split_documents(financial_docs)

print(f"Technical: {len(technical_docs)} docs → {len(technical_chunks)} chunks")
print(f"Medical: {len(medical_docs)} docs → {len(medical_chunks)} chunks")
print(f"Financial: {len(financial_docs)} docs → {len(financial_chunks)} chunks")

# ============================================================================
# Part 8: Create Vector Indexes for Each Domain
# ============================================================================

print("\n" + "="*60)
print("CREATING VECTOR INDEXES")
print("="*60)

embeddings = OpenAIEmbeddings()

# Create separate vector stores for each domain
tech_vectorstore = Chroma.from_documents(
    documents=technical_chunks,
    embedding=embeddings,
    collection_name="tech_pdfs",
    persist_directory="./chroma_db/tech"
)

medical_vectorstore = Chroma.from_documents(
    documents=medical_chunks,
    embedding=embeddings,
    collection_name="medical_pdfs",
    persist_directory="./chroma_db/medical"
)

financial_vectorstore = Chroma.from_documents(
    documents=financial_chunks,
    embedding=embeddings,
    collection_name="financial_pdfs",
    persist_directory="./chroma_db/financial"
)

# Create retrievers
tech_retriever = tech_vectorstore.as_retriever(search_kwargs={"k": 3})
medical_retriever = medical_vectorstore.as_retriever(search_kwargs={"k": 3})
financial_retriever = financial_vectorstore.as_retriever(search_kwargs={"k": 3})

print("✓ Vector indexes created and persisted!")

# ============================================================================
# Part 9: Router Schema and LLM Setup
# ============================================================================

print("\n" + "="*60)
print("SETTING UP QUERY ROUTER")
print("="*60)

class RouteQuery(BaseModel):
    """Route a user query to the most relevant PDF collection."""
    
    datasource: Literal["technical", "medical", "financial"] = Field(
        ...,
        description="""
        Choose the most relevant datasource:
        - technical: Programming, software, DevOps, IT documentation
        - medical: Health, diseases, treatments, medical research
        - financial: Investing, retirement, taxes, personal finance
        """
    )

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create router chain
router_function = convert_pydantic_to_openai_function(RouteQuery)
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

print("✓ Router configured!")

# ============================================================================
# Part 10: Complete Multi-Index RAG System
# ============================================================================

retriever_map = {
    "technical": tech_retriever,
    "medical": medical_retriever,
    "financial": financial_retriever
}

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions based on PDF documents.

Context from PDF documents:
{context}

Question: {question}

Provide a detailed answer based on the information in the PDF documents above. 
If the answer isn't in the documents, say so.

Answer:""")

def format_docs(docs: List[Document]) -> str:
    """Format documents with source information."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def rag_with_pdfs(question: str, verbose: bool = True) -> Dict:
    """
    Multi-Index RAG pipeline for PDF documents.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print(f"{'='*70}")
    
    # Step 1: Route query
    selected_datasource = router_chain.invoke({"question": question})
    if verbose:
        print(f"\n🔀 Routed to: {selected_datasource.upper()} PDF collection")
    
    # Step 2: Retrieve from selected index
    selected_retriever = retriever_map[selected_datasource]
    retrieved_docs = selected_retriever.invoke(question)
    
    if verbose:
        print(f"📚 Retrieved {len(retrieved_docs)} relevant chunks")
        print(f"\nSources:")
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            print(f"  • {source} (Page {page})")
    
    # Step 3: Generate answer
    context = format_docs(retrieved_docs)
    rag_chain = rag_prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })
    
    if verbose:
        print(f"\n💡 Answer:\n{answer}")
    
    return {
        "question": question,
        "route": selected_datasource,
        "answer": answer,
        "sources": [(doc.metadata.get("source"), doc.metadata.get("page")) 
                    for doc in retrieved_docs]
    }

# ============================================================================
# Part 11: Test with Various Questions
# ============================================================================

print("\n" + "="*70)
print("TESTING MULTI-INDEX RAG WITH PDF DOCUMENTS")
print("="*70)

test_questions = [
    "How do I create a Docker container?",
    "What are the symptoms and treatment options for Type 2 diabetes?",
    "Should I invest in index funds for retirement?",
    "Explain Python virtual environments",
    "What is normal blood pressure?",
    "How does a 401(k) employer match work?",
]

results = []
for question in test_questions:
    result = rag_with_pdfs(question, verbose=True)
    results.append(result)
    print("\n" + "-"*70)

# ============================================================================
# Part 12: Advanced Features
# ============================================================================

print("\n" + "="*70)
print("ADVANCED FEATURES")
print("="*70)

print("""
1. METADATA FILTERING:
   You can filter by specific metadata fields:
   
   retriever = vectorstore.as_retriever(
       search_kwargs={
           "k": 5,
           "filter": {"page": {"$gte": 10}}  # Only pages 10+
       }
   )

2. HYBRID SEARCH:
   Combine semantic search with keyword matching:
   - Use both dense (embeddings) and sparse (BM25) retrieval
   - Ensemble retrievers for better results

3. MULTIPLE PDFS PER DOMAIN:
   Load entire directories:
   
   loader = DirectoryLoader(
       "pdf_documents/technical/",
       glob="**/*.pdf",
       loader_cls=PyPDFLoader
   )

4. PERSISTENT STORAGE:
   Vector stores are already persisted in ./chroma_db/
   To reload without re-embedding:
   
   vectorstore = Chroma(
       persist_directory="./chroma_db/tech",
       embedding_function=embeddings,
       collection_name="tech_pdfs"
   )

5. CITATION TRACKING:
   Each answer includes source PDFs and page numbers
   for easy verification and follow-up reading
""")

print("\n✅ Tutorial Complete!")
print("\nNext Steps:")
print("1. Add your own PDF documents to the directories")
print("2. Adjust chunk sizes based on your document structure")
print("3. Experiment with different embedding models")
print("4. Add more domains as needed")
print("5. Implement feedback loops to improve routing accuracy")
