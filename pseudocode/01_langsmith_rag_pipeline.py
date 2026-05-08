import os
from pathlib import Path
import config # Import config to load environment variables first
import time

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# Import QA_PAIRS
from qa_pairs import QA_PAIRS

# ── 1. LLM and Embeddings ───────────────────────────────────────────────────
# Initialize LLM and Embeddings using config
llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
)


# ── 2. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.
    """
    kb_path = Path("data/knowledge_base.txt")
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found at {kb_path}")
    
    text = kb_path.read_text(encoding="utf-8")

    # Split text with RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    # Build and return the FAISS vectorstore
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ── 3. RAG prompt template ──────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])

# ── 4. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── 5. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    The @traceable decorator sends input/output/latency to LangSmith.
    """
    return chain.invoke(question)

# ── 6. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    try:
        # Build the vectorstore
        vectorstore = build_vectorstore()

        # Build the RAG chain
        chain, retriever = build_rag_chain(vectorstore)

        # Loop through all QA_PAIRS, call ask(), print results
        print(f"Running {len(QA_PAIRS)} questions...")
        for i, qa in enumerate(QA_PAIRS, 1):
            question = qa["question"]
            answer = ask(chain, question)
            print(f"[{i:02d}/{len(QA_PAIRS)}] Q: {question[:60]}")
            print(f"       A: {answer[:100]}\n")
            # Wait to respect rate limit (5 RPM for free tier)
            time.sleep(12)

        # Print confirmation that traces were sent
        print(f"[OK] {len(QA_PAIRS)} traces sent to LangSmith project '{config.LANGCHAIN_PROJECT}'")
        print("   Open https://smith.langchain.com to view traces.")

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("Please check your .env file and ensure API keys are correct.")

if __name__ == "__main__":
    main()
