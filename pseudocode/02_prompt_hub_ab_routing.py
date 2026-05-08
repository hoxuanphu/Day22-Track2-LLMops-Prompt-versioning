import os
from pathlib import Path
import config # Import config to load environment variables first
import time
import hashlib

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from langsmith import Client

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

# ── 3. Prompt Hub Setup ─────────────────────────────────────────────────────
# Define two distinct system prompts
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer the question as concisely as possible.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer the question. Provide a detailed, structured answer with bullet points if applicable.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])

def push_prompts_to_hub():
    """
    Push both prompt versions to LangSmith Prompt Hub.
    """
    client = Client()
    
    # We use a unique name or follow the pattern. Let's use 'rag-prompt-v1' and 'rag-prompt-v2'
    # or prefix with user initials or project name to avoid collisions if needed, 
    # but the instructions say 'my-rag-prompt-v1' or similar.
    # Let's use 'day22-rag-prompt-v1' and 'day22-rag-prompt-v2'
    
    try:
        print("Pushing PROMPT_V1 to Prompt Hub...")
        client.push_prompt("day22-rag-prompt-v1", object=PROMPT_V1, description="Concise prompt for RAG")
    except Exception as e:
        print(f"Warning pushing V1: {e}")
        
    try:
        print("Pushing PROMPT_V2 to Prompt Hub...")
        client.push_prompt("day22-rag-prompt-v2", object=PROMPT_V2, description="Structured/Detailed prompt for RAG")
    except Exception as e:
        print(f"Warning pushing V2: {e}")

def pull_prompts_from_hub():
    """
    Pull prompts back from Prompt Hub.
    """
    client = Client()
    print("Pulling prompts from Hub...")
    v1 = client.pull_prompt("day22-rag-prompt-v1")
    v2 = client.pull_prompt("day22-rag-prompt-v2")
    return v1, v2

# ── 4. A/B Routing Logic ────────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Deterministically route 50/50 based on request_id.
    """
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return "prompt-v1" if h % 2 == 0 else "prompt-v2"

# ── 5. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore, prompt):
    """
    Build a LangChain RAG chain using LCEL with a specific prompt.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── 6. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query-ab", tags=["rag", "step2"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    """
    return chain.invoke(question)

# ── 7. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    try:
        # Push prompts (Run once or ignore if already pushed, but let's try to push)
        push_prompts_to_hub()
        
        # Pull prompts
        v1_prompt, v2_prompt = pull_prompts_from_hub()

        # Build the vectorstore
        vectorstore = build_vectorstore()

        # We will build chains dynamically or just use the pulled prompts in the loop
        
        # Loop through all QA_PAIRS, route, call ask(), print results
        print(f"Running {len(QA_PAIRS)} questions with A/B routing...")
        
        # Open log file for evidence
        os.makedirs("evidence", exist_ok=True)
        log_file = open("evidence/02_ab_routing_log.txt", "w", encoding="utf-8")
        
        for i, qa in enumerate(QA_PAIRS, 1):
            question = qa["question"]
            request_id = f"req_{i:03d}" # Synthetic request ID
            
            version = get_prompt_version(request_id)
            
            # Select prompt and label
            if version == "prompt-v1":
                prompt = v1_prompt
                label = "[prompt-v1]"
            else:
                prompt = v2_prompt
                label = "[prompt-v2]"
                
            # Build chain for this specific prompt
            chain = build_rag_chain(vectorstore, prompt)
            
            # Run query
            answer = ask(chain, question)
            
            output_str = f"[{i:02d}/{len(QA_PAIRS)}] {label} Q: {question[:60]}\n       A: {answer[:100].replace('\n', ' ')}\n"
            print(output_str, end="")
            log_file.write(output_str)
            
            # Wait to respect rate limit (5 RPM for free tier)
            time.sleep(12)

        log_file.close()
        print(f"[OK] A/B routing log saved to evidence/02_ab_routing_log.txt")
        print(f"[OK] Traces sent to LangSmith project '{config.LANGCHAIN_PROJECT}'")

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("Please check your .env file and ensure API keys are correct.")

if __name__ == "__main__":
    main()
