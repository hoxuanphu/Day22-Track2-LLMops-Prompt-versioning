import os
from pathlib import Path
import config # Import config to load environment variables first
import time
import asyncio
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from langsmith import Client

# RAGAS imports
import warnings; warnings.filterwarnings("ignore")
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# Import QA_PAIRS
from qa_pairs import QA_PAIRS

# ── 1. LLM and Embeddings with OpenAI & Rate Limiting ───────────────────────
class RateLimitedChatOpenAI(ChatOpenAI):
    """
    Subclass để tự động delay giữa các request.
    Hỗ trợ cả gọi đồng bộ và bất đồng bộ (RAGAS) để không làm nghẽn Event Loop.
    """
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(3)
        return super()._generate(messages, stop, run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.sleep(3)
        return await super()._agenerate(messages, stop, run_manager, **kwargs)

llm = RateLimitedChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=60,
    max_retries=5,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)


# ── 2. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    kb_path = Path("data/knowledge_base.txt")
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found at {kb_path}")
    
    text = kb_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ── 3. Pull Prompts from Hub (or fallback) ──────────────────────────────────
def get_prompts():
    client = Client()
    try:
        print("Pulling prompts from Hub...")
        v1 = client.pull_prompt("day22-rag-prompt-v1")
        v2 = client.pull_prompt("day22-rag-prompt-v2")
        print("[OK] Prompts pulled successfully from Hub")
        return v1, v2
    except Exception as e:
        print(f"Warning pulling prompts: {e}. Using local fallbacks.")
        # Fallback prompts if Hub fails or not pushed yet
        v1 = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the context below to answer the question as concisely as possible.\n\nContext:\n{context}"),
            ("human",  "{question}"),
        ])
        v2 = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the context below to answer the question. Provide a detailed, structured answer with bullet points if applicable.\n\nContext:\n{context}"),
            ("human",  "{question}"),
        ])
        return v1, v2

# ── 4. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore, prompt):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── 5. Generate Answers for Evaluation ──────────────────────────────────────
def generate_responses(vectorstore, prompt, label, cache_file):
    import pickle
    
    # Nếu đã có file cache thì đọc luôn, không gọi API nữa
    if os.path.exists(cache_file):
        print(f"-> Đang tải câu trả lời mẫu cho {label} từ cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
            
    chain, retriever = build_rag_chain(vectorstore, prompt)
    samples = []
    
    print(f"\nGenerating responses for {label}...")
    for i, qa in enumerate(QA_PAIRS, 1):
        question = qa["question"]
        reference = qa["reference"]
        
        # Get contexts separately for RAGAS
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # Get answer
        answer = chain.invoke(question)
        
        print(f"[{i:02d}/{len(QA_PAIRS)}] Q: {question[:50]}...")
        
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference
        )
        samples.append(sample)
        
        # Giảm thời gian chờ xuống 0.5s vì là API trả phí
        time.sleep(0.5)
        
    # Lưu lại cache sau khi tạo xong
    print(f"-> Đang lưu câu trả lời mẫu cho {label} vào cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(samples, f)
        
    return samples

# ── 6. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    try:
        # Build the vectorstore
        vectorstore = build_vectorstore()

        # Get prompts
        v1_prompt, v2_prompt = get_prompts()

        # Generate responses for V1 (Sử dụng cache)
        v1_samples = generate_responses(vectorstore, v1_prompt, "V1 (Concise)", "data/v1_samples.pkl")
        
        # Generate responses for V2 (Sử dụng cache)
        v2_samples = generate_responses(vectorstore, v2_prompt, "V2 (Structured)", "data/v2_samples.pkl")

        # Evaluate V1
        print("\nEvaluating V1...")
        dataset_v1 = EvaluationDataset(samples=v1_samples)
        result_v1 = evaluate(
            dataset_v1,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm,
            embeddings=embeddings
        )

        # Evaluate V2
        print("\nEvaluating V2...")
        dataset_v2 = EvaluationDataset(samples=v2_samples)
        result_v2 = evaluate(
            dataset_v2,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm,
            embeddings=embeddings
        )

        # Print comparison table
        print("\n" + "=" * 60)
        print("  RAGAS Evaluation Results Comparison")
        print("=" * 60)
        
        metrics_names = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
        
        print(f"DEBUG: type(result_v1)={type(result_v1)}")
        print(f"DEBUG: result_v1={result_v1}")
        
        print(f"{'Metric':<20} | {'V1 (Concise)':<15} | {'V2 (Structured)':<15}")
        print("-" * 56)
        
        def get_score(res, metric):
            try:
                # Thử truy cập trực tiếp (nếu là dict hoặc hỗ trợ __getitem__)
                return float(res[metric])
            except Exception:
                try:
                    # Thử truy cập qua thuộc tính .scores (nếu là đối tượng Result của RAGAS)
                    return float(res.scores[metric])
                except Exception:
                    try:
                        # Thử tính trung bình nếu là mảng
                        return float(np.mean(res[metric]))
                    except Exception:
                        return 0.0

        for m in metrics_names:
            v1_score = get_score(result_v1, m)
            v2_score = get_score(result_v2, m)
            print(f"{m:<20} | {v1_score:<15.4f} | {v2_score:<15.4f}")
            
        print("=" * 60)

        # Save results to data/ragas_report.json
        import json
        
        report = {
            "v1": {m: get_score(result_v1, m) for m in metrics_names},
            "v2": {m: get_score(result_v2, m) for m in metrics_names}
        }
        
        os.makedirs("data", exist_ok=True)
        with open("data/ragas_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        print("[OK] Results saved to data/ragas_report.json")
        
        # Deliverable check
        v1_faith = get_score(result_v1, 'faithfulness')
        v2_faith = get_score(result_v2, 'faithfulness')
        
        if v1_faith >= 0.8 or v2_faith >= 0.8:
            print("✅ Target met: Faithfulness score >= 0.8 for at least one prompt version.")
        else:
            print("❌ Target not met: Faithfulness score below 0.8 for both versions.")

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("Please check your .env file and ensure API keys are correct.")

if __name__ == "__main__":
    main()
