import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangSmith configuration
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "day22-lab")
LANGCHAIN_ENDPOINT = os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Models
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")

def check_config():
    print("[OK] Config loaded successfully")
    print(f"   LangSmith project : {LANGCHAIN_PROJECT}")
    print(f"   LLM model         : {LLM_MODEL}")
    print(f"   Embedding model   : {EMBEDDING_MODEL}")
    print(f"   Google API Key set: {'Yes' if GOOGLE_API_KEY else 'No'}")


if __name__ == "__main__":
    check_config()
