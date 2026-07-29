import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = Path(__file__).resolve().parent / "vector_db"
COLLECTION_NAME = "bitcoin_rag"

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(f"OPENAI_API_KEY not found. Add it to {PROJECT_ROOT / '.env'}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
