from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "azure").lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    azure_openai_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    azure_openai_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    database_path: str = os.getenv("DATABASE_PATH", "data/app.db")
    docs_path: str = os.getenv("DOCS_PATH", "data/docs")
    chroma_path: str = os.getenv("CHROMA_PATH", "data/chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "erp_knowledge")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    retrieve_candidates: int = int(os.getenv("RETRIEVE_CANDIDATES", "8"))
    min_retrieval_score: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.35"))
    max_chunk_chars: int = int(os.getenv("MAX_CHUNK_CHARS", "900"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))
    eval_questions_path: str = os.getenv("EVAL_QUESTIONS_PATH", "data/eval/questions.csv")


settings = Settings()
