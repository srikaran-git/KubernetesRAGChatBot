import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- LOGFIRE ---
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
    GEMINI_API_KEY_1= os.getenv("GEMINI_API_KEY_1")
    GEMINI_API_KEY_2= os.getenv("GEMINI_API_KEY_2")
    GEMINI_API_KEY_3= os.getenv("GEMINI_API_KEY_3")
    GEMINI_API_KEY_4= os.getenv("GEMINI_API_KEY_4")
    GEMINI_API_KEY_5= os.getenv("GEMINI_API_KEY_5")
    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"
    QDRANT_HYBRID_COLLECTION = os.getenv("QDRANT_HYBRID_COLLECTION", "ENTERPRISE_RAG_CHATBOT")
    
    # --- DIRECT GROQ ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GUARDRAIL_API_KEY= os.getenv("GUARDRAIL_API_KEY")
    
    JUDGE_GROQ_API_KEY_1= os.getenv("JUDGE_GROQ_API_KEY_1")
    JUDGE_GROQ_API_KEY_2= os.getenv("JUDGE_GROQ_API_KEY_2")
    JUDGE_GROQ_API_KEY_3= os.getenv("JUDGE_GROQ_API_KEY_3")
    JUDGE_GROQ_API_KEY_4= os.getenv("JUDGE_GROQ_API_KEY_4")
    GROQ_FALLBACK_API_KEY= os.getenv("GROQ_FALLBACK_API_KEY")

    GROQ_MODEL = "openai/gpt-oss-120b"

    # --- PORTKEY ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")

    # These MUST match your Portkey provider/model slugs
    GOOGLE_SLUG = "ragchatbot"
    GROQ_SLUG = "groq-slug"


settings = Settings()