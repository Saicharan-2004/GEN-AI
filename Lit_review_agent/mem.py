from mem0 import Memory
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = ""

QUADRANT_HOST = "localhost"

NEO4J_URL="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="reform-william-center-vibrate-press-5829"

config = {
    "version": "v1.1",
   # LLM: Use Groq + LLaMA 3
# "llm": {
#     "provider": "openai",  # still use 'openai' because Groq mimics OpenAI API
#     "config": {
#         "api_key": os.getenv("DEEPSEEK_API_KEY"),
#         "model": "llama3-70b-8192",
#         "deepseek_base_url": "https://api.deepseek.com"  # Custom base URL
#     }
# },

# Embedder: Use a free model (e.g., BGE or Instructor)
"embedder": {
    "provider": "huggingface",  # use local or hosted HF model
    "config": {
        "model": "BAAI/bge-small-en-v1.5"  # or any other free embedding model
    }
}
,
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QUADRANT_HOST,
            "port": 6333,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URL, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
}

mem_client = Memory.from_config(config)