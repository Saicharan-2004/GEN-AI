from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
# from repository_loader import load_repo_documents
from services.repository_loader import load_repo_documents
from services.embeddings import embed_documents
from core.config import GROQ_API_KEY
from groq import Groq

client = Groq(api_key=GROQ_API_KEY)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def answer_query_to_repo(repo_link: str,question:str):
    documents = load_repo_documents(repo_link)
    vec_store = embed_documents(documents)
    relavant_chunks = vec_store.similarity_search(question)
    context = ""
    for doc in relavant_chunks:
        context += "\n\n" + doc.page_content
    system_prompt = """
    You are a helpful AI assistant that answers questions of users for understanding github repositories using provided document context.
    Refer to the context and provide a concise, accurate response. If context is missing, reply with "No relevant information found."
    You make sure that the user is able to understand his question in a proper fashion and is able to navigate through the query asked.
    """
    
    message = [
        {"role":"system","content":system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    response = client.chat.completions.create(
        model = "deepseek-r1-distill-llama-70b",
        messages=message
    )
    return response.choices[0].message.content.strip()