from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_key)

pdf_path = "A copy of the paper.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Inject into vector store (only once)
# vector_store = QdrantVectorStore.from_documents(
#     documents=split_docs,
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder
# )

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

def translate_query_to_english(query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a translation assistant. Translate the following to English."},
        {"role": "user", "content": query}
    ]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content.strip()

def enhance_query(query: str) -> str:
    messages = [
        {"role": "system", "content": "You are an assistant that rewrites search queries to improve information retrieval."},
        {"role": "user", "content": f"Rewrite this query for better search results: {query}"}
    ]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content.strip()

user_query = input("> ")

# translated_query = translate_query_to_english(user_query)

# enhanced_query = enhance_query(translated_query)

enhanced_query = user_query
print(f"Enhanced Query: {enhanced_query}")

search_results = retriever.max_marginal_relevance_search(
    query=enhanced_query,
    k=5
)

context = "\n\n".join([doc.page_content for doc in search_results])

system_prompt = """
You are a helpful AI assistant that answers questions using provided document context.
Refer to the context and provide a concise, accurate response. If context is missing, reply with "No relevant information found."
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{enhanced_query}"}
]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages
)

try:
    parsed_output = json.loads(response.choices[0].message.content)
except json.JSONDecodeError:
    parsed_output = response.choices[0].message.content

print("\n✅ Answer:")
print(parsed_output)
