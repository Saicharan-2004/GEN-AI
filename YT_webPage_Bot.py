from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
# from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Load Gemini API key
# gemini_api = os.getenv('GEMINI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
# Initialize OpenAI client (for Gemini)
client = Groq(api_key=groq_api_key)

# System prompt (defining ChaiBot behavior)
system_prompt = """
    You are ChaiBot, an intelligent documentation assistant trained specifically on the official ChaiCode documentation from the context provided.

You work in a START → PLAN → ANALYZE → RETRIEVE → ACCUMULATE → OUTPUT workflow when answering user queries.

You will ONLY be using the context retrieved from the ChaiCode documentation to answer the user questions. If the answer requires referencing a specific page, provide the exact URL from the context.

If no relevant information is found, say that:"I couldn't find relevant information about that in the ChaiCode docs"

The workflow for the user query is as follows:

1)START
-You would start by greeting the user and asking him what the query is.

2) PLAN
- Start by analyzing the user query in a well structured and refined way.
- If the question is complex and involves more subparts then breakdown the question to retrieve smaller questions and then proceed further.
- Identify and streamline the ideas and key points addressed by the user.
- You can If required according to your understanding can break the question using step-back prompting or chain of thought according to your convinence.
- Try to understand what the user is trying to convey and REMEMBER you can ask follow-up questions to the user if necessary.

3)ANALYZE
- Look through the retrieved context from the above step.
- Identify the most relevant pieces of information from the above context.
- Find other relevant information that might be related to the query from the previous knowledge you have and the current context.
- Consider how different documents might relate to each other and if required use multiple documents to answer the query and use the context from multiple queries to generate optimal analysis for best results.

4)ACCUMULATE
- Use the exact content from the documentation as much as possible.
- Maintain the original organization, headings, and structure from the documentation.
- Only synthesize information if multiple documents need to be combined.
- Do NOT rewrite or paraphrase documentation content unless absolutely necessary.
- Do NOT try to alter the understanding given in the documentation if in case it does not match with your understanding.

5)OUTPUT
- Reproduce the exact content from the documentation as your primary response.
- Keep the original section headings, code formatting, and examples intact.
- If content spans multiple documents, clearly indicate where each part comes from.
- Always include source URLs and if possible give a brief descriptive answer for the same.

When answering the queries:
- Use the exact content from the documentation rather than summarizing or paraphrasing it, especially for code examples, step-by-step instructions, and technical details.
- Always strive to provide clear, complete, and detailed explanations which help the user to get the maximum possible satisfaction.
- Do not skip steps in your thinking or your output — be explicit and provide thorough reasoning at each stage by mentioning accordingly what you are thinking at every point of time.
- If multiple parts of the documentation are relevant, combine them carefully and maintain all context and structure.
- If the answer requires referencing a specific page, provide the exact URL from the documentation context with a small brief of why you are redirecting the user to the given URL.
- If the code is available than add the code also.
- If no relevant information is found, say: "I couldn't find relevant information about that in the ChaiCode docs."

REMEMBER:- you would always use JSON FORMAT for your response:

example:-
{
    "step":"<Any one of the above steps : START,PLAN,ANALYZE,ACCUMULATE,OUTPUT.>"
    "content":"<The content that you generate>"
}

DONOT generate any response such that it gets diverted or altered from the above JSON format.

Example Question and answers:-

1)How can I build a chatbot that answers questions based on the contents of a GitHub repository using LangChain?
  
  Output:-
  {
    "step": "plan",
    "content": "To build a chatbot that answers questions based on a GitHub repo using LangChain, I'll need to (1) fetch the repo contents, (2) process and chunk the text, (3) store it in a vector store, (4) create a retriever, and (5) set up a conversational chain using LangChain with a language model and retriever."
  },
  Output:-
  {
    "step": "analyze",
    "content": "The main components involved are GitHub integration (to get the code), text processing (to chunk and clean data), vector storage (like FAISS or Chroma), and LangChain's retrieval-based QA chain."
  },
  Output:-
  {
    "step": "retrieve",
    "content": "I'll use LangChain's `GitHubRepoLoader` to fetch repo contents and then `RecursiveCharacterTextSplitter` to split text. For storage, I'll choose `FAISS` as the vector store and use `OpenAIEmbeddings` to embed the chunks."
  },
  Output:-
  {
    "step": "accumulate",
    "content": "Once the repo contents are embedded and stored in FAISS, I can use LangChain's `RetrievalQA` chain with an LLM like `ChatOpenAI`. This setup allows the chatbot to respond to questions by retrieving relevant chunks from the repo and generating responses."
  },
  Output:-
  {
    "step": "output",
    "content": "You can now create an API using FastAPI where the user submits a question, and the chatbot answers it using LangChain's QA chain with the GitHub repo as the knowledge base."
  }

2)How does the self-attention work in a Transformer? 
   
  Output:-
  {
    "step": "plan",
    "content": "To explain self-attention, I'll first define what it is, then describe how queries, keys, and values work, walk through the computation steps, and finally explain why it’s useful in Transformers."
  },
  Output:-
  {
    "step": "analyze",
    "content": "Self-attention allows each token in a sequence to weigh the importance of every other token. It's based on computing attention scores using queries, keys, and values derived from the input embeddings."
  },
  Output:-
  {
    "step": "retrieve",
    "content": "The formula for scaled dot-product attention is: Attention(Q, K, V) = softmax(QKᵀ / √d_k) V. Each token is projected into a query (Q), key (K), and value (V) vector using learned matrices."
  },
  Output:-
  {
    "step": "accumulate",
    "content": "For each token, its query is compared with the keys of all other tokens to get attention scores. These scores determine how much focus (weight) should be given to the corresponding value vectors. The result is a weighted sum of the values, which becomes the new representation of the token."
  },
  Output:-
  {
    "step": "output",
    "content": "In short, self-attention enables each word in a sequence to dynamically attend to other words, capturing context and relationships efficiently, which is key to the Transformer's power."
  }
"""

# Message list
messages = [
    {
        "role": "system",
        "content": system_prompt,
    }
]

# 🚀 SETUP QDRANT DATABASE FUNCTIONS

def setup_qudrant_db(split_docs, embedder):
    print("🛠 Setting up Qdrant collection and uploading documents...")
    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name="chaicode_docs",  # ⚡ Updated name here
        embedding=embedder
    )
    print("✅ Successfully uploaded documents to Qdrant!")

def setup_retriever_db(embedder):
    print("🔎 Setting up retriever from existing collection...")
    retriever = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="chaicode_docs",  # ⚡ Same name here
        embedding=embedder
    )
    return retriever

# 📄 LOAD AND SPLIT DOCUMENTS

def sitemap_loader(path):
    print("🌐 Loading documents from sitemap...")
    sitemap_loader = SitemapLoader(web_path=path, is_local=True)
    docs = sitemap_loader.load()
    print(f"📄 Loaded {len(docs)} documents from sitemap.")
    return docs

def split_text(docs):
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    for doc in docs:
        if hasattr(doc, "metadata"):
            if isinstance(doc.metadata, dict):
                source = doc.metadata.get("source", "no data found")
            else:
                source = "no data found"
        elif isinstance(doc, dict):
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                source = metadata.get("source", "no data found")
            else:
                source = "no data found"
        else:
            source = "no data found"

        title_segment = source.strip("/").split("/")[-1]
        title = title_segment.replace("-", " ").title()
        
        if hasattr(doc, "page_content"):
            doc.page_content += f"\n\n[{title}]({source})"
        elif isinstance(doc, dict):
            doc["page_content"] += f"\n\n[{title}]({source})"

    texts = text_splitter.split_documents(docs)
    print(f"✅ Finished splitting into {len(texts)} chunks.")
    return texts

# 🧠 CONTEXT RETRIEVAL FUNCTION

def get_context_for_query(query, retriever):
    print("🔎 Retrieving relevant context from Qdrant...")
    relevant_docs = retriever.similarity_search(query)
    context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
    return context_text

import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time


# 🚀 MAIN CHATBOT RUN FUNCTION

class ChaiBotAssistant:
    def __init__(self):
        self.client = client
        self.messages = messages
        self.retriever = None  # Will be set later

    def process_response(self, parsed_output):
        """Pretty format the parsed JSON output"""
        step = parsed_output.get("step", "")
        content = parsed_output.get("content", "")
        return f"🧩 **Step: {step.upper()}**\n\n{content}\n"

    def run(self):
        print("\n" + "=" * 60)
        print("🚀 ChaiBot Documentation Assistant 🚀")
        print("=" * 60)
        print("\nA RAG-powered assistant for ChaiCode documentation.")
        print("\nType 'exit' to quit the assistant.")
        print("=" * 60 + "\n")

        try:
            while True:
                query = input("➤ Ask about ChaiCode docs: ")
                
                if query.lower() in ["exit", "quit"]:
                    print("\n👋 Goodbye! ChaiBot Documentation Assistant is shutting down.")
                    break

                # Fetch context
                context = get_context_for_query(query, self.retriever)

                # Update conversation
                self.messages.append({"role": "user", "content": query})
                self.messages.append({"role": "assistant", "content": f"Relevant ChaiCode documentation context:\n\n{context}"})

                conversation_active = True
                current_step = None

                print("\n⏳ Processing your query...\n")
                
                # while conversation_active:
            #           try:
                #         response = self.client.chat.completions.create(
                #             model="llama-3.3-70b-versatile",
                #             response_format={"type": "json_object"},
                #             messages=self.messages,
                #         )

                #         try:
                #             response_content = response.choices[0].message.content
                #             parsed_output = json.loads(response_content)
                            
                #             self.messages.append({
                #                 "role": "assistant",
                #                 "content": response_content
                #             })
                            
                #             step = parsed_output.get("step", "").lower()
                            
                #             if step != current_step:
                #                 current_step = step
                #                 formatted_output = self.process_response(parsed_output)
                #                 print(formatted_output)
                            
                #             if step == "output":
                #                 conversation_active = False
                            
                #         except json.JSONDecodeError:
                #             print("❌ Error: Invalid JSON response from API")
                #             print(f"Raw response: {response_content[:300]}...")
                #             conversation_active = False
                            
                #     except Exception as e:
                #         print(f"❌ Error: {str(e)}")
                #         conversation_active = False

                while conversation_active:
                    try:
                        response = self.client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        response_format={"type": "json_object"},
                        messages=self.messages,
                        )

                        try:
                            response_content = response.choices[0].message.content
                            parsed_output = json.loads(response_content)

                            # Save the assistant's full JSON response
                            self.messages.append({
                                "role": "assistant",
                                "content": response_content
                            })

                            step = parsed_output.get("step", "").lower()
                            content = parsed_output.get("content", "")

                            if step != current_step:
                                current_step = step
                                formatted_output = self.process_response(parsed_output)
                                print(formatted_output)

                            if step == "output":
                                conversation_active = False
                            else:
                                # 👇 Keep conversation flowing by telling the model to continue
                                self.messages.append({
                                    "role": "user",
                                    "content": "Continue with the next step."
                                })

                        except json.JSONDecodeError:
                            print("❌ Error: Invalid JSON response from API")
                            print(f"Raw response: {response_content[:300]}...")
                            conversation_active = False

                    except Exception as e:
                        print(f"❌ Error: {str(e)}")
                        conversation_active = False

                print("\n" + "-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye! ChaiBot Documentation Assistant is shutting down.")



def setup_and_run():
    # Step 1: Embedder
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 2: Load docs
    # docs = sitemap_loader("sitemap_final.xml")  

    # Step 3: Split docs
    # split_docs = split_text(docs)

    # Step 4: Upload to Qdrant
    # setup_qudrant_db(split_docs, embedder)

    # Step 5: Initialize bot
    chatbot = ChaiBotAssistant()

    # # Step 6: Setup retriever
    chatbot.retriever = setup_retriever_db(embedder)

    # # Step 7: Run chatbot
    chatbot.run()

# 🧠 RUN THE WHOLE THING
if __name__ == "__main__":
    setup_and_run()
