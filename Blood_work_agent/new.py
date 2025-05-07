from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
import json
import os
from dotenv import load_dotenv
import shutil
import uuid

# Langchain-related
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# --- ENVIRONMENT SETUP ---
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

# --- FASTAPI APP ---
app = FastAPI()

# --- SYSTEM PROMPT ---
system_prompt = """You are Bloodwork BOT, an intelligent Doctor assistant trained specifically on the official bloodwork from the context provided.

You work in a START → PLAN → ANALYZE → RETRIEVE → ACCUMULATE → OUTPUT workflow when answering user queries.

If no relevant information is found, say that:"I couldn't find relevant information about that in the report"

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
- If no relevant information is found, say: "I couldn't find relevant information about that in the report."

REMEMBER:- you would always use JSON FORMAT for your response:

example:-
{
    "step":"<Any one of the above steps : START,PLAN,ANALYZE,ACCUMULATE,OUTPUT.>"
    "content":"<The content that you generate>"
}

DONOT generate any response such that it gets diverted or altered from the above JSON format.

Example Question and answers:-

1)what does high BP levels signify with respect to my report?

  [
  {
    "step": "plan",
    "content": "First, I'll explain what BP (blood pressure) is, then discuss what 'high' BP means medically, review possible reasons in a report context, and finally summarize potential health implications."
  },
  {
    "step": "analyze",
    "content": "Blood pressure measures the force of blood against artery walls. A higher than normal BP suggests the heart is working harder than it should, possibly due to lifestyle, stress, diet, or underlying health conditions."
  },
  {
    "step": "retrieve",
    "content": "According to medical guidelines, normal BP is around 120/80 mmHg. Values consistently above 130/80 mmHg are considered high (hypertension) and could increase risks for heart disease, stroke, and kidney problems."
  },
  {
    "step": "synthesize",
    "content": "In the context of your report, high BP levels could be a sign of early hypertension. It might suggest a need for lifestyle changes (like diet, exercise, stress management) or further medical evaluation to prevent complications."
  },
  {
    "step": "output",
    "content": "High BP in your report indicates your heart and blood vessels are under extra strain, and it would be wise to discuss these findings with your doctor for personalized advice and possible preventive steps."
  }
]"""

# --- Global State (in-memory) ---
db_collections = {}  # store user-specific retrievers

# --- Request/Response Schemas ---

class ChatRequest(BaseModel):
    query: str
    session_id: str

class ChatResponse(BaseModel):
    responses: List[dict]

class UploadResponse(BaseModel):
    session_id: str
    message: str

# --- Bloodwork Assistant Class ---

class BloodworkAssistant:
    def __init__(self, retriever):
        self.client = client
        self.retriever = retriever
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def get_context_for_query(self, query):
        relevant_docs = self.retriever.similarity_search(query)
        context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
        return context_text

    def chat(self, query):
        # Fetch context
        context = self.get_context_for_query(query)

        # Prepare conversation
        self.messages.append({"role": "user", "content": query})
        self.messages.append({"role": "assistant", "content": f"Relevant documentation context:\n\n{context}"})

        conversation_active = True
        outputs = []
        current_step = None

        while conversation_active:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                messages=self.messages,
            )

            response_content = response.choices[0].message.content

            try:
                parsed_output = json.loads(response_content)
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="Model returned invalid JSON.")

            self.messages.append({"role": "assistant", "content": response_content})

            step = parsed_output.get("step", "").lower()
            outputs.append(parsed_output)

            if step == "output":
                conversation_active = False
            else:
                self.messages.append({"role": "user", "content": "Continue with the next step."})

        return outputs

# --- Utility Functions ---

def process_pdf(file_path, session_id):
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)

    # Embed and Store
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    collection_name = f"Bloodwork_db_{session_id}"
    
    QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name=collection_name,
        embedding=embedder
    )

    retriever = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name=collection_name,
        embedding=embedder
    )

    return retriever

# --- API Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    # Save the file locally
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    session_id = str(uuid.uuid4())
    save_path = f"./uploaded_reports/{session_id}.pdf"

    os.makedirs("./uploaded_reports", exist_ok=True)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process PDF and setup retriever
    retriever = process_pdf(save_path, session_id)

    # Save retriever into memory
    db_collections[session_id] = retriever

    return UploadResponse(
        session_id=session_id,
        message="✅ File uploaded and processed successfully!"
    )

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    session_id = request.session_id

    if session_id not in db_collections:
        raise HTTPException(status_code=404, detail="Session ID not found. Please upload a report first.")

    assistant = BloodworkAssistant(db_collections[session_id])
    outputs = assistant.chat(request.query)
    return {"responses": outputs}
