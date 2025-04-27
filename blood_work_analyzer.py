# from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader

# from langchain_community.document_loaders.sitemap import SitemapLoader
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
    You are Bloodwork BOT, an intelligent Doctor assistant trained specifically on the official bloodwork from the context provided.

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
]
"""

# Message list
messages = [
    {
        "role": "system",
        "content": system_prompt,
    }
]


def setup_qudrant_db(split_docs, embedder):
    print("🛠 Setting up Qdrant collection and uploading documents...")
    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name="Bloodwork_db",  
        embedding=embedder
    )
    print("✅ Successfully uploaded documents to Qdrant!")

def setup_retriever_db(embedder):
    print("🔎 Setting up retriever from existing collection...")
    retriever = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="Bloodwork_db",  
        embedding=embedder
    )
    return retriever

def doc_generator(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    return docs

def split_text(docs):
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(docs)
    print(f"✅ Finished splitting into {len(texts)} chunks.")
    return texts

# 🧠 CONTEXT RETRIEVAL FUNCTION

def get_context_for_query(query, retriever):
    print("🔎 Retrieving relevant context from Qdrant...")
    relevant_docs = retriever.similarity_search(query)
    context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
    return context_text


# 🚀 MAIN CHATBOT RUN FUNCTION

class bloodworkAssistant:
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
        print("🚀 Bloodwork Bot 🚀")
        print("=" * 60)
        print("\nA RAG-powered assistant for your bloodwork.")
        print("\nType 'exit' to quit the assistant.")
        print("=" * 60 + "\n")

        try:
            while True:
                query = input("➤ Ask about your uploaded bloodwork: ")
                
                if query.lower() in ["exit", "quit"]:
                    print("\n👋 Goodbye! Your Bloodwork Assistant is shutting down.")
                    break

                # Fetch context
                context = get_context_for_query(query, self.retriever)

                # Update conversation
                self.messages.append({"role": "user", "content": query})
                self.messages.append({"role": "assistant", "content": f"Relevant documentation context:\n\n{context}"})

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
    docs = doc_generator("PE0451300080542717_RLS.pdf")  

    # Step 3: Split docs
    split_docs = split_text(docs)

    # Step 4: Upload to Qdrant
    setup_qudrant_db(split_docs, embedder)

    # Step 5: Initialize bot
    chatbot = bloodworkAssistant()

    # # Step 6: Setup retriever
    chatbot.retriever = setup_retriever_db(embedder)

    # # Step 7: Run chatbot
    chatbot.run()

# 🧠 RUN THE WHOLE THING
if __name__ == "__main__":
    setup_and_run()
