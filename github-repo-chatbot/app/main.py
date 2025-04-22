# from services.chat_engine import answer_query_to_repo

# print(answer_query_to_repo("https://github.com/piyushgarg-dev/genai-cohort.git","what is this repository about?"))
from fastapi import FastAPI
from api.endpoints import ask_router

app = FastAPI(title="GitHub Repo Chatbot")

app.include_router(ask_router, prefix="/api")