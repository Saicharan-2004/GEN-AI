from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.chat_engine import answer_query_to_repo

ask_router = APIRouter()

class QueryRequest(BaseModel):
    repo_url: str
    question: str

@ask_router.post("/ask")
async def ask_repo_question(req: QueryRequest):
    try:
        answer = answer_query_to_repo(req.repo_url, req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))