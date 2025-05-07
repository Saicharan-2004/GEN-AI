# TOPIC ---> {User's broad topic of interest , Field of study}
# OUTPUT ---> {Focussed research Question, Research Gap, Research Problem and Outline}


from fastapi import APIRouter, HTTPException , FastAPI
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from mem import mem_client
from typing import List
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from groq import Groq
from langchain_core.output_parsers import StrOutputParser


app = FastAPI()

class TopicRequest(BaseModel):
    interests: str = Field(..., description="User's broad interests or keywords")
    field: str      = Field(..., description="Academic field, e.g. Computer Science")

class TopicResponse(BaseModel):
    research_question: str = Field(..., description="Refined research question")
    outline: List[str]     = Field(..., description="High-level outline points")


topic_prompt = PromptTemplate(
    input_variables=["interests", "field"],
    template=(
        "You are a smart topic and scope agent who helps the user identify the best possible research topic of choice and scope.\n"
        "The user is a {field} student who is keenly interested in the following topics: {interests}.\n"
        "You being helpful and smart, will help the user narrow down to the best possible scope of research by generating a relavant research question and outline.\n"
        "Your output should be in the following format:\n"
        "1. Research Question: <Research Question>\n"
        "2. Outline:\n"
        "   - <Outline Point 1>\n"
        "   - <Outline Point 2>\n"
        "etc.\n"
        "Make sure to be concise and clear in your response.\n"
        "Do not include any other information or explanation.\n"
        
    )
)
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatOpenAI(
    model="llama3-70b-8192",
    openai_api_key=groq_api_key,
    openai_api_base="https://api.groq.com/openai/v1"
)

chain = topic_prompt | llm | StrOutputParser()

router = APIRouter(prefix="/topic_agent", tags=["agents"])

@router.post("/", response_model=TopicResponse)
def run_topic_agent(request: TopicRequest):
    try:
        raw = chain.invoke({
            "interests": request.interests,
            "field":     request.field
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    lines = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise HTTPException(status_code=500, detail="Empty response from LLM")

    research_question = lines[0]
    outline= lines[1:]
    memory_entry = {
        "type": "research_topic",
        "field": request.field,
        "interests": request.interests,
        "research_question": research_question,
        "outline": outline
    }
    mem_client.add(
        [memory_entry],user_id = "me123"
    )
    return TopicResponse(
        research_question=research_question,
        outline=outline
    )
