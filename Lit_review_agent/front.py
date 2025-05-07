import streamlit as st
from pydantic import BaseModel
import requests

# ---------------------------
# Define Pydantic models for request/response
# ---------------------------
class TopicRequest(BaseModel):
    interests: str
    field: str

class TopicResponse(BaseModel):
    research_question: str
    outline: list[str]

# Add other agent models similarly...

# ---------------------------
# Agent invocation functions
# ---------------------------
API_BASE = "http://localhost:8000"

def call_agent(agent_name: str, payload: dict) -> dict:
    """
    Generic function to call backend FastAPI agent endpoints
    """
    url = f"{API_BASE}/{agent_name}"
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()

def main():
    st.set_page_config(page_title="Research Paper Assistant", layout="wide")
    st.title("📄 Research Paper Agentic Workflow")

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 'topic'
        st.session_state.data = {}

    # Topic & Scope Step
    if st.session_state.step == 'topic':
        st.header("Step 1: Refine Your Research Topic")
        interests = st.text_input("Enter your broad interests:")
        field = st.selectbox("Select your academic field:", ['Computer Science', 'Biology', 'Engineering', 'Social Sciences'])
        if st.button("Generate Research Question"):
            payload = TopicRequest(interests=interests, field=field).dict()
            result = call_agent('topic_agent', payload)
            st.session_state.data['topic'] = result
            st.session_state.step = 'literature'
            st.write("**Research Question:**", result['research_question'])
            st.write("**Outline:**", result['outline'])

    # Literature Review Step    
    if st.session_state.step == 'literature':
        st.header("Step 2: Literature Review")
        topic = st.session_state.data['topic']['research_question']
        st.write(f"**Research Question:** {topic}")
        if st.button("Fetch & Summarize Papers"):
            payload = { 'question': topic, 'keywords': topic.split() }
            result = call_agent('LitReviewAgent', payload)
            st.session_state.data['literature'] = result
            st.session_state.step = 'hypothesis'
            st.experimental_rerun()

    # Hypothesis Step
    if st.session_state.step == 'hypothesis':
        st.header("Step 3: Formulate Hypotheses")
        lit = st.session_state.data['literature']
        st.write("**Key Themes & Gaps:**")
        st.json(lit)
        if st.button("Generate Hypotheses"):
            payload = { 'themes': lit['themes'], 'gaps': lit['gaps'] }
            result = call_agent('HypothesisAgent', payload)
            st.session_state.data['hypotheses'] = result
            st.session_state.step = 'methodology'
            st.experimental_rerun()

    # Add further steps (Methodology, Data Collection, Writing, Revision, Submission)...
    # Use similar patterns: header, display prior outputs, input for agent invocation, button to continue.

    # Final: show all collected outputs
    if st.session_state.step == 'submission':
        st.header("All Steps Completed")
        st.json(st.session_state.data)
        st.success("Your research workflow is ready for final review and submission!")

if __name__ == '__main__':
    main()
