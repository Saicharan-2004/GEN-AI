import streamlit as st
import requests

FASTAPI_URL = "https://chatbot-bloodwork-1.onrender.com"

st.title("🩺 Bloodwork Assistant Dashboard")

# --- Session state ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# --- File upload ---
st.header("Upload your bloodwork report (PDF)")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Upload"):
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        response = requests.post(f"{FASTAPI_URL}/upload", files=files)

        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data["session_id"]
            st.success(f"File uploaded! Session ID: {st.session_state.session_id}")
        else:
            st.error("Upload failed. Please try again.")

# --- Chat Section ---
if st.session_state.session_id:
    st.header("Ask a question about your bloodwork")

    user_query = st.text_input("Enter your question")

    if st.button("Ask"):
        payload = {
            "query": user_query,
            "session_id": st.session_state.session_id
        }
        chat_response = requests.post(f"{FASTAPI_URL}/chat", json=payload)

        if chat_response.status_code == 200:
            responses = chat_response.json()["responses"]

            for r in responses:
                st.markdown(f"**Step: {r['step']}**")
                st.write(r['content'])
                st.divider()
        else:
            st.error("Failed to fetch response. Try again.")
