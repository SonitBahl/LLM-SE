import streamlit as st
from utils.loader import load_pdf_text
from utils.vector_store import create_faiss_index
from components.chatbot import ask_ollama

st.set_page_config(page_title="LLM Semantic Search", layout="wide")
st.title("📄 LLM-Powered Semantic Search Engine")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        raw_text = load_pdf_text(uploaded_file)
        index = create_faiss_index([raw_text])
        st.success("Document indexed. Ask away!")

    user_query = st.text_input("Ask a question about the document:")

    if user_query:
        docs = index.similarity_search(user_query, k=3)
        combined_text = "\n".join([doc.page_content for doc in docs])

        response = ask_ollama(f"Context:\n{combined_text}\nQuestion: {user_query}", st.session_state.chat_history)
        st.session_state.chat_history.append((user_query, response))

    for user_q, answer in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {user_q}")
        st.markdown(f"**Bot:** {answer}")