
import streamlit as st
from rag_pipeline import (
    answer_query,
    retrieve_docs_from_uploaded_pdf,
    create_vectorstore,
    llm_model
)

st.set_page_config(
    page_title="Kacheri AI",
    page_icon="⚖️",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.main-title {
    text-align: center;
    color: white;
    font-size: 48px;
    font-weight: 800;
}
.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}
.stTextArea textarea {
    border-radius: 14px;
    font-size: 16px;
}
.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 48px;
    font-size: 17px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚖️ Kacheri AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask questions from your uploaded legal PDF</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    with st.spinner("Reading and indexing your PDF..."):
        vectorstore = create_vectorstore(uploaded_file)
    st.success("PDF uploaded and indexed successfully!")

user_query = st.text_area(
    "💬 Enter your prompt:",
    height=150,
    placeholder="Ask anything about your uploaded legal document..."
)

ask_question = st.button("Ask AI Lawyer ⚖️")

if ask_question:
    if uploaded_file and user_query.strip():
        st.chat_message("user").write(user_query)

        retrieved_docs = retrieve_docs_from_uploaded_pdf(user_query, vectorstore)

        response = answer_query(
            documents=retrieved_docs,
            model=llm_model,
            query=user_query
        )

        st.chat_message("AI Lawyer").write(response)

    elif not uploaded_file:
        st.error("Kindly upload a valid PDF file first!")

    else:
        st.warning("Please enter a question.")
