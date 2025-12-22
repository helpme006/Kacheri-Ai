import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import (
    llm_model, 
    embedding_model, 
    load_pdf, 
    create_chunks,
    get_context
)
from langchain_community.vectorstores import FAISS

# Load environment variables (GROQ_API_KEY)
load_dotenv()
if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Please create a .env file with your key.")
    st.stop()

# --- Caching Functions (for efficiency) ---

@st.cache_resource
def get_llm():
    """Returns the (cached) LLM model from the pipeline."""
    return llm_model

@st.cache_resource
def create_vector_store(_uploaded_file):
    """
    Loads, chunks, and embeds the PDF into a FAISS vector store.
    This is cached, so it only runs ONCE per uploaded file.
    """
    if _uploaded_file is None:
        return None
    
    try:
        # 1. Load the PDF
        # We read the file's bytes, then load it.
        file_bytes = _uploaded_file.getvalue()
        documents = load_pdf(file_bytes, filename=_uploaded_file.name)
        
        # 2. Create Chunks
        text_chunks = create_chunks(documents)
        if not text_chunks:
            st.error("Could not extract text from the PDF.")
            return None
        
        # 3. Create FAISS Vector Store
        faiss_db = FAISS.from_documents(text_chunks, embedding_model)
        
        return faiss_db

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# --- New Summary Function ---
def generate_summary(documents, llm):
    """
    Summarization workflow. This BYPASSES the vector store.
    """
    # 1. Get all text
    full_text = get_context(documents) # Re-use get_context to join all doc pages
    
    # 2. Create a summary prompt
    # Note: Groq models have context limits. This might fail on very large PDFs.
    # We are truncating the text to the first ~12,000 characters as a safeguard.
    truncated_text = full_text[:12000] 
    
    summary_prompt = f"""
    Please provide a concise, high-level summary of the following document.
    Focus on the main objectives, key articles, and overall purpose.

    Document Text:
    ---
    {truncated_text}
    ---

    Summary:
    """
    
    # 3. Call LLM directly
    st.info("Summarizing... This may take a moment.")
    response = llm.invoke(summary_prompt)
    return response.content

# --- Streamlit App UI ---

st.set_page_config(page_title="Kacheri Ai ⚖️", layout="wide")
st.title("Kacheri Ai ⚖️")
st.markdown("---")

# File Uploader in the sidebar
with st.sidebar:
    st.header("1. Upload Your PDF")
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type="pdf",
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

if not uploaded_file:
    st.info("Please upload a PDF document in the sidebar to begin.")
    st.stop()

# --- Main App Logic ---

# 1. Load and process the PDF
# This is cached, so it's fast on re-runs
db = create_vector_store(uploaded_file)
if db is None:
    st.error("Failed to process the PDF. Please try another file.")
    st.stop()

# 2. Get the LLM model
llm = get_llm()

# 3. Create two columns for RAG (Chat) and Summary
col1, col2 = st.columns([2, 1])

with col1:
    st.header("2. Ask a Question (RAG)")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_query := st.chat_input("Ask a question about your document..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # This is the RAG pipeline
                retrieved_docs = db.similarity_search(user_query, k=4)
                context = get_context(retrieved_docs)
                
                # Create the prompt
                prompt = custom_prompt_template.invoke({
                    "question": user_query,
                    "context": context
                })
                
                # Get response
                response = llm.invoke(prompt)
                response_content = response.content
            
            st.markdown(response_content)
            # Add AI response to history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

with col2:
    st.header("3. Get a Summary")
    st.write("This will summarize the *entire* document. (Bypasses the RAG pipeline).")
    
    if st.button("Summarize PDF", use_container_width=True):
        # We need to re-load the docs here for the summary function
        # (This is fast because the file is in memory)
        docs = load_pdf(uploaded_file.getvalue(), filename=uploaded_file.name)
        summary = generate_summary(docs, llm)
        st.subheader("Document Summary")
        st.success(summary)