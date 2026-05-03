
import os
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError("GROQ_API_KEY not found. Please create a .env file and add it.")

llm_model = ChatGroq(model="llama-3.3-70b-versatile")

ollama_model_name = "nomic-embed-text"
embedding_model = OllamaEmbeddings(model=ollama_model_name)


def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)
    return documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def create_vectorstore(uploaded_file):
    documents = load_pdf(uploaded_file)
    chunks = create_chunks(documents)
    return FAISS.from_documents(chunks, embedding_model)


def retrieve_docs_from_uploaded_pdf(query, vectorstore):
    return vectorstore.similarity_search(query, k=4)


def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])


custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Don't provide anything outside the given context.

Question: {question}

Context: {context}

Answer:
"""


def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    response = chain.invoke({"question": query, "context": context})
    return response.content
