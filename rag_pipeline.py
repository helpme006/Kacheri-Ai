import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --- Step 1: Load Environment Variables ---
# This loads the GROQ_API_KEY from your .env file
load_dotenv()

# Check if the API key is available
if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError("GROQ_API_KEY not found in environment variables. Please create a .env file and add it.")

# --- Step 2: Setup LLM (Chat Model) ---
# This model answers the question. It uses the Groq API.
llm_model = ChatGroq(model="llama-3.3-70b-versatile")


# --- Step 3: Setup Embedding Model and Load Database ---

# This model is for searching the database.
# It MUST be the same one you used in vector_database.py
ollama_model_name = "nomic-embed-text"
embedding_model = OllamaEmbeddings(model=ollama_model_name)

# This is the path to your SAVED database
FAISS_DB_PATH = "vectorstore/db_faiss"

# Load the saved FAISS database from disk
try:
    print("Loading vector database...")
    # We pass in the embedding_model to create embeddings for the user's query
    # allow_dangerous_deserialization is required for FAISS with langchain >= 0.2.0
    faiss_db = FAISS.load_local(
        FAISS_DB_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True 
    )
    print("Vector database loaded successfully.")
except RuntimeError as e:
    print(f"Error loading FAISS database: {e}")
    print(f"Did you run 'python vector_database.py' (with nomic-embed-text) first to create the database at '{FAISS_DB_PATH}'?")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()


# --- Step 4: Retrieve Docs ---

def retrieve_docs(query):
    print(f"Retrieving documents for query: {query}")
    # Use the loaded database to find similar documents
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# --- Step 5: Answer Question ---

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    
    print("Generating answer...")
    response = chain.invoke({"question": query, "context": context})
    return response.content # .content is needed for ChatGroq response

# --- Example of running this file directly ---
# if __name__ == "__main__":
#     print("Testing RAG pipeline...")
#     question = "If a government forbids the right to assemble peacefully which articles are violated and why?"
#     retrieved_docs = retrieve_docs(question)
    
#     print(f"Retrieved {len(retrieved_docs)} documents.")
    
#     ai_answer = answer_query(documents=retrieved_docs, model=llm_model, query=question)
#     print(f"\nAI Lawyer: {ai_answer}")