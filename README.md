#Kacheri Ai ⚖️

Kacheri Ai is a Retrieval-Augmented Generation (RAG) application that acts as an "AI Lawyer." It uses a Large Language Model to answer questions based on a legal document (PDF) you provide.

This app is built with a hybrid approach: it uses a local Ollama model for document embeddings and the high-speed Groq API for generating chat responses.

Core Features

Chat with your PDF: Upload any PDF document and ask questions about its content.

High-Accuracy Answers: Get answers based only on the information in the provided document.

Blazing Fast Responses: Powered by the Groq API (llama-3.3-70b-versatile) for near-instant chat generation.

Local & Private Embeddings: Uses a local Ollama model (nomic-embed-text) to create the vector database, so your document's text isn't sent to a third party for indexing.

Tech Stack

Frontend: Streamlit

RAG Framework: LangChain

Chat LLM (Answering): Groq (llama-3.3-70b-versatile)

Embedding LLM (Indexing): Ollama (nomic-embed-text)

Vector Store: FAISS (CPU)

PDF Parsing: PDFPlumber

Environment: Pipenv

How It Works

This project is divided into two main parts:

Indexing (vector_database.py):

A PDF file (e.g., universal_declaration_of_human_rights.pdf) is loaded.

The document is split into smaller, overlapping chunks of text.

The local nomic-embed-text model (via Ollama) converts these chunks into numerical vectors (embeddings).

These embeddings are stored in a local FAISS vector database inside the vectorstore/ directory.

RAG Pipeline (rag_pipeline.py & frontend.py):

A user asks a question in the Streamlit UI.

The local nomic-embed-text model converts this question into an embedding.

FAISS searches the vector database for the document chunks with embeddings most similar to the question's embedding.

These relevant chunks (the "context") and the user's original question are sent to the Groq API.

The Groq model (llama-3.3-70b-versatile) generates a final answer based only on the provided context.

The answer is streamed back to the Streamlit UI for the user to see.

🚀 Setup and Installation

Follow these steps to get the project running on your Mac (M-series).

1. Clone the Repository

git clone [https://github.com/helpmef006/Kacheri-Ai.git](https://github.com/helpmef006/Kacheri-Ai.git)
cd Kacheri-Ai



2. Install System Dependencies (for pyarrow)

You must install these with Homebrew before installing the Python packages.

# Install Homebrew (if you don't have it)
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"

# Install CMake and Apache Arrow
brew install cmake
brew install apache-arrow



3. Install and Set Up Ollama

Download and install Ollama from ollama.com.

Pull the correct embedding model:

ollama pull nomic-embed-text



4. Install Python Dependencies

This project uses pipenv to manage dependencies.

# Install pipenv (if you don't have it)
pip3 install pipenv

# Install all project dependencies
pipenv install



5. Set Up Your API Key

Get your free API key from Groq Console.

Create a file named .env in the Kacheri-Ai folder:

touch .env



Add your API key to this .env file:

GROQ_API_KEY="your-api-key-goes-here"



6. Add Your PDF

Place the PDF file you want to query into the root of the Kacheri-Ai folder.

Open vector_database.py and change the file_path variable to match your PDF's filename. (The default is universal_declaration_of_human_rights.pdf).

▶️ How to Run

Make sure your Ollama application is running in the background.

Step 1: Create the Vector Database

You only need to do this once (or any time you change your PDF).

# Activate the virtual environment
pipenv shell

# Run the database creation script
python vector_database.py



This will create the vectorstore/db_faiss directory.

Step 2: Run the Streamlit App

# (Make sure you are still in the pipenv shell)
streamlit run frontend.py



Your browser will automatically open to http://localhost:8501, and you can start chatting with your "AI Lawyer"!
