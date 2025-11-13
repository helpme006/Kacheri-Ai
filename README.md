# KACHERI AI⚖️

This is a Python project for a RAG (Retrieval-Augmented Generation) application called "Kacheri Ai." It acts as an "AI Lawyer" ⚖️ by answering questions about a PDF document you provide.

It uses a local Ollama model (nomic-embed-text) for indexing and the high-speed Groq API (llama-3.3-70b-versatile) for chat responses.

* Table of Contents

1. Environment Setup

2. Running the Project

# 🚀 Environment Setup

Follow these steps to get the project running on your Mac (M-series).

1. Clone the Repository

   git clone [https://github.com/helpmef006/Kacheri-Ai.git](https://github.com/helpmef006/Kacheri-Ai.git)
     cd Kacheri-Ai



2. Install System Dependencies (for pyarrow)

	You must install these with Homebrew before installing the Python packages, or the installation will fail.

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



5. Set Up Your Groq API Key

	Get your free API key from Groq Console.

	Create a file named .env in the Kacheri-Ai folder:

	touch .env



	Add your API key to this .env file. (This file is in .gitignore and will not be uploaded).

	GROQ_API_KEY="your-api-key-goes-here"



6. Add Your PDF

	Place the PDF file you want to query into the root of the Kacheri-Ai folder.

	Open vector_database.py and change the file_path variable to match your PDF's filename. (The default is 					universal_declaration_of_human_rights.pdf).

# ▶️ Running the Project

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

