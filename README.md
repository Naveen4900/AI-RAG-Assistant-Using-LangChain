![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![LLM](https://img.shields.io/badge/LLM-Mixtral-orange)
![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-purple)
![License](https://img.shields.io/badge/License-IBM-yellow)

# AI RAG Assistant using LangChain

An AI-powered **Retrieval-Augmented Generation (RAG) assistant** that answers user questions using information extracted from uploaded PDF documents.
The system combines **Large Language Models (LLMs)** with **vector embeddings** to retrieve relevant document context and generate accurate responses.

---

## Project Overview

This project implements a **document-based Question Answering system** using a Retrieval-Augmented Generation (RAG) pipeline.

Instead of relying only on pretrained model knowledge, the assistant retrieves relevant information from uploaded documents and provides context-aware answers.

The application includes an interactive web interface where users can upload PDFs and ask questions.

---

## Features

* Document-based Question Answering
* Retrieval-Augmented Generation (RAG)
* Semantic document search using embeddings
* Large Language Model response generation
* Interactive chatbot interface
* PDF document ingestion

---

## Tech Stack

* Python
* LangChain
* IBM watsonx.ai
* Chroma Vector Database
* Gradio (Web Interface)
* Mistral / Mixtral LLM

---

## System Architecture

User Query
->
Embedding Model
->
Vector Database (Chroma)
->
Relevant Document Retrieval
->
Large Language Model
->
Generated Response

---

## How It Works

1. User uploads a PDF document.
2. The document is split into smaller text chunks.
3. Each chunk is converted into embeddings.
4. The embeddings are stored in a vector database.
5. When the user asks a question:

   * The system retrieves relevant document chunks.
   * The retrieved context is passed to the LLM.
6. The LLM generates a response based on the retrieved knowledge.

---

## Installation

Clone the repository

git clone https://github.com/Naveen4900/AI-RAG-Assistant-Using-LangChain.git

Navigate to the project folder

cd AI-RAG-Assistant-Using-LangChain

Install dependencies

pip install -r requirements.txt

---

## Run the Application

python app.py

After running the script, the Gradio interface will launch in your browser.

---

## Example Use Case

Upload a research paper or technical document and ask questions like:

"What are the main conclusions of this paper?"

The system will retrieve relevant sections from the document and generate an answer.

---

## Future Improvements

* Add support for multiple document uploads
* Implement hybrid search (BM25 + embeddings)
* Add conversation memory
* Deploy the application using Docker or cloud services
* Integrate with larger vector databases

---

## Author

Naveen Chowdary
AI/ML  Enthusiast

---

## License

This project is open-source and available under the MIT License.
