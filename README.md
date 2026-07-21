# 📘 OnDoChat — Multi‑User Login‑Based RAG System
(test api in: https://ondochat-git-1066973804646.europe-west1.run.app/docs)

https://ondochat.onrender.com/docs (not currently working due to memory issue)

A lightweight, high-performance RAG (Retrieval-Augmented Generation) API built with **FastAPI**, **LangChain**, **FAISS**, and **Groq**. 

OnDoChat features a **lazy-initialization architecture**—it defers document parsing, embedding generation, and vector indexing until the user actually sends their first chat prompt, optimizing system resources and speeding up initial file uploads.

---

## ✨ Features

* **⚡ Fast & Lightweight File Uploads**: Instantly receives PDF files and stores them per user without blocking on heavy processing.
* **🧠 On-Demand / Lazy-Init RAG**: Document chunking (`PDFPlumberLoader`) and vector store indexing (`FAISS` + `sentence-transformers`) are executed only on the first query.
* **⚡ Blazing Fast Generation**: Leverages the **Groq API** for ultra-fast LLM responses.
* **📍 Contextual Grounding**: Restricts responses to provided context with explicit fallback handling ("I don't know") to reduce hallucinations.
* **📂 Multi-Tenant User Contexts**: Separates document storage and vector retrieval by `user_id`.
* **🍃 MongoDB Ready**: Includes health-check integrations for state/session management with MongoDB Atlas.

---

## 🛠️ Tech Stack

* **Framework**: FastAPI
* **LLM Orchestration**: Groq SDK / LangChain
* **Vector Store**: FAISS
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
* **PDF Parser**: `pdfplumber` (LangChain Community Loader)
* **Database**: MongoDB (PyMongo)

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have **Python 3.10+** installed along with `pip`.

### 2. Environment Setup

Clone the repository and install the dependencies from `requirements.txt`:

    git clone [https://github.com/xdrnc/ondochat.git](https://github.com/xdrnc/ondochat.git)
    cd ondochat

    # Create and activate virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate

    # Install dependencies from requirements file
    pip install -r requirements.txt

### 3. Environment Variables

Create a `.env` file in the root directory:

    GROQ_API_KEY=your_groq_api_key_here
    GROQ_MODEL=llama-3.3-70b-versatile  # Or your preferred Groq model
    MONGO_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority

---

## 🏃 Running the Application

Start the development server using `uvicorn`:

    uvicorn main:app --reload

The server will spin up at `[http://127.0.0.1:8000](http://127.0.0.1:8000)`. You can test the endpoints interactively via Swagger UI at `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`.

---

## 📡 API Endpoints

### 1. Upload Document
Uploads a PDF file for a specific user.

* **Endpoint**: `POST /upload`
* **Content-Type**: `multipart/form-data`
* **Body Parameters**:
  * `user_id`: String
  * `file`: File (PDF format)

**Example Request (`curl`):**

    curl -X POST "[http://127.0.0.1:8000/upload](http://127.0.0.1:8000/upload)" \
      -F "user_id=user123" \
      -F "file=@/path/to/document.pdf"

---

### 2. Chat Query
Queries the uploaded document using RAG. If the document hasn't been embedded yet, the server automatically builds the vector store first.

* **Endpoint**: `POST /chat`
* **Content-Type**: `application/json`
* **Body**:

    {
      "user_id": "user123",
      "question": "What are the key takeaways from the document?"
    }

**Example Response:**

    {
      "status": "ok",
      "answer": "The document outlines the Q3 financial goals and highlights...",
      "sources": [
        "ds/user123/document.pdf"
      ]
    }

---

### 3. MongoDB Health Check
Verifies connectivity with the configured MongoDB cluster.

* **Endpoint**: `GET /mongo-test`

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph Upload ["1. Upload Flow (/upload)"]
        A[User Uploads PDF] --> B[Save PDF to User Folder]
        B --> C[Return Quick ACK]
    end

    subgraph Chat ["2. Chat Flow (/chat - Lazy RAG)"]
        D[User Question] --> E{Vector Store Exists?}
        E -- No --> F[Load PDF with PDFPlumber]
        F --> G[Split Chunks & Embed with HuggingFace]
        G --> H[Initialize FAISS Store & Retriever]
        E -- Yes --> H
        H --> I[Retrieve Relevant Context Chunks]
        I --> J[Construct Grounded Prompt]
        J --> K[Query Groq API]
        K --> L[Return Answer + Source Metadata]
    end

---
