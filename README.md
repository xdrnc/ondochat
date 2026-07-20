# 📘 OnDoChat — Multi‑User Login‑Based RAG System
(test api in: https://ondochat.onrender.com/docs)

## 🔧 Overview
OnDoChat is a multi‑user Retrieval‑Augmented Generation (RAG) backend built with FastAPI.  
It supports:

- Per‑user login  
- Per‑user document upload  
- Per‑user FAISS vector indexing  
- Per‑user chat history  
- Per‑user cleanup  
- Portable filesystem (Render/GCR‑safe)  
- Lazy heavy imports (fast startup)

This architecture ensures **no cross‑user contamination**, **scalable multi‑user support**, and **clean isolation** of all RAG components.

---

## 🧩 System Flow

### 1️⃣ Login → Get a `user_id`
Every user must start by logging in.

```
POST /login
```

Response:

```json
{
  "user_id": "generated-uuid"
}
```

Use this `user_id` for all future requests.

---

### 2️⃣ Upload a document
Upload a PDF or DOCX for this user.

```
POST /upload?user_id=<your_user_id>
```

Stored under:

```
ds/<user_id>/<filename>
```

---

### 3️⃣ Initialize the RAG system
Build embeddings + FAISS index for this user.

```
GET /init?user_id=<your_user_id>
```

This loads:

- The uploaded document  
- Splits it  
- Embeds it  
- Builds FAISS  
- Stores retriever in memory  

After this step, the user is ready to chat.

---

### 4️⃣ Chat with the document
Ask questions about the uploaded document.

```
POST /chat
{
  "user_id": "<your_user_id>",
  "user_input": "Your question here"
}
```

The system retrieves relevant chunks and answers using Groq LLM.

---

### 5️⃣ Cleanup (optional)
Delete all data for a user.

```
DELETE /cleanup?user_id=<your_user_id>
```

This removes:

- User folder  
- FAISS index  
- Embeddings  
- MongoDB conversation history  
- In‑memory retriever  

---

## 📁 Folder Structure

```
ds/
 ├── <user_id_1>/
 │     └── document.pdf
 ├── <user_id_2>/
 │     └── contract.docx
 └── ...
```

Each user has their own isolated folder.

---

## 🚀 Deployment Notes

- No hardcoded paths  
- Uses `BASE_DIR` for portability  
- Works on Render, GCR, Railway, Docker  
- Heavy imports only inside `/init`  
- Startup is instant (Render‑safe)

---

## 🛠 Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- LangChain
- Groq API key
- MongoDB Atlas or local MongoDB

---

## ▶️ Run Locally

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📌 Notes

- Each user is fully isolated  
- No cross‑user context mixing  
- FAISS and embeddings are per‑user  
- Cleanup is optional but recommended for long‑running deployments  

