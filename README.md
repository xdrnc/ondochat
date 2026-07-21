# 📘 OnDoChat — Multi‑User Login‑Based RAG System
(test api in: https://ondochat-git-1066973804646.europe-west1.run.app/docs)

https://ondochat.onrender.com/docs (not currently working due to memory issue)

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

## 🔄 Checking Initialization Status

After uploading a document, OnDoChat begins a multi‑stage background initialization process.  
Because this work happens asynchronously, the client should **poll the `/init` endpoint** to check the current status.

### **Endpoint**
```
GET /init?user_id=<your_user_id>
```

### **Example Response**
```json
{
  "user_id": "alexgc",
  "stage": "building_faiss"
}
```

### **Possible Stages**
- `idle` — no file uploaded yet  
- `loading_file` — reading the uploaded document  
- `splitting_chunks` — breaking the document into chunks  
- `building_embeddings` — generating vector embeddings  
- `building_faiss` — constructing the FAISS index  
- `ready` — initialization complete  
- `error` — something went wrong (see `error` field)

### **When can you start chatting?**
You can call `/chat` **only when**:
```
"stage": "ready"
```

Before that, `/chat` will return a message indicating the system is still initializing.


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

