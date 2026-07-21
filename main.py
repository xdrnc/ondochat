import os
import uuid
import shutil
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_community.llms import HuggingFaceHub
from groq import Groq
from langchain_classic.chains import RetrievalQA

from pymongo import MongoClient

MONGO_URL = os.getenv("MONGO_URL")
client = MongoClient(MONGO_URL)

GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini")

import pickle



# ---------------------------------------------------------
# Basic setup
# ---------------------------------------------------------

app = FastAPI(title="OnDoChat Simplified Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DS_ROOT = os.path.join(BASE_DIR, "ds")
os.makedirs(DS_ROOT, exist_ok=True)

# In-memory user state
user_data = {}


def get_user_folder(user_id: str) -> str:
    folder = os.path.join(DS_ROOT, user_id)
    os.makedirs(folder, exist_ok=True)
    return folder


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

class LoginRequest(BaseModel):
    user_id: Optional[str] = None


class ChatRequest(BaseModel):
    user_id: str
    question: str


# ---------------------------------------------------------
# /login
# ---------------------------------------------------------

@app.post("/login")
def login(req: LoginRequest):
    user_id = req.user_id or str(uuid.uuid4())

    # Initialize user state
    user_data[user_id] = {
        "file_path": None,
        "chunks_path": None,
        "vectorstore": None,
        "retriever": None,
        "init_stage": "idle",
        "init_error": None,
    }

    return {"user_id": user_id, "stage": "idle"}


# ---------------------------------------------------------
# /upload
# ---------------------------------------------------------

@app.post("/upload")
def upload_file(user_id: str = Form(...), file: UploadFile = File(...)):
    if user_id not in user_data:
        return {"status": "error", "message": "Unknown user_id."}

    folder = get_user_folder(user_id)

    # Save uploaded file
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".pdf", ".docx"]:
        return {"status": "error", "message": "Only .pdf or .docx supported."}

    saved_path = os.path.join(folder, file.filename)
    with open(saved_path, "wb") as f:
        f.write(file.file.read())

    # Reset state related to this file
    chunks_path = os.path.join(folder, "chunks.pkl")
    if os.path.exists(chunks_path):
        os.remove(chunks_path)

    user_data[user_id]["file_path"] = saved_path
    user_data[user_id]["chunks_path"] = None
    user_data[user_id]["vectorstore"] = None
    user_data[user_id]["retriever"] = None
    user_data[user_id]["init_stage"] = "idle"
    user_data[user_id]["init_error"] = None

    return {"status": "ok", "user_id": user_id, "file": file.filename}


# ---------------------------------------------------------
# /init (synchronous)
# ---------------------------------------------------------

@app.get("/init")
def init(user_id: str):
    if user_id not in user_data:
        return {"status": "error", "message": "Unknown user_id."}

    folder = get_user_folder(user_id)
    chunks_path = os.path.join(folder, "chunks.pkl")

    # 1. Find uploaded file
    file_path = user_data[user_id].get("file_path")
    if not file_path:
        # Try to discover file from folder
        for f in os.listdir(folder):
            if f.lower().endswith((".pdf", ".docx")):
                file_path = os.path.join(folder, f)
                user_data[user_id]["file_path"] = file_path
                break

    if not file_path:
        return {"status": "error", "message": "No uploaded file found."}

    # 2. If chunks already exist, rebuild vectorstore
    if os.path.exists(chunks_path):
        try:
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever()

            user_data[user_id]["vectorstore"] = vectorstore
            user_data[user_id]["retriever"] = retriever
            user_data[user_id]["chunks_path"] = chunks_path
            user_data[user_id]["init_stage"] = "ready"
            user_data[user_id]["init_error"] = None

            return {"user_id": user_id, "stage": "ready"}

        except Exception as e:
            user_data[user_id]["init_stage"] = "error"
            user_data[user_id]["init_error"] = str(e)
            return {
                "status": "error",
                "message": f"Failed to rebuild vectorstore: {e}",
            }

    # 3. Split PDF into chunks
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        user_data[user_id]["chunks_path"] = chunks_path

    except Exception as e:
        user_data[user_id]["init_stage"] = "error"
        user_data[user_id]["init_error"] = str(e)
        return {"status": "error", "message": f"Failed to split PDF: {e}"}

    # 4. Build embeddings + FAISS
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        user_data[user_id]["vectorstore"] = vectorstore
        user_data[user_id]["retriever"] = retriever
        user_data[user_id]["init_stage"] = "ready"
        user_data[user_id]["init_error"] = None

        return {"user_id": user_id, "stage": "ready"}

    except Exception as e:
        user_data[user_id]["init_stage"] = "error"
        user_data[user_id]["init_error"] = str(e)
        return {
            "status": "error",
            "message": f"Failed to build embeddings/FAISS: {e}",
        }


# ---------------------------------------------------------
# /chat
# ---------------------------------------------------------

@app.post("/chat")
def chat(req: ChatRequest):
    user_id = req.user_id
    question = req.question

    if user_id not in user_data:
        return {"status": "error", "message": "Unknown user_id."}

    retriever = user_data[user_id].get("retriever")
    if retriever is None:
        return {
            "status": "error",
            "message": "Retriever not ready. Call /init first.",
        }

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        You are a helpful assistant. Use the following context to answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )

        answer = completion.choices[0].message["content"]
        sources = [doc.metadata.get("source", "") for doc in docs]

        return {
            "status": "ok",
            "answer": answer,
            "sources": sources,
        }

    except Exception as e:
        return {"status": "error", "message": f"Chat failed: {e}"}


# ---------------------------------------------------------
# /cleanup
# ---------------------------------------------------------

@app.delete("/cleanup")
def cleanup(user_id: str):
    folder = os.path.join(DS_ROOT, user_id)

    if os.path.exists(folder):
        shutil.rmtree(folder)

    if user_id in user_data:
        del user_data[user_id]

    return {"status": "ok", "message": f"Cleaned up user {user_id}."}


# ---------------------------------------------------------
# MONGO TEST ENDPOINTS
# ---------------------------------------------------------
@app.get("/mongo-test")
def mongo_test():
    try:
        client.admin.command("ping")
        return {"status": "ok", "message": "MongoDB connection successful"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/mongo-insert")
def mongo_insert():
    try:
        testcol = client["OnDoChat"]["alextestapi"]
        result = testcol.insert_one({"msg": "hello from backend", "time": "now"})
        return {"status": "ok", "inserted_id": str(result.inserted_id)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/mongo-read")
def mongo_read():
    try:
        testcol = client["OnDoChat"]["alextestapi"]
        docs = list(testcol.find())
        for d in docs:
            d["_id"] = str(d["_id"])
        return {"status": "ok", "docs": docs}
    except Exception as e:
        return {"status": "error", "message": str(e)}
