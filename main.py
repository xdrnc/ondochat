import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
import pymongo

# ---------------------------------------------------------
# GLOBAL STATE (per-session data)
# ---------------------------------------------------------
session_data = {}  # { session_id: { "file_path", "embeddings", "vectorstore", "retriever" } }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DS_ROOT = os.path.join(BASE_DIR, "ds")
os.makedirs(DS_ROOT, exist_ok=True)

app = FastAPI()

# ---------------------------------------------------------
# MongoDB Setup
# ---------------------------------------------------------
MONGO_URL = os.getenv("MONGO_URL")
client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000)
db = client["OnDoChat"]
conversationcol = db["history"]


# ---------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------
def load_memory(session_id):
    doc = conversationcol.find_one({"session_id": session_id})
    if not doc:
        return []
    conv = doc["conversation"]
    return [(conv[i], conv[i+1]) for i in range(0, len(conv), 2)]


def save_memory(session_id, user_msg, bot_msg):
    doc = conversationcol.find_one({"session_id": session_id})
    if doc:
        conv = doc["conversation"]
        conv.extend([user_msg, bot_msg])
        conversationcol.update_one({"session_id": session_id}, {"$set": {"conversation": conv}})
    else:
        conversationcol.insert_one({"session_id": session_id, "conversation": [user_msg, bot_msg]})


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str | None = None
    user_input: str


class UploadRequest(BaseModel):
    session_id: str | None = None


# ---------------------------------------------------------
# Helper: ensure session folder
# ---------------------------------------------------------
def get_session_folder(session_id: str) -> str:
    folder = os.path.join(DS_ROOT, session_id)
    os.makedirs(folder, exist_ok=True)
    return folder


# ---------------------------------------------------------
# UPLOAD ENDPOINT (per-session)
# ---------------------------------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str | None = None):
    if session_id is None:
        session_id = str(uuid.uuid4())

    folder = get_session_folder(session_id)
    file_location = os.path.join(folder, file.filename)

    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Store file path in session_data
    if session_id not in session_data:
        session_data[session_id] = {}
    session_data[session_id]["file_path"] = file_location

    return {"session_id": session_id, "file_path": file_location}


# ---------------------------------------------------------
# INIT ENDPOINT (per-session heavy work)
# ---------------------------------------------------------
@app.get("/init")
def init(session_id: str):
    if session_id not in session_data or "file_path" not in session_data[session_id]:
        return {"status": "error", "message": "No file uploaded for this session."}

    file_path = session_data[session_id]["file_path"]

    # Heavy imports INSIDE endpoint (Render-safe)
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    # Load document
    loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else Docx2txtLoader(file_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    chunks = splitter.split_documents(docs)

    # Embeddings (per session)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS (per session)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    session_data[session_id]["embeddings"] = embeddings
    session_data[session_id]["vectorstore"] = vectorstore
    session_data[session_id]["retriever"] = retriever

    return {"status": "ok", "message": f"Session {session_id} initialized using {file_path}"}


# ---------------------------------------------------------
# CHAT ENDPOINT (per-session, lightweight)
# ---------------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in session_data or "retriever" not in session_data[session_id]:
        return {"session_id": session_id, "response": "Session not initialized. Upload a file and call /init first."}

    retriever = session_data[session_id]["retriever"]

    # Light imports only
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    llm = ChatGroq(model="openai/gpt-oss-120b")

    prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

    rag_chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    result = rag_chain.invoke(request.user_input)
    bot_reply = result.content

    save_memory(session_id, request.user_input, bot_reply)

    return {"session_id": session_id, "response": bot_reply}


# ---------------------------------------------------------
# CLEANUP ENDPOINT (optional: delete session data)
# ---------------------------------------------------------
@app.delete("/cleanup")
def cleanup(session_id: str):
    # Remove session folder
    folder = os.path.join(DS_ROOT, session_id)
    if os.path.exists(folder):
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except Exception:
                pass
        try:
            os.rmdir(folder)
        except Exception:
            pass

    # Remove in-memory session data
    if session_id in session_data:
        del session_data[session_id]

    # Remove conversation from MongoDB
    conversationcol.delete_one({"session_id": session_id})

    return {"status": "ok", "message": f"Session {session_id} cleaned up"}


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
