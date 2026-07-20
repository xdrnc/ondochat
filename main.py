import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
import pymongo

# ---------------------------------------------------------
# GLOBAL STATE (per-user data)
# ---------------------------------------------------------
user_data = {}  # { user_id: { "file_path", "embeddings", "vectorstore", "retriever" } }

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
def load_memory(user_id):
    doc = conversationcol.find_one({"user_id": user_id})
    if not doc:
        return []
    conv = doc["conversation"]
    return [(conv[i], conv[i+1]) for i in range(0, len(conv), 2)]


def save_memory(user_id, user_msg, bot_msg):
    doc = conversationcol.find_one({"user_id": user_id})
    if doc:
        conv = doc["conversation"]
        conv.extend([user_msg, bot_msg])
        conversationcol.update_one({"user_id": user_id}, {"$set": {"conversation": conv}})
    else:
        conversationcol.insert_one({"user_id": user_id, "conversation": [user_msg, bot_msg]})


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    user_input: str


class LoginRequest(BaseModel):
    user_id: str | None = None


# ---------------------------------------------------------
# Helper: ensure user folder
# ---------------------------------------------------------
def get_user_folder(user_id: str) -> str:
    folder = os.path.join(DS_ROOT, user_id)
    os.makedirs(folder, exist_ok=True)
    return folder


# ---------------------------------------------------------
# LOGIN ENDPOINT
# ---------------------------------------------------------
@app.post("/login")
def login(req: LoginRequest):
    user_id = req.user_id or str(uuid.uuid4())
    if user_id not in user_data:
        user_data[user_id] = {}
    return {"user_id": user_id}


# ---------------------------------------------------------
# UPLOAD ENDPOINT (per-user)
# ---------------------------------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = ""):
    if not user_id:
        return {"status": "error", "message": "user_id is required. Call /login first."}

    folder = get_user_folder(user_id)
    file_location = os.path.join(folder, file.filename)

    with open(file_location, "wb") as f:
        f.write(await file.read())

    if user_id not in user_data:
        user_data[user_id] = {}
    user_data[user_id]["file_path"] = file_location

    return {"user_id": user_id, "file_path": file_location}


# ---------------------------------------------------------
# INIT ENDPOINT (per-user heavy work)
# ---------------------------------------------------------
@app.get("/init")
def init(user_id: str):
    if user_id not in user_data or "file_path" not in user_data[user_id]:
        return {"status": "error", "message": "No file uploaded for this user."}

    file_path = user_data[user_id]["file_path"]

    # Heavy imports INSIDE endpoint (Render-safe)
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else Docx2txtLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    user_data[user_id]["embeddings"] = embeddings
    user_data[user_id]["vectorstore"] = vectorstore
    user_data[user_id]["retriever"] = retriever

    return {"status": "ok", "message": f"User {user_id} initialized using {file_path}"}


# ---------------------------------------------------------
# CHAT ENDPOINT (per-user, lightweight)
# ---------------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    user_id = request.user_id

    if user_id not in user_data or "retriever" not in user_data[user_id]:
        return {"user_id": user_id, "response": "User not initialized. Upload a file and call /init first."}

    retriever = user_data[user_id]["retriever"]

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

    save_memory(user_id, request.user_input, bot_reply)

    return {"user_id": user_id, "response": bot_reply}


# ---------------------------------------------------------
# CLEANUP ENDPOINT (per-user)
# ---------------------------------------------------------
@app.delete("/cleanup")
def cleanup(user_id: str):
    folder = os.path.join(DS_ROOT, user_id)
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

    if user_id in user_data:
        del user_data[user_id]

    conversationcol.delete_one({"user_id": user_id})

    return {"status": "ok", "message": f"User {user_id} cleaned up"}


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
