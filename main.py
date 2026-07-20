import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
import pymongo

# ---------------------------------------------------------
# GLOBAL STATE (loaded only after /init)
# ---------------------------------------------------------
uploaded_file_path = None
embeddings = None
vectorstore = None
retriever = None

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
# Request Model
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str | None = None
    user_input: str


# ---------------------------------------------------------
# UPLOAD ENDPOINT
# ---------------------------------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global uploaded_file_path

    file_location = f"/workspaces/ondochat/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    uploaded_file_path = file_location
    return {"file_path": file_location}


# ---------------------------------------------------------
# INIT ENDPOINT (heavy work happens here)
# ---------------------------------------------------------
@app.get("/init")
def init():
    global uploaded_file_path, embeddings, vectorstore, retriever

    if uploaded_file_path is None:
        return {"status": "error", "message": "No file uploaded yet."}

    # Heavy imports INSIDE endpoint (Render-safe)
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    # Load document
    loader = PyPDFLoader(uploaded_file_path) if uploaded_file_path.endswith(".pdf") else Docx2txtLoader(uploaded_file_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    chunks = splitter.split_documents(docs)

    # Embeddings (load once)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS (build once)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    return {"status": "ok", "message": "System initialized"}


# ---------------------------------------------------------
# CHAT ENDPOINT (lightweight)
# ---------------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    global retriever

    if retriever is None:
        return {"response": "System not initialized. Call /init first."}

    # Light imports only
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    session_id = request.session_id or str(uuid.uuid4())

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
        result = testcol.insert_one({"msg": "hello from codespace", "time": "now"})
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
