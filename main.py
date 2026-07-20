import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import uuid
import pymongo
import threading
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
# STAGE 1: load file + split chunks (automatic chain to stage 2)
# ---------------------------------------------------------
def background_stage_1(user_id: str):
    try:
        if user_id not in user_data or not user_data[user_id].get("file_path"):
            user_data[user_id]["init_stage"] = "error"
            user_data[user_id]["init_error"] = "No file uploaded for this user."
            return

        file_path = user_data[user_id]["file_path"]
        if not os.path.exists(file_path):
            user_data[user_id]["init_stage"] = "error"
            user_data[user_id]["init_error"] = "File not found on server."
            return

        # Stage 1: loading_file
        user_data[user_id]["init_stage"] = "loading_file"
        from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = Docx2txtLoader(file_path)

        docs = loader.load()

        # Stage 2: splitting_chunks
        user_data[user_id]["init_stage"] = "splitting_chunks"
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        chunks = splitter.split_documents(docs)

        # Persist chunks to disk to reduce memory footprint
        folder = get_user_folder(user_id)
        chunks_path = os.path.join(folder, "chunks.pkl")

        import pickle
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        user_data[user_id]["chunks_path"] = chunks_path

        # Free docs from memory
        user_data[user_id]["docs"] = None
        del docs

        # Automatically chain to stage 2
        threading.Thread(target=background_stage_2, args=(user_id,)).start()

    except Exception as e:
        user_data[user_id]["init_stage"] = "error"
        user_data[user_id]["init_error"] = str(e)


# ---------------------------------------------------------
# STAGE 2: build embeddings + FAISS + retriever
# ---------------------------------------------------------
def background_stage_2(user_id: str):
    try:
        if user_id not in user_data or not user_data[user_id].get("chunks_path"):
            user_data[user_id]["init_stage"] = "error"
            user_data[user_id]["init_error"] = "Chunks not available for this user."
            return

        chunks_path = user_data[user_id]["chunks_path"]
        if not os.path.exists(chunks_path):
            user_data[user_id]["init_stage"] = "error"
            user_data[user_id]["init_error"] = "Chunks file not found on server."
            return

        import pickle
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        # Stage 3: building_embeddings
        user_data[user_id]["init_stage"] = "building_embeddings"
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        user_data[user_id]["embeddings"] = embeddings

        # Stage 4: building_faiss
        user_data[user_id]["init_stage"] = "building_faiss"
        from langchain_community.vectorstores import FAISS

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        user_data[user_id]["vectorstore"] = vectorstore
        user_data[user_id]["retriever"] = retriever

        # Free chunks and embeddings from memory if you want to be extra cautious
        user_data[user_id]["embeddings"] = None
        del chunks

        # Stage 5: ready
        user_data[user_id]["init_stage"] = "ready"
        user_data[user_id]["init_error"] = None

    except Exception as e:
        user_data[user_id]["init_stage"] = "error"
        user_data[user_id]["init_error"] = str(e)


# ---------------------------------------------------------
# UPLOAD ENDPOINT (starts stage 1, which chains to stage 2)
# ---------------------------------------------------------
@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = "",
):
    if not user_id:
        return {"status": "error", "message": "user_id is required. Call /login first."}

    if user_id not in user_data:
        return {"status": "error", "message": "Unknown user_id. Call /login first."}

    folder = get_user_folder(user_id)
    file_location = os.path.join(folder, file.filename)

    with open(file_location, "wb") as f:
        f.write(await file.read())

    user_data[user_id]["file_path"] = file_location
    user_data[user_id]["init_stage"] = "loading_file"
    user_data[user_id]["init_error"] = None

    # Start Stage 1; Stage 1 will automatically chain to Stage 2
    background_tasks.add_task(background_stage_1, user_id)

    return {
        "status": "ok",
        "user_id": user_id,
        "file_path": file_location,
        "message": "File uploaded. Initialization started in background.",
    }


# ---------------------------------------------------------
# INIT ENDPOINT (detailed progress)
# ---------------------------------------------------------
@app.get("/init")
def init(user_id: str):
    if user_id not in user_data:
        return {"status": "error", "message": "Unknown user_id."}

    stage = user_data[user_id].get("init_stage", "idle")
    error = user_data[user_id].get("init_error")

    response = {"user_id": user_id, "stage": stage}

    if stage == "error" and error:
        response["error"] = error

    return response



# ---------------------------------------------------------
# CHAT ENDPOINT (per-user, uses retriever)
# ---------------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    user_id = request.user_id

    if user_id not in user_data:
        return {"user_id": user_id, "response": "Unknown user_id. Please login first."}

    if user_data[user_id].get("init_stage") != "ready":
        return {
            "user_id": user_id,
            "response": f"Initialization not ready. Current stage: {user_data[user_id].get('init_stage')}",
        }

    retriever = user_data[user_id].get("retriever")
    if retriever is None:
        return {
            "user_id": user_id,
            "response": "Retriever not available. Initialization may have failed.",
        }

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
