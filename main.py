import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Groq
from groq import Groq

# Mongo (optional, kept clean)
from pymongo import MongoClient

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

app = FastAPI(title="OnDoChat Lazy-Init Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ds")
os.makedirs(DATA_DIR, exist_ok=True)

# In-memory stores
user_files = {}          # user_id -> file_path
user_vectorstores = {}   # user_id -> FAISS
user_retrievers = {}     # user_id -> retriever

# Groq client (create once)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "gpt-4o-mini")

# Mongo (optional)
MONGO_URL = os.getenv("MONGO_URL")
mongo_client = MongoClient(MONGO_URL)

# In-memory chat history (temporary until saved)
user_chats = {}   # user_id -> list of {"role": "...", "content": "..."}


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str
    question: str


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_user_folder(user_id: str) -> str:
    folder = os.path.join(DATA_DIR, user_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def build_vectorstore(user_id: str):
    """Lazy init: build FAISS only when needed."""
    if user_id not in user_files:
        return None

    file_path = user_files[user_id]
    if not os.path.exists(file_path):
        return None

    # Load PDF
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # Embed + FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Save in memory
    user_vectorstores[user_id] = vectorstore
    user_retrievers[user_id] = retriever

    return retriever


# ---------------------------------------------------------
# /upload
# ---------------------------------------------------------
@app.post("/upload")
async def upload_file(user_id: str = Form(...), file: UploadFile = File(...)):
    folder = os.path.join(DATA_DIR, user_id)
    os.makedirs(folder, exist_ok=True)

    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".pdf":
        return {"status": "error", "message": "Only PDF supported."}

    # Use original filename (cleaner, no user_id prefix)
    safe_name = file.filename
    file_path = os.path.join(folder, safe_name)

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Reset user state
    user_files[user_id] = file_path
    user_vectorstores.pop(user_id, None)
    user_retrievers.pop(user_id, None)

    return {"status": "ok", "file_path": file_path}



# ---------------------------------------------------------
# /chat (lazy init)
# ---------------------------------------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id
    question = req.question

    # Lazy init: build vectorstore if missing
    if user_id not in user_retrievers:
        retriever = build_vectorstore(user_id)
        if retriever is None:
            return {
                "status": "error",
                "message": "No PDF found. Upload first."
            }
    else:
        retriever = user_retrievers[user_id]

    try:
        # FIXED: use invoke() instead of get_relevant_documents()
        docs = retriever.invoke(question)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Retriever failed: {e}"
        }

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )

    answer = completion.choices[0].message.content
    sources = list({d.metadata.get("source", "") for d in docs})

    # alex: store conversation in memory
    if user_id not in user_chats:
        user_chats[user_id] = []

    user_chats[user_id].append({"role": "user", "content": question})
    user_chats[user_id].append({"role": "assistant", "content": answer})

    return {
        "status": "ok",
        "answer": answer,
        "sources": sources
    }



# ---------------------------------------------------------
# Optional Mongo endpoints
# ---------------------------------------------------------

@app.get("/mongo-test")
def mongo_test():
    try:
        mongo_client.admin.command("ping")
        return {"status": "ok", "message": "MongoDB connection successful"}
    except Exception as e:
        return {"status": "error", "message": str(e)}