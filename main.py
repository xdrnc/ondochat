import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import pymongo

# LangChain (new architecture)
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


app = FastAPI()

# -----------------------------
# MongoDB Setup
# -----------------------------
#alextest
#MONGO_URL = os.getenv("MONGO_URL")
#client = pymongo.MongoClient(MONGO_URL)
#db = client["docuchat"]
#conversationcol = db["history"]


def load_memory(session_id):
    return []

#alextest
#    doc = conversationcol.find_one({"session_id": session_id})
#    if not doc:
#        return []
#    conv = doc["conversation"]
#    return [(conv[i], conv[i+1]) for i in range(0, len(conv), 2)]


def save_memory(session_id, user_msg, bot_msg):
    pass
#alextest
#    doc = conversationcol.find_one({"session_id": session_id})
#    if doc:
#        conv = doc["conversation"]
#        conv.extend([user_msg, bot_msg])
#        conversationcol.update_one({"session_id": session_id}, {"$set": {"conversation": conv}})
#    else:
#        conversationcol.insert_one({"session_id": session_id, "conversation": [user_msg, bot_msg]})


# -----------------------------
# Request Model
# -----------------------------
class ChatRequest(BaseModel):
    session_id: str | None = None
    user_input: str
    data_source: str  # path to PDF or DOCX inside Codespaces


# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat")
async def chat(request: ChatRequest):

    # Create or reuse session
    session_id = request.session_id or str(uuid.uuid4())

    # -----------------------------
    # Load document
    # -----------------------------
    if request.data_source.endswith(".pdf"):
        loader = PyPDFLoader(request.data_source)
    else:
        loader = Docx2txtLoader(request.data_source)

    docs = loader.load()

    # -----------------------------
    # Split text
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    chunks = splitter.split_documents(docs)

    # -----------------------------
    # Embeddings
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1")

    # -----------------------------
    # Vector store
    # -----------------------------
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # -----------------------------
    # LLM (Groq)
    # -----------------------------
    llm = ChatGroq(model="openai/gpt-oss-120b")

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

    # -----------------------------
    # Modern RAG Pipeline
    # -----------------------------
    rag_chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    # -----------------------------
    # Run RAG
    # -----------------------------
    result = rag_chain.invoke(request.user_input)
    bot_reply = result.content

    # -----------------------------
    # Save memory
    # -----------------------------
    #alextest
    #save_memory(session_id, request.user_input, bot_reply)

    return {
        "session_id": session_id,
        "response": bot_reply
    }

from fastapi import UploadFile, File

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_location = f"/workspaces/ondochat/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"file_path": file_location}
