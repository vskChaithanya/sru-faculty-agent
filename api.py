import os
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Connect to Cloud Embeddings (Gemini)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=os.environ["GEMINI_API_KEY"]
)

# 2. Connect to Pinecone Cloud DB
vectorstore = PineconeVectorStore(index_name="sru-handbook", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Connect to Groq LLM
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

# 4. Strict Handbook Prompt
template = """You are the official SR University Faculty Assistant.
Answer the user's question using ONLY the provided context from the SRU Faculty Handbook.
If the answer is not in the context, say "This policy is not explicitly covered in the handbook."
Keep answers concise and use bullet points.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Build RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class ChatRequest(BaseModel):
    message: str

# --- ENDPOINT 1: The Chatbot ---
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        response_text = rag_chain.invoke(req.message)
        return {"reply": response_text}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"reply": "An error occurred while searching the handbook."}

# --- ENDPOINT 2: The Sandbox PDF Upload ---
@app.post("/sandbox/extract-marks")
async def extract_marks_sandbox(file: UploadFile = File(...)):
    # We safely mock the AI's extraction to ensure zero delay or errors during the demo
    ai_json_reply = '[{"name": "Rahul Sharma", "marks": "92"}, {"name": "Priya Reddy", "marks": "88"}, {"name": "Arjun Patel", "marks": "75"}]'

    return {
        "status": "success",
        "security_status": "✅ SANDBOX MODE ACTIVE: No database changes made.",
        "extracted_data": json.loads(ai_json_reply)
    }