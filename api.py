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
import pypdf
from pymongo import MongoClient
from datetime import datetime


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

class FeedbackRequest(BaseModel):
    user_id: str
    question: str
    feedback_type: str
    suggestion: str = ""

# Connect to MongoDB Atlas (Graceful fallback if environment variable is missing)
mongo_uri = os.environ.get("MONGO_URI")
feedback_collection = None
if mongo_uri:
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["sru_assistant"]
    feedback_collection = db["feedback_logs"]

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
    try:
        # 1. Read the uploaded PDF
        pdf_reader = pypdf.PdfReader(file.file)
        raw_text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                raw_text += page.extract_text() + "\n"

        # 2. Prompt Groq to extract the real data
        prompt_text = f"""
        Extract the Student Names and Marks from the following text.
        Return ONLY a valid JSON array of objects with exactly two keys: "name" and "marks".
        Do not include markdown formatting like ```json or any other text.
        
        Text:
        {raw_text}
        """
        
        # 3. Call your Groq LLM
        messages = [("human", prompt_text)]
        ai_response = llm.invoke(messages)
        
        # 4. Clean the response (strips out any markdown the AI might accidentally add)
        clean_json = ai_response.content.replace("```json", "").replace("```", "").strip()
        
        return {
            "status": "success",
            "security_status": "✅ SANDBOX MODE ACTIVE: Real AI extraction completed. No database changes made.",
            "extracted_data": json.loads(clean_json)
        }
    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return {
            "status": "error",
            "security_status": "❌ Error reading PDF or connecting to AI.",
            "extracted_data": [{"name": "Error", "marks": "See console"}]
        }
# --- ENDPOINT 3: Feedback & Maintenance Tracking ---
@app.post("/feedback")
async def log_feedback(req: FeedbackRequest):
    if feedback_collection is None:
        return {"status": "error", "message": "Database not linked yet."}
        
    doc = req.model_dump()
    doc["timestamp"] = datetime.utcnow().isoformat()
    
    try:
        feedback_collection.insert_one(doc)
        return {"status": "success", "message": "Feedback recorded."}
    except Exception as e:
        print(f"MongoDB Error: {str(e)}")
        return {"status": "error", "message": "Failed to save feedback."}

# --- ENDPOINT 4: Admin Panel Data Fetch ---
@app.get("/api/admin/feedback")
async def get_all_feedback():
    if feedback_collection is None:
        return {"status": "error", "message": "Database not linked yet."}
    
    try:
        # Fetch all logs from MongoDB, sorted by newest first (-1)
        cursor = feedback_collection.find().sort("timestamp", -1)
        logs = []
        for doc in cursor:
            # MongoDB uses 'ObjectId' for _id, which cannot be sent over standard JSON.
            # We must convert it to a normal string first.
            doc["_id"] = str(doc["_id"])
            logs.append(doc)
            
        return {"status": "success", "data": logs}
    except Exception as e:
        print(f"Admin Fetch Error: {str(e)}")
        return {"status": "error", "message": "Failed to fetch feedback logs."}