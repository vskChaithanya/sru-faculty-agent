from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI()

# Allow the frontend portal to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load AI Resources & Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="faculty_handbook",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}
)



# Replace your local Ollama line with this:
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)


# ==========================================
# NEW: The "Google Auto-Correct" Prompt (Few-Shot)
# ==========================================
correction_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict spelling and grammar correction engine for SR University. "
        "Your ONLY job is to output the corrected text. "
        "DO NOT include any conversational text, greetings, or explanations. "
        "DO NOT change university acronyms like SRU, SRAAP, BOS, or HOD.\n\n"
        "EXAMPLES:\n"
        "Input: 'casul lev policy'\n"
        "Output: 'casual leave policy'\n\n"
        "Input: 'what is the roll of hod'\n"
        "Output: 'what is the role of hod'\n\n"
        "Input: 'matnity leaves'\n"
        "Output: 'maternity leaves'\n\n"
        "Input: 'SRU exam guidlines'\n"
        "Output: 'SRU exam guidelines'"
    )),
    ("human", "Input: '{input}'\nOutput:")
])
correction_chain = correction_prompt | llm | StrOutputParser()

# ==========================================
# EXISTING: The Main Assistant Prompt
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an administrative assistant for SR University faculty.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer ONLY using the provided Context.\n"
        "2. Act as a logical reasoning engine. If the Context says a limit is 12 days, and the user asks for 15, directly deduce the answer (e.g., 'No, you are only entitled to 12 days.').\n"
        "3. Provide direct, conversational answers.\n"
        "4. DO NOT write code or answer general trivia.\n"
        "5. If the Context does not contain the answer, say EXACTLY: 'I do not have that information in the Faculty Handbook.'\n\n"
        "Context:\n{context}"
    )),
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()

# 2. Define API Request Format
class ChatRequest(BaseModel):
    message: str

# 3. Create Endpoint
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    
    # STEP A: Auto-correct the user's spelling (The Google Feature)
    try:
        corrected_query = correction_chain.invoke({"input": req.message}).strip()
        print(f"--- ORIGINAL QUERY ---: {req.message}")
        print(f"--- CORRECTED QUERY ---: {corrected_query}")
    except Exception:
        corrected_query = req.message # Fallback to original if the correction fails

    # STEP B: Search the database using the clean, corrected query
    docs = retriever.invoke(corrected_query)
    context_str = "\n\n".join(d.page_content for d in docs)
    
    # Debug print to verify retrieved context in terminal
    print("--- RETRIEVED CONTEXT ---")
    print(context_str[:300])
    print("-------------------------")
    
    # STEP C: Generate the final answer
    try:
        response_text = chain.invoke({"context": context_str, "input": corrected_query})
        return {"reply": response_text}
    except Exception as e:
        return {"reply": "Error generating response. Please check server logs."}