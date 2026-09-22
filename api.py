import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing lightweight AI router...")
# Using pure Groq chat model which requires zero local RAM for heavy embeddings!
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an administrative assistant for SR University faculty.\n"
        "Answer questions accurately based on standard SR University faculty guidelines. "
        "Keep answers professional, direct, and conversational."
    )),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        response_text = chain.invoke({"input": req.message})
        return {"reply": response_text}
    except Exception as e:
        return {"reply": "Error generating response. Please check server logs."}