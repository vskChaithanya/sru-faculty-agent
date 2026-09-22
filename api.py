import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an administrative assistant for SR University faculty. Answer questions accurately based on standard SR University faculty guidelines. Keep answers professional, direct, and conversational."),
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
        print(f"Error: {str(e)}")
        return {"reply": "An error occurred while communicating with the AI model."}