import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
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

# 1. Connect to Cloud Embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HF_TOKEN"],
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        response_text = rag_chain.invoke(req.message)
        return {"reply": response_text}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"reply": "An error occurred while searching the handbook."}