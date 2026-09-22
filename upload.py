import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

loader = PyPDFLoader("handbook.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
splits = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-005", 
    google_api_key=os.environ["GEMINI_API_KEY"]
)

PineconeVectorStore.from_documents(splits, embeddings, index_name="sru-handbook")
print("Upload complete via Gemini!")